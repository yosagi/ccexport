# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025-2026 @yosagi
"""
JSONL file parsing and message extraction module

Extracts necessary information from Claude Code conversation logs
and provides preprocessing functionality.

Uses DAG-based parser (parser.py) to leverage graph structure
based on uuid/parentUuid.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple

from .parser import SessionParser, SessionDAG, MessageNode, SummaryEntry


class MessageExtractor:
    """Class for extracting messages from JSONL files"""

    def __init__(self, projects_dir: str, config=None):
        """
        Args:
            projects_dir: Path to Claude projects directory
            config: Configuration object
        """
        self.projects_dir = Path(projects_dir)
        self.config = config
        self._parser = SessionParser()
        # DAG cache (project path → DAG)
        self._dag_cache: Dict[str, SessionDAG] = {}

    def list_projects(self) -> List[str]:
        """Get list of available projects"""
        if not self.projects_dir.exists():
            return []

        projects = []
        for project_dir in self.projects_dir.iterdir():
            if project_dir.is_dir():
                jsonl_files = list(project_dir.glob('*.jsonl'))
                if jsonl_files:
                    # Collect cwd from multiple JSONLs and select the shortest one (root)
                    project_name = self._get_project_root_from_jsonls(jsonl_files)
                    if project_name:
                        projects.append(project_name)

        return sorted(projects)

    def _get_project_root_from_jsonls(self, jsonl_files: List[Path]) -> Optional[str]:
        """Collect cwd from multiple JSONLs and return the shortest one (project root)"""
        cwds = set()
        for jsonl_path in jsonl_files[:5]:  # Check only first 5 files (performance optimization)
            cwd = self._get_cwd_from_jsonl(jsonl_path)
            if cwd:
                cwds.add(cwd)

        if not cwds:
            return None

        # Select the shortest path (= topmost directory)
        return min(cwds, key=len)

    def _get_cwd_from_jsonl(self, jsonl_path: Path) -> Optional[str]:
        """Get cwd from first line of JSONL file"""
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if 'cwd' in data:
                            return data['cwd']
                    except json.JSONDecodeError:
                        continue
            return None
        except Exception:
            return None

    def _find_project_dir(self, project_name: str) -> Optional[Path]:
        """
        Search for project directory from project name

        Since Claude Code's directory encoding is complex ('/' → '-', '.' → '-', etc.),
        we scan directories to find a match instead of simple string conversion.

        Args:
            project_name: Project name (e.g., /home/yos/.config/wezterm)

        Returns:
            Path to project directory, or None
        """
        if not self.projects_dir.exists():
            return None

        for project_dir in self.projects_dir.iterdir():
            if not project_dir.is_dir():
                continue
            jsonl_files = list(project_dir.glob('*.jsonl'))
            if jsonl_files:
                # Get actual cwd and compare
                cwd = self._get_cwd_from_jsonl(jsonl_files[0])
                if cwd and cwd == project_name:
                    return project_dir

        return None

    def find_project_by_session(self, session_id: str) -> Optional[tuple]:
        """
        Search for project name from session ID (prefix match)

        Args:
            session_id: Session ID (exact match or prefix match)

        Returns:
            Tuple of (project_name, full_session_id, jsonl_path), or None
        """
        if not self.projects_dir.exists():
            return None

        matches = []
        for project_dir in self.projects_dir.iterdir():
            if not project_dir.is_dir():
                continue
            # Prefix match search
            for jsonl_file in project_dir.glob(f"{session_id}*.jsonl"):
                # Get actual project path from JSONL
                project_name = self._get_cwd_from_jsonl(jsonl_file)
                if not project_name:
                    # Fallback: infer from directory name
                    project_name = project_dir.name.replace('-', '/')
                full_session_id = jsonl_file.stem
                matches.append((project_name, full_session_id, str(jsonl_file)))

        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            # Return None for multiple matches (error handling by caller)
            return None
        return None

    def _get_project_dag(
        self,
        project_name: str,
        session_ids: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> SessionDAG:
        """
        Get project DAG

        Args:
            project_name: Project name
            session_ids: List of target session IDs (all sessions if omitted)
            use_cache: Whether to use cache

        Returns:
            SessionDAG
        """
        # Search for project directory
        project_path = self._find_project_dir(project_name)
        if not project_path:
            raise FileNotFoundError(f"Project not found: {project_name}")

        # Get list of JSONL files
        jsonl_files = list(project_path.glob('*.jsonl'))
        if not jsonl_files:
            raise FileNotFoundError(f"JSONL files not found: {project_name}")

        # Filter files if sessions are specified
        if session_ids:
            jsonl_files = [f for f in jsonl_files if f.stem in session_ids]

        # Cache key
        cache_key = str(project_path)
        if session_ids:
            cache_key += ':' + ','.join(sorted(session_ids))

        # Check cache
        if use_cache and cache_key in self._dag_cache:
            return self._dag_cache[cache_key]

        # Build DAG
        dag = self._parser.parse_files(jsonl_files)

        # Save to cache
        if use_cache:
            self._dag_cache[cache_key] = dag

        return dag

    def _extract_messages_from_dag(
        self,
        dag: SessionDAG,
        session_id: Optional[str] = None,
        apply_filter: bool = True,
        apply_preprocessing: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Extract messages from DAG

        Args:
            dag: SessionDAG
            session_id: Specific session ID (all sessions if omitted)
            apply_filter: Whether to apply filters for continuation messages, etc.
            apply_preprocessing: Whether to apply text preprocessing

        Returns:
            List of extracted messages
        """
        messages = []
        seen_content_keys: Set[Tuple[str, int]] = set()  # For duplicate detection

        # Sort all nodes by timestamp
        sorted_nodes = sorted(
            dag.nodes.values(),
            key=lambda n: n.timestamp
        )

        for node in sorted_nodes:
            # Extract only user/assistant
            if node.type not in ('user', 'assistant'):
                continue

            # Filter by session ID
            if session_id and node.session_id != session_id:
                continue

            # Apply filter (exclude continuation messages, etc.)
            if apply_filter and self._should_exclude_node(node):
                continue

            # Check duplicates by content+timestamp
            # Exclude messages reconstructed during session continuation
            content_key = self._get_content_key(node)
            if content_key in seen_content_keys:
                continue  # Skip duplicate
            seen_content_keys.add(content_key)

            # Process message and convert to dictionary format
            msg = self._process_node(node, apply_preprocessing)
            if msg:
                messages.append(msg)

        return messages

    def _get_content_key(self, node: MessageNode) -> Tuple[str, int]:
        """
        Get content key from node (for duplicate detection)

        Messages reconstructed during session continuation have different UUIDs,
        but user messages have the same timestamp and content.
        This key detects such duplicates.

        Args:
            node: Target node

        Returns:
            Tuple of (timestamp, content_hash)
        """
        timestamp = node.timestamp
        raw_data = node.raw_data

        # Get message content
        message = raw_data.get('message', {})
        if isinstance(message, dict):
            content = message.get('content', '')
            if isinstance(content, list):
                # Use first text element
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        content = item.get('text', '')
                        break
                else:
                    # If text element not found
                    content = str(content)
            elif not isinstance(content, str):
                content = str(content)
        elif isinstance(message, str):
            content = message
        else:
            content = str(message)

        # Hash first 500 characters (ensure sufficient uniqueness)
        return (timestamp, hash(content[:500]))

    def _should_exclude_node(self, node: MessageNode) -> bool:
        """
        Determine whether node should be excluded

        Args:
            node: Node to check

        Returns:
            True if should be excluded
        """
        # Don't exclude if configuration is disabled
        if not self.config or not self.config.exclude_continuation_summaries:
            return self._is_node_continuation_message(node)

        return self._is_node_continuation_message(node)

    def _is_node_continuation_message(self, node: MessageNode) -> bool:
        """Determine whether node is a session continuation message"""
        raw_data = node.raw_data

        # Nodes with isCompactSummary flag are definitely session resumptions
        if raw_data.get('isCompactSummary', False):
            return True

        # Also check message content
        if node.type == 'user':
            message = raw_data.get('message', {})
            if isinstance(message, dict):
                content = message.get('content', [])
                if isinstance(content, list) and content:
                    first_item = content[0] if content else {}
                    if isinstance(first_item, dict):
                        text = first_item.get('text', '')
                        continuation_phrases = [
                            "This session is being continued from a previous conversation",
                            "This session is being continued",
                            "The conversation is summarized below",
                            "ran out of context",
                            "Please continue the conversation from where we left",
                            "Analysis:",
                            "Summary:"
                        ]

                        if any(phrase in text for phrase in continuation_phrases):
                            return True

        return False

    def _format_qa_answers(self, answers: Dict[str, str]) -> str:
        """
        Format AskUserQuestion answers

        Args:
            answers: Dictionary of questions and answers

        Returns:
            Formatted string
        """
        lines = ['━━━ Q&A ━━━']
        for i, (question, answer) in enumerate(answers.items(), 1):
            # Shorten if question is long
            q_short = question[:60] + '...' if len(question) > 60 else question
            lines.append('')
            lines.append(f'**[Q{i}]** {q_short}')
            lines.append(f'**[A{i}]** {answer}')
        return '\n'.join(lines)

    def _format_tool_rejection(self, processed: Dict[str, Any]) -> str:
        """
        Format tool rejection message

        Args:
            processed: Processed message dictionary

        Returns:
            Formatted string
        """
        tool_name = processed.get('rejected_tool_name', '')
        comment = processed.get('rejection_comment', '')

        if tool_name:
            return f'(Rejected {tool_name}) {comment}'
        else:
            return f'(Rejected tool execution) {comment}'

    def _get_rejected_tool_name(self, node: MessageNode) -> Optional[str]:
        """
        Get rejected tool name from parent message

        Args:
            node: Current node (user message containing tool_result)

        Returns:
            Tool name (None if not available)
        """
        try:
            # Get tool_use_id from current message
            message = node.raw_data.get('message', {})
            content = message.get('content', [])
            tool_use_id = None

            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'tool_result':
                        tool_use_id = item.get('tool_use_id')
                        break

            if not tool_use_id or not node.parent:
                return None

            # Find corresponding tool_use from parent message (assistant)
            parent_message = node.parent.raw_data.get('message', {})
            parent_content = parent_message.get('content', [])

            if isinstance(parent_content, list):
                for item in parent_content:
                    if isinstance(item, dict) and item.get('type') == 'tool_use':
                        if item.get('id') == tool_use_id:
                            return item.get('name')

            return None
        except Exception:
            return None

    def _process_node(
        self,
        node: MessageNode,
        apply_preprocessing: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Process node and convert to dictionary format

        Args:
            node: Node to process
            apply_preprocessing: Whether to apply preprocessing

        Returns:
            Converted message dictionary, None if processing fails
        """
        try:
            raw_data = node.raw_data

            # Extract basic information
            processed = {
                'type': node.type,
                'timestamp': node.timestamp,
                'sessionId': node.session_id,
                'cwd': raw_data.get('cwd'),
                'uuid': node.uuid,
                'parentUuid': node.parent_uuid  # Keep DAG information
            }

            # Extract plan approval/rejection/Q&A information
            tool_use_result = raw_data.get('toolUseResult', {})
            if isinstance(tool_use_result, dict):
                if tool_use_result.get('plan'):
                    # Approved plan
                    processed['approved_plan'] = tool_use_result['plan']
                elif tool_use_result.get('answers'):
                    # Answer to AskUserQuestion
                    processed['qa_answers'] = tool_use_result['answers']
            elif isinstance(tool_use_result, str) and 'rejected' in tool_use_result.lower():
                # Tool execution rejection (extract only when user rejects with comment)
                if 'the user said:\n' in tool_use_result:
                    processed['tool_rejected'] = True
                    user_comment = tool_use_result.split('the user said:\n', 1)[1]
                    processed['rejection_comment'] = user_comment.strip()
                    # Get rejected tool name
                    tool_name = self._get_rejected_tool_name(node)
                    if tool_name:
                        processed['rejected_tool_name'] = tool_name

            # Extract message content
            message = raw_data.get('message', {})
            if isinstance(message, dict):
                processed['role'] = message.get('role')

                # Extract text from content array
                content = message.get('content', [])
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict):
                            # Skip tool_use type
                            if item.get('type') == 'tool_use':
                                continue
                            # Extract text type
                            elif item.get('type') == 'text':
                                text_parts.append(item.get('text', ''))
                            # Skip tool_result type (exclude verbose tool output)
                            # Plan approval info is handled separately in approved_plan field
                            elif item.get('type') == 'tool_result':
                                continue

                    processed['content'] = '\n'.join(text_parts)
                else:
                    processed['content'] = str(content)
            else:
                processed['content'] = str(message)

            # Text preprocessing
            if processed.get('content'):
                if apply_preprocessing:
                    processed['content'] = self.preprocess_text(processed['content'])
                    # Keep if plan info, Q&A, or tool rejection comment exists, even if empty after preprocessing
                    if not processed['content'].strip():
                        if processed.get('approved_plan'):
                            processed['content'] = '(Plan approved)'
                        elif processed.get('tool_rejected'):
                            processed['content'] = self._format_tool_rejection(processed)
                        elif processed.get('qa_answers'):
                            processed['content'] = self._format_qa_answers(processed['qa_answers'])
                        else:
                            return None
                elif not processed['content'].strip():
                    if processed.get('approved_plan'):
                        processed['content'] = '(Plan approved)'
                    elif processed.get('tool_rejected'):
                        processed['content'] = self._format_tool_rejection(processed)
                    elif processed.get('qa_answers'):
                        processed['content'] = self._format_qa_answers(processed['qa_answers'])
                    else:
                        return None
            else:
                # Keep if plan info, Q&A, or tool rejection comment exists, even if content is empty
                if processed.get('approved_plan'):
                    processed['content'] = '(Plan approved)'
                elif processed.get('tool_rejected'):
                    processed['content'] = self._format_tool_rejection(processed)
                elif processed.get('qa_answers'):
                    processed['content'] = self._format_qa_answers(processed['qa_answers'])
                else:
                    return None

            return processed

        except Exception as e:
            print(f"Node processing error: {e}")
            return None

    def extract_project_messages(
        self,
        project_name: str,
        session_id: Optional[str] = None,
        conversation_group_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract messages from project

        Args:
            project_name: Project name (e.g., "home/user/myproject")
            session_id: Specific session ID (all sessions if omitted)
            conversation_group_id: Conversation group ID (only sessions in that group if specified)

        Returns:
            List of extracted messages
        """
        # If conversation group is specified, get session IDs for that group
        target_session_ids = None
        if conversation_group_id:
            target_session_ids = self._get_conversation_group_sessions(project_name, conversation_group_id)
            if not target_session_ids:
                raise ValueError(f"Conversation group not found: {conversation_group_id}")

        # Session ID specification
        if session_id:
            target_session_ids = [session_id]

        # Get DAG
        dag = self._get_project_dag(project_name, target_session_ids)

        # Extract messages from DAG
        all_messages = self._extract_messages_from_dag(
            dag,
            session_id=session_id,
            apply_filter=True,
            apply_preprocessing=True
        )

        # Create session list
        sessions = set()
        for msg in all_messages:
            if 'sessionId' in msg:
                sessions.add(msg['sessionId'])

        # Chronological integration (only when conversation group is specified)
        if self.config and self.config.merge_continuous_sessions and conversation_group_id:
            all_messages = self._merge_continuous_sessions(all_messages)

        print(f"Extraction complete: {len(all_messages)} messages, {len(sessions)} sessions")

        return all_messages

    def _merge_messages_with_uuid_deduplication(self, jsonl_files: List[Path]) -> List[Dict[str, Any]]:
        """
        Merge-sort multiple JSONL files with UUID deduplication

        Assuming messages within each JSONL file are in chronological order,
        merge messages from multiple files in chronological order while
        removing duplicates with the same UUID.

        Args:
            jsonl_files: List of JSONL file Paths

        Returns:
            Message list in chronological order with UUID deduplication
        """
        if not jsonl_files:
            return []

        # Manage messages and pointers for each file
        file_data = []
        for jsonl_file in jsonl_files:
            try:
                messages = self._parse_jsonl_file(jsonl_file)
                if messages:
                    file_data.append({'messages': messages, 'index': 0, 'file': jsonl_file})
            except Exception as e:
                print(f"File parse error {jsonl_file}: {e}")
                continue

        if not file_data:
            return []

        merged_messages = []

        while True:
            # Collect candidate messages from current position in each file
            current_messages = []
            for i, data in enumerate(file_data):
                if data['index'] < len(data['messages']):
                    msg = data['messages'][data['index']]
                    current_messages.append((
                        msg.get('timestamp', ''),
                        msg.get('uuid'),
                        i,
                        msg
                    ))

            if not current_messages:
                break

            # Identify minimum timestamp
            min_timestamp = min(current_messages, key=lambda x: x[0])[0]

            # Get all messages with minimum timestamp
            min_timestamp_messages = [
                (timestamp, uuid, file_idx, msg)
                for timestamp, uuid, file_idx, msg in current_messages
                if timestamp == min_timestamp
            ]

            # Group by UUID and remove duplicates
            uuid_groups = {}
            for timestamp, uuid, file_idx, msg in min_timestamp_messages:
                # Handle individually if no UUID (for legacy message format)
                key = uuid if uuid else f"no_uuid_{file_idx}_{timestamp}"
                if key not in uuid_groups:
                    uuid_groups[key] = []
                uuid_groups[key].append((file_idx, msg))

            # Output one message from each UUID group and advance all pointers
            for uuid_key, group in uuid_groups.items():
                # Output first message in group
                merged_messages.append(group[0][1])
                # Advance pointers for all files in group
                for file_idx, _ in group:
                    file_data[file_idx]['index'] += 1

        print(f"UUID deduplication merge-sort complete: {len(merged_messages)} messages")
        return merged_messages

    def group_messages_by_user_instruction(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Group messages by user instruction

        Args:
            messages: Sorted message list

        Returns:
            Grouped message list
        """
        if not messages:
            return []
        
        grouped_messages = []
        current_group = None
        
        for msg in messages:
            if msg.get('type') == 'user':
                # Save current group when new user instruction arrives
                if current_group:
                    grouped_messages.append(current_group)

                # Detect skill invocation
                content = msg.get('content', '')
                is_skill = content.startswith('Base directory for this skill:')
                skill_name = None
                if is_skill:
                    # Extract skill name from path
                    first_line = content.split('\n')[0]
                    skill_path = first_line.replace('Base directory for this skill:', '').strip()
                    skill_name = skill_path.rstrip('/').split('/')[-1]

                # Start new group
                current_group = {
                    'type': 'user_instruction_group',
                    'user_message': msg,
                    'assistant_messages': [],
                    'timestamp': msg.get('timestamp'),
                    'sessionId': msg.get('sessionId'),
                    'cwd': msg.get('cwd'),
                    'is_skill_invocation': is_skill,
                    'skill_name': skill_name
                }

            elif msg.get('type') == 'assistant' and current_group:
                # Add assistant message to current group
                current_group['assistant_messages'].append(msg)

        # Save last group
        if current_group:
            grouped_messages.append(current_group)

        # Combine assistant messages within groups
        for group in grouped_messages:
            # Combine content of assistant messages
            assistant_contents = []
            for msg in group['assistant_messages']:
                content = msg.get('content', '').strip()
                if content:
                    assistant_contents.append(content)

            # Save combined content
            group['assistant_combined_content'] = '\n\n'.join(assistant_contents)

            # Metadata
            group['assistant_message_count'] = len(group['assistant_messages'])

            # Final timestamp (time of last assistant message)
            if group['assistant_messages']:
                group['end_timestamp'] = group['assistant_messages'][-1].get('timestamp')
            else:
                group['end_timestamp'] = group['timestamp']

        print(f"Grouping complete: {len(messages)} items → {len(grouped_messages)} groups")
        
        return grouped_messages
    
    def _parse_jsonl_file(self, file_path: Path, skip_preprocessing: bool = False) -> List[Dict[str, Any]]:
        """Parse JSONL file and extract messages"""
        from rich.console import Console
        from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn

        console = Console()
        messages = []

        # Get total line count in file (for progress bar)
        with open(file_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)

        # Check if code block summarization is enabled
        # ccexport defaults to 'keep' (no LLM summarization)
        code_block_handling = self.config.code_block_handling if self.config else 'keep'
        show_progress = (code_block_handling == 'summarize' and not skip_preprocessing)

        if show_progress:
            console.print(f"[cyan]Processing {file_path.name} (with code block summarization)[/cyan]")

        with open(file_path, 'r', encoding='utf-8') as f:
            if show_progress:
                with Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    console=console
                ) as progress:
                    task = progress.add_task(f"Processing messages...", total=total_lines)

                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            progress.update(task, advance=1)
                            continue

                        try:
                            data = json.loads(line)

                            # Exclude summary type
                            if data.get('type') == 'summary':
                                progress.update(task, advance=1)
                                continue

                            # Process only user/assistant types
                            if data.get('type') in ['user', 'assistant']:
                                # Exclude continuation summaries according to settings
                                if self._should_exclude_continuation_message(data):
                                    progress.update(task, advance=1)
                                    continue

                                processed_msg = self._process_message(data, skip_preprocessing=skip_preprocessing)
                                if processed_msg:
                                    messages.append(processed_msg)
                                progress.update(task, advance=1)
                            else:
                                progress.update(task, advance=1)

                        except json.JSONDecodeError as e:
                            import sys; print(f"JSON parse error {file_path}:{line_num}: {e}", file=sys.stderr)
                            progress.update(task, advance=1)
                            continue
            else:
                # Process without progress bar (fast mode)
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)

                        # Exclude summary type
                        if data.get('type') == 'summary':
                            continue

                        # Process only user/assistant types
                        if data.get('type') in ['user', 'assistant']:
                            # Exclude continuation summaries according to settings
                            if self._should_exclude_continuation_message(data):
                                continue

                            processed_msg = self._process_message(data, skip_preprocessing=skip_preprocessing)
                            if processed_msg:
                                messages.append(processed_msg)

                    except json.JSONDecodeError as e:
                        import sys; print(f"JSON parse error {file_path}:{line_num}: {e}", file=sys.stderr)
                        # Simple implementation: skip error lines
                        continue

        return messages

    def _extract_summaries_from_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract summary entries from JSONL file

        Args:
            file_path: JSONL file path

        Returns:
            List of summary entries. Each entry contains:
            - summary: Summary text
            - leafUuid: UUID of corresponding message
            - preceding_user_uuid: UUID of immediately preceding user message
            - message_position: Number of user messages at the time summary appeared
        """
        summaries = []
        last_user_uuid = None
        user_message_count = 0

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    msg_type = data.get('type')

                    # Count user messages
                    if msg_type == 'user':
                        last_user_uuid = data.get('uuid', '')
                        user_message_count += 1

                    # Collect summary entries
                    if msg_type == 'summary':
                        summaries.append({
                            'summary': data.get('summary', ''),
                            'leafUuid': data.get('leafUuid', ''),
                            'preceding_user_uuid': last_user_uuid,
                            'message_position': user_message_count
                        })
                except json.JSONDecodeError:
                    continue

        return summaries

    def extract_project_summaries(
        self,
        project_name: str,
        session_id: Optional[str] = None,
        conversation_group_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Extract summary entries from project

        Get summaries using DAG and return with resolved message information for leafUuid.

        Args:
            project_name: Project name
            session_id: Specific session ID (all sessions if omitted)
            conversation_group_id: Conversation group ID

        Returns:
            List of summary entries (including resolved_uuid, resolved_node)
        """
        # Get session IDs for the group if conversation group is specified
        target_session_ids = None
        if conversation_group_id:
            target_session_ids = self._get_conversation_group_sessions(project_name, conversation_group_id)
            if not target_session_ids:
                return []

        # Session ID specification
        if session_id:
            target_session_ids = [session_id]

        try:
            # Get DAG
            dag = self._get_project_dag(project_name, target_session_ids)
        except FileNotFoundError:
            return []

        # Get summaries from DAG and resolve leafUuid
        result_summaries = []
        for summary in dag.summaries:
            summary_dict = {
                'summary': summary.summary,
                'leafUuid': summary.leaf_uuid,
                'raw_data': summary.raw_data
            }

            # Resolve leafUuid (find valid node)
            resolved_node = self._resolve_summary_leaf(dag, summary.leaf_uuid)
            if resolved_node:
                summary_dict['resolved_uuid'] = resolved_node.uuid
                summary_dict['resolved_timestamp'] = resolved_node.timestamp
            else:
                summary_dict['resolved_uuid'] = None
                summary_dict['resolved_timestamp'] = None

            result_summaries.append(summary_dict)

        return result_summaries

    def _resolve_summary_leaf(
        self,
        dag: SessionDAG,
        leaf_uuid: str
    ) -> Optional[MessageNode]:
        """
        Resolve summary leafUuid and return user node

        Since leafUuid typically points to an assistant message, traverse parents to find user node.
        TOC maps using user_message.uuid, so need to return user node.

        Args:
            dag: SessionDAG
            leaf_uuid: leafUuid of summary

        Returns:
            Resolved MessageNode of user type, None if not found
        """
        if not leaf_uuid:
            return None

        node = dag.get_node(leaf_uuid)
        if not node:
            return None

        # Find user node (since leafUuid often points to assistant)
        current = node
        while current:
            if current.type == 'user' and not self._should_exclude_node(current):
                return current
            current = current.parent

        # If user node not found
        return None

    def get_dag(self, project_name: str, session_ids: Optional[List[str]] = None) -> SessionDAG:
        """
        Get project DAG (public API)

        Args:
            project_name: Project name
            session_ids: List of target session IDs

        Returns:
            SessionDAG
        """
        return self._get_project_dag(project_name, session_ids)

    def _is_session_continuation_message(self, data: Dict[str, Any]) -> bool:
        """Determine if message is a context message from session resumption"""
        # Messages with isCompactSummary flag are definitely session resumptions
        if data.get('isCompactSummary', False):
            return True

        # Also check message content
        if data.get('type') == 'user':
            message = data.get('message', {})
            if isinstance(message, dict):
                content = message.get('content', [])
                if isinstance(content, list) and content:
                    first_item = content[0] if content else {}
                    if isinstance(first_item, dict):
                        text = first_item.get('text', '')
                        # Check for characteristic phrases (extended)
                        continuation_phrases = [
                            "This session is being continued from a previous conversation",
                            "This session is being continued",
                            "The conversation is summarized below", 
                            "ran out of context",
                            "Please continue the conversation from where we left",
                            "Analysis:",  # Pattern indicating start of summary
                            "Summary:"    # Pattern indicating start of summary
                        ]
                        
                        if any(phrase in text for phrase in continuation_phrases):
                            return True
        
        return False
    
    def _get_conversation_group_sessions(self, project_name: str, conversation_group_id: str) -> Optional[List[str]]:
        """
        Get list of session IDs corresponding to conversation group ID

        Args:
            project_name: Project name
            conversation_group_id: Conversation group ID (e.g., "conv-1")

        Returns:
            List of session IDs, None if not found
        """
        try:
            # Get conversation group using ConversationGroupManager
            from .conversation_manager import ConversationGroupManager
            manager = ConversationGroupManager(self)
            groups = manager.detect_conversation_groups(project_name)

            # Search for specified group ID
            for group in groups:
                if group.group_id == conversation_group_id:
                    return group.session_ids
            
            return None

        except Exception as e:
            print(f"Conversation group retrieval error: {e}")
            return None
    
    def _merge_continuous_sessions(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chronologically integrate messages within conversation group

        Args:
            messages: Message list from multiple sessions (already UUID-deduplicated)

        Returns:
            Chronologically integrated message list
        """
        if not messages:
            return messages

        # Detect session boundaries and integrate
        integrated_messages = []
        session_boundaries = set()

        # Identify session boundaries
        current_session = None
        for msg in messages:
            msg_session = msg.get('sessionId')
            if current_session != msg_session:
                if current_session is not None:
                    session_boundaries.add(len(integrated_messages))
                current_session = msg_session

        # Integration: standardize session information
        for msg in messages:
            # Assign cross-session message number
            integrated_msg = msg.copy()
            integrated_msg['session_sequence'] = len(integrated_messages) + 1

            # Flag indicating continuity within conversation group
            integrated_msg['is_session_boundary'] = len(integrated_messages) in session_boundaries

            integrated_messages.append(integrated_msg)

        print(f"Chronological integration complete: integrated {len(messages)} messages")
        return integrated_messages
    
    def _should_exclude_continuation_message(self, data: Dict[str, Any]) -> bool:
        """Determine whether to exclude continuation message according to settings"""
        # Use conventional check if settings are disabled
        if not self.config or not self.config.exclude_continuation_summaries:
            return self._is_session_continuation_message(data)

        # Exclude continuation messages if settings are enabled
        return self._is_session_continuation_message(data)
    
    def _process_message(self, raw_data: Dict[str, Any], skip_preprocessing: bool = False) -> Optional[Dict[str, Any]]:
        """Convert raw message data for processing"""
        try:
            # Extract basic information
            processed = {
                'type': raw_data.get('type'),
                'timestamp': raw_data.get('timestamp'),
                'sessionId': raw_data.get('sessionId'),
                'cwd': raw_data.get('cwd'),
                'uuid': raw_data.get('uuid')
            }

            # Extract message content
            message = raw_data.get('message', {})
            if isinstance(message, dict):
                processed['role'] = message.get('role')

                # Extract text from content array
                content = message.get('content', [])
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict):
                            # Skip tool_use type
                            if item.get('type') == 'tool_use':
                                continue
                            # Extract only text type
                            elif item.get('type') == 'text':
                                text_parts.append(item.get('text', ''))
                    
                    processed['content'] = '\n'.join(text_parts)
                else:
                    processed['content'] = str(content)
            else:
                processed['content'] = str(message)

            # Text preprocessing (only if skip_preprocessing is False)
            if processed.get('content'):
                if not skip_preprocessing:
                    processed['content'] = self.preprocess_text(processed['content'])
                    # Skip if empty after preprocessing
                    if not processed['content'].strip():
                        return None
                # Only basic empty string check if skip_preprocessing is True
                elif not processed['content'].strip():
                    return None
            else:
                # Skip if content is empty
                return None

            return processed

        except Exception as e:
            print(f"Message processing error: {e}")
            return None
    
    def _is_simple_inline_code(self, code_content: str) -> bool:
        """
        Determine if code is short inline code

        Args:
            code_content: Code block content

        Returns:
            True: Short inline code, False: Code subject to summarization
        """
        lines = [line.strip() for line in code_content.strip().split('\n') if line.strip()]

        # Basically inline if only 1 line
        if len(lines) <= 1:
            return True

        # Treat as inline if 2-3 lines under following conditions
        if len(lines) <= 3:
            # Combination of short configuration lines
            if all(len(line) <= 50 for line in lines):
                return True

            # Sequence of simple commands
            simple_patterns = [
                r'^[a-zA-Z_][a-zA-Z0-9_-]*\s*=',  # Variable assignment
                r'^[a-zA-Z][a-zA-Z0-9_-]*\s+',    # Command
                r'^\w+\s*\(',                      # Function call
                r'^\s*#',                          # Comment line
                r'^\s*//|^\s*/\*',                 # Comment line (C-style)
                r'^\s*"[^"]*"',                    # String literal
                r'^\s*\{|\s*\}',                   # Braces only
            ]

            simple_line_count = 0
            for line in lines:
                for pattern in simple_patterns:
                    if re.match(pattern, line):
                        simple_line_count += 1
                        break

            # Treat as inline if majority are simple lines
            if simple_line_count >= len(lines) * 0.8:
                return True

        return False

    def preprocess_text(self, content: str) -> str:
        """
        Text preprocessing

        - Code block processing (summarize/remove/keep)
        - Normalize excessive whitespace
        - Remove control characters
        """
        if not content:
            return ""

        # Determine code block processing method according to settings
        # ccexport defaults to 'keep' (no LLM summarization)
        code_block_handling = self.config.code_block_handling if self.config else 'keep'

        # Code block processing
        if code_block_handling == 'keep':
            # Keep code blocks as-is
            pass
        elif code_block_handling == 'remove':
            # Simple removal
            content = re.sub(r'```.*?```', '[Code block removed]', content, flags=re.DOTALL)

        # Keep inline code (don't remove)
        # Contains important information like directory names, file names, variable names

        # Simplify large JSON blocks (only multi-line)
        # Keep small single-line JSON
        content = re.sub(r'\{[^\{\}]*\n[^\{\}]*\}', '[JSON omitted]', content, flags=re.DOTALL)

        # Normalize consecutive whitespace/newlines
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        content = re.sub(r'[ \t]+', ' ', content)

        # Remove leading/trailing whitespace
        content = content.strip()
        
        return content
    
    def get_session_info(self, project_name: str, session_id: str = None) -> List[Dict[str, Any]]:
        """
        Get session information within project

        Extract session information quickly from DAG

        Args:
            project_name: Project name
            session_id: Specify to get only specific session ID

        Returns:
            Information on session ID, start time, end time, message count
            If session_id specified, dictionary for that session (None if not found)
        """
        try:
            # Get DAG (use cache)
            dag = self._get_project_dag(project_name)
        except FileNotFoundError as e:
            raise e

        sessions = {}

        # Get project path (for constructing JSONL path)
        project_dir_name = project_name.replace('/', '-')
        project_path = self.projects_dir / project_dir_name

        # Collect session information from DAG nodes
        for node in dag.nodes.values():
            node_session_id = node.session_id
            if not node_session_id:
                continue

            # Process only that session if session_id specified
            if session_id and node_session_id != session_id:
                continue

            if node_session_id not in sessions:
                jsonl_path = project_path / f"{node_session_id}.jsonl"
                sessions[node_session_id] = {
                    'session_id': node_session_id,
                    'start_time': node.timestamp,
                    'end_time': node.timestamp,
                    'message_count': 0,
                    'cwd': node.raw_data.get('cwd'),
                    'jsonl_path': str(jsonl_path)
                }

            sessions[node_session_id]['message_count'] += 1

            # Update start_time with earlier time
            if node.timestamp and (not sessions[node_session_id]['start_time'] or
                                   node.timestamp < sessions[node_session_id]['start_time']):
                sessions[node_session_id]['start_time'] = node.timestamp

            # Update end_time with later time
            if node.timestamp and (not sessions[node_session_id]['end_time'] or
                                   node.timestamp > sessions[node_session_id]['end_time']):
                sessions[node_session_id]['end_time'] = node.timestamp

        # Return dictionary for corresponding session if session_id specified
        if session_id:
            return sessions.get(session_id)

        # Sort by start time
        session_list = list(sessions.values())
        session_list.sort(key=lambda x: x.get('start_time', ''))

        return session_list

    def get_session_initial_summaries(
        self,
        project_name: str,
        session_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get summaries at beginning of session (summaries indicating continuation source)

        Returns summaries at the beginning of JSONL file (before first user message).
        These are embedded as summary of previous session chain when continued with `claude -c`.

        Args:
            project_name: Project name
            session_id: Session ID

        Returns:
            List of summary entries at file beginning (typically 0-1 entries)
        """
        project_dir = self.projects_dir / project_name.replace('/', '-')
        jsonl_file = project_dir / f"{session_id}.jsonl"

        if not jsonl_file.exists():
            return []

        summaries = self._extract_summaries_from_file(jsonl_file)
        # message_position == 0 are summaries at file beginning
        return [s for s in summaries if s.get('message_position', -1) == 0]

    def find_session_by_uuid(
        self,
        project_name: str,
        uuid: str
    ) -> Optional[str]:
        """
        Search for session containing specific UUID

        Args:
            project_name: Project name
            uuid: UUID to search for

        Returns:
            Found session ID, None if not found
        """
        project_dir = self.projects_dir / project_name.replace('/', '-')

        if not project_dir.exists():
            return None

        for jsonl_file in project_dir.glob("*.jsonl"):
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if data.get('uuid') == uuid:
                            return jsonl_file.stem
                    except json.JSONDecodeError:
                        continue

        return None

    def get_session_chain(
        self,
        project_name: str,
        session_id: str
    ) -> List[str]:
        """
        Return list of session IDs

        Note: Previously traced back to continuation source sessions using leafUuid,
        but sessions resumed with `-c` option are appended to the same file,
        so leading summary in separate file is from auto-compact.
        Continuation chain search is deprecated; return only single session.

        Args:
            project_name: Project name
            session_id: Session ID

        Returns:
            List of session IDs (single element)
        """
        return [session_id]

