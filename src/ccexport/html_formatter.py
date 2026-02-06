"""HTML formatter for extracted conversations with code block collapsing."""

import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import mistune
from jinja2 import Environment, FileSystemLoader, select_autoescape
from .osc_tap_loader import parse_iso_timestamp, format_local_timestamp


class HTMLExtractFormatter:
    """Formatter class that formats extracted conversations into HTML"""

    def __init__(self, config=None):
        """Initialize

        Args:
            config: Configuration object (used for Ollama settings in style filter)
        """
        self.config = config
        # Setup Jinja2 environment
        template_dir = Path(__file__).parent / "templates"
        self.jinja_env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(['html', 'xml'])
        )

        # Initialize Mistune Markdown parser (GFM support)
        self.markdown = mistune.create_markdown(
            plugins=['strikethrough', 'footnotes', 'table', 'task_lists']
        )

    def format_extract(
        self,
        messages: List[Dict[str, Any]],
        project_name: str = "Claude Code Digest",
        collapse_code_blocks: bool = True,
        min_lines_to_collapse: int = 10,
        summaries: Optional[List[Dict[str, Any]]] = None,
        titles_map: Optional[Dict[int, str]] = None
    ) -> str:
        """
        Format extracted messages into HTML

        Args:
            messages: List of messages (obtained from extractor)
            project_name: Project name
            collapse_code_blocks: Whether to collapse code blocks
            min_lines_to_collapse: Minimum number of lines to collapse
            summaries: List of summary entries (for TOC generation)
            titles_map: Title map from osc-tap (index → title)

        Returns:
            HTML string
        """
        # Prepare conversation data
        conversations = self._prepare_conversations(
            messages, collapse_code_blocks, min_lines_to_collapse
        )

        # Count session IDs
        session_ids = set()
        for conv in conversations:
            if 'session_id' in conv:
                session_ids.add(conv['session_id'])

        # Generate TOC from summaries
        toc_items = self._build_toc_from_summaries(summaries, messages) if summaries else []

        # Add TOC items from titles_map (supplement when summary is missing)
        if titles_map:
            toc_items = self._merge_titles_into_toc(toc_items, titles_map, messages)

        # Attach section information to conversations
        if toc_items:
            conversations = self._assign_sections_to_conversations(
                conversations, messages, toc_items
            )

        # Generate TOC from approved plans
        plan_toc_items = self._build_plan_toc(conversations)

        # Build unified TOC (merge topics and plans in chronological order)
        unified_toc = self._build_unified_toc(toc_items, plan_toc_items, conversations)

        # Load and render template
        template = self.jinja_env.get_template('extract.html.j2')
        html = template.render(
            title=project_name,
            generated_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            conversation_count=len(conversations),
            session_count=len(session_ids) if session_ids else None,
            conversations=conversations,
            unified_toc=unified_toc
        )

        return html

    def _prepare_conversations(
        self,
        messages: List[Dict[str, Any]],
        collapse_code_blocks: bool,
        min_lines_to_collapse: int
    ) -> List[Dict[str, Any]]:
        """
        Format messages into conversation format

        Args:
            messages: Raw message list
            collapse_code_blocks: Whether to collapse code blocks
            min_lines_to_collapse: Minimum lines to collapse

        Returns:
            List of conversation data
        """
        conversations = []
        previous_session_id = None

        for msg in messages:
            if msg.get('type') == 'user_instruction_group':
                # Get user message
                user_msg = msg.get('user_message', {})
                user_content = user_msg.get('content', '')
                timestamp = user_msg.get('timestamp', '')
                session_id = user_msg.get('sessionId', '')

                # Detect session change
                session_change = session_id and session_id != previous_session_id
                previous_session_id = session_id

                # Get assistant message
                assistant_content = msg.get('assistant_combined_content', '')
                assistant_timestamp = ''

                # Detect and process skill invocation
                is_skill = msg.get('is_skill_invocation', False)
                skill_name = msg.get('skill_name', 'unknown')

                if is_skill:
                    # Collapse skill invocations with details tag
                    inner_html = self._convert_markdown_to_html(
                        user_content, collapse_code_blocks, min_lines_to_collapse
                    )
                    user_html = (
                        f'<details class="skill-invocation">\n'
                        f'<summary>[Skill: {skill_name} invoked]</summary>\n'
                        f'{inner_html}\n'
                        f'</details>'
                    )
                else:
                    # Normal Markdown → HTML conversion
                    user_html = self._convert_markdown_to_html(
                        user_content, collapse_code_blocks, min_lines_to_collapse
                    )

                assistant_html = ''
                if assistant_content:
                    assistant_html = self._convert_markdown_to_html(
                        assistant_content, collapse_code_blocks, min_lines_to_collapse
                    )
                    # Assistant timestamp (obtained from first message)
                    assistant_msgs = msg.get('assistant_messages', [])
                    if assistant_msgs:
                        assistant_timestamp = assistant_msgs[0].get('timestamp', '')

                # Build conversation data
                conversation_data = {
                    'session_id': session_id,
                    'session_change': session_change,
                    'timestamp': self._format_timestamp(timestamp),
                    'user_content': user_html,
                    'assistant_content': assistant_html,
                    'assistant_timestamp': self._format_timestamp(assistant_timestamp)
                }

                # Process approved plan
                if user_msg.get('approved_plan'):
                    plan_html = self._convert_markdown_to_html(
                        user_msg['approved_plan'], collapse_code_blocks, min_lines_to_collapse
                    )
                    conversation_data['approved_plan'] = user_msg['approved_plan']
                    conversation_data['approved_plan_html'] = plan_html
                    # Generate anchor ID for plan (for TOC links)
                    conversation_data['approved_plan_id'] = f"plan-{len([c for c in conversations if c.get('approved_plan_id')])}"

                # Flag for rejected plan
                if user_msg.get('plan_rejected'):
                    conversation_data['plan_rejected'] = True

                conversations.append(conversation_data)

        return conversations

    def _convert_markdown_to_html(
        self,
        markdown_text: str,
        collapse_code_blocks: bool,
        min_lines_to_collapse: int
    ) -> str:
        """
        Convert Markdown to HTML and make code blocks collapsible if needed

        Args:
            markdown_text: Markdown string
            collapse_code_blocks: Whether to collapse code blocks
            min_lines_to_collapse: Minimum lines to collapse

        Returns:
            HTML string
        """
        if not markdown_text:
            return ''

        # Markdown → HTML conversion
        html = self.markdown(markdown_text)

        # Process code block collapsing
        if collapse_code_blocks:
            html = self._wrap_code_blocks_in_details(html, min_lines_to_collapse)

        return html

    def _wrap_code_blocks_in_details(
        self,
        html: str,
        min_lines: int
    ) -> str:
        """
        Make HTML code blocks collapsible with <details> tag

        Args:
            html: HTML string
            min_lines: Minimum lines to collapse

        Returns:
            Processed HTML string
        """
        # Pattern 1: with language <pre><code class="language-xxx">...</code></pre>
        pattern_with_lang = r'<pre><code class="language-(\w+)">(.*?)</code></pre>'
        # Pattern 2: without language <pre><code>...</code></pre>
        pattern_no_lang = r'<pre><code>(.*?)</code></pre>'

        def replace_code_block_with_lang(match):
            lang = match.group(1)
            code_content = match.group(2)

            # Count lines
            line_count = code_content.count('\n') + 1

            # Don't collapse if less than minimum lines
            if line_count < min_lines:
                return match.group(0)

            # Make collapsible with <details>
            lang_display = lang.upper()
            summary = f'<summary><span class="code-lang">{lang_display}</span><span class="code-lines">{line_count} lines</span></summary>'

            return (
                f'<details class="code-block">'
                f'{summary}'
                f'<pre><code class="language-{lang} line-numbers">{code_content}</code></pre>'
                f'</details>'
            )

        def replace_code_block_no_lang(match):
            code_content = match.group(1)

            # Count lines
            line_count = code_content.count('\n') + 1

            # Don't collapse if less than minimum lines
            if line_count < min_lines:
                return match.group(0)

            # Make collapsible with <details>
            summary = f'<summary><span class="code-lang">CODE</span><span class="code-lines">{line_count} lines</span></summary>'

            return (
                f'<details class="code-block">'
                f'{summary}'
                f'<pre><code class="line-numbers">{code_content}</code></pre>'
                f'</details>'
            )

        # Replace with regex (DOTALL flag to match newlines)
        # Process with-language first (more specific pattern takes priority)
        html = re.sub(pattern_with_lang, replace_code_block_with_lang, html, flags=re.DOTALL)
        # Process without-language later
        html = re.sub(pattern_no_lang, replace_code_block_no_lang, html, flags=re.DOTALL)

        return html

    def _format_timestamp(self, timestamp: str) -> str:
        """
        Convert timestamp to local time and format it

        Args:
            timestamp: ISO format timestamp

        Returns:
            Formatted timestamp in local time
        """
        if not timestamp:
            return ''

        try:
            ts = parse_iso_timestamp(timestamp)
            return format_local_timestamp(ts, include_seconds=True)
        except (ValueError, AttributeError, TypeError):
            # Return as-is if parsing fails
            return timestamp

    def _build_toc_from_summaries(
        self,
        summaries: List[Dict[str, Any]],
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Build TOC from summary entries

        Uses resolved_uuid resolved based on DAG to locate message position.

        Args:
            summaries: List of summary entries (including resolved_uuid)
            messages: Message list

        Returns:
            List of TOC items
        """
        if not summaries:
            return []

        # Filter error messages and get unique summaries
        unique_summaries = []
        seen_texts = set()
        for s in summaries:
            summary_text = s.get('summary', '')
            if not summary_text:
                continue
            # Skip error messages
            if summary_text.startswith('API Error:') or summary_text.startswith('Error:'):
                continue
            if summary_text not in seen_texts:
                seen_texts.add(summary_text)
                unique_summaries.append(s)  # Keep entire summary entry

        if not unique_summaries:
            return []

        # Create map from message UUID to index
        # Map both user_message UUID and assistant_messages UUID
        uuid_to_index = {}
        for idx, msg in enumerate(messages):
            if msg.get('type') == 'user_instruction_group':
                # user_message UUID
                user_msg = msg.get('user_message', {})
                uuid = user_msg.get('uuid', '')
                if uuid:
                    uuid_to_index[uuid] = idx
                # assistant_messages UUID (leafUuid usually points to assistant)
                for assistant_msg in msg.get('assistant_messages', []):
                    a_uuid = assistant_msg.get('uuid', '')
                    if a_uuid:
                        uuid_to_index[a_uuid] = idx

        # Create timestamp list of groups (for fallback)
        group_timestamps = []
        for idx, msg in enumerate(messages):
            if msg.get('type') == 'user_instruction_group':
                ts = msg.get('timestamp', '')
                group_timestamps.append((idx, ts))

        def find_closest_group_by_timestamp(target_ts: str) -> int:
            """Return index of group closest to timestamp"""
            if not target_ts or not group_timestamps:
                return -1
            # Sort with simple string comparison (works correctly for ISO 8601 format)
            closest_idx = -1
            min_diff = None
            for idx, ts in group_timestamps:
                if ts:
                    # Determine closeness with string comparison (closeness within same session)
                    diff = abs(ord(ts[0]) - ord(target_ts[0])) if ts and target_ts else float('inf')
                    # Compare as strings for more accuracy
                    if ts <= target_ts:
                        if closest_idx == -1 or ts > group_timestamps[closest_idx][1]:
                            closest_idx = idx
            # Return first group if no closest one before target_ts
            return closest_idx if closest_idx >= 0 else 0

        # Find corresponding message index for each summary
        toc_items = []
        for summary_entry in unique_summaries:
            summary_text = summary_entry.get('summary', '')

            # Search with resolved_uuid resolved by DAG, fallback to leafUuid if not found
            resolved_uuid = summary_entry.get('resolved_uuid', '')
            leaf_uuid = summary_entry.get('leafUuid', '')
            idx = uuid_to_index.get(resolved_uuid, -1)
            if idx < 0:
                idx = uuid_to_index.get(leaf_uuid, -1)

            # Fallback to timestamp if UUID doesn't match
            if idx < 0:
                resolved_ts = summary_entry.get('resolved_timestamp', '')
                idx = find_closest_group_by_timestamp(resolved_ts)

            # Add to TOC
            toc_items.append({
                'id': f'section-{len(toc_items)}' if idx >= 0 else None,
                'title': summary_text,
                'start_index': idx
            })

        # Sort by index (-1 items placed at end)
        valid_items = [item for item in toc_items if item['start_index'] >= 0]
        invalid_items = [item for item in toc_items if item['start_index'] < 0]
        valid_items.sort(key=lambda x: x['start_index'])
        toc_items = valid_items + invalid_items

        # Reassign IDs (valid ones only)
        section_counter = 0
        for item in toc_items:
            if item['start_index'] >= 0:
                item['id'] = f'section-{section_counter}'
                section_counter += 1
            else:
                item['id'] = None

        return toc_items

    def _merge_titles_into_toc(
        self,
        toc_items: List[Dict[str, Any]],
        titles_map: Dict[int, str],
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge TOC items from titles_map into existing toc_items

        Add titles from titles_map only for indices without summary.

        Args:
            toc_items: Existing TOC item list
            titles_map: Title map from osc-tap (index → title)
            messages: Message list

        Returns:
            Merged TOC item list
        """
        if not titles_map:
            return toc_items

        # Collect indices covered by existing TOC
        covered_indices = set()
        for item in toc_items:
            if item.get('start_index', -1) >= 0:
                covered_indices.add(item['start_index'])

        # Add new TOC items from titles_map
        new_items = []
        for idx, title in titles_map.items():
            if idx not in covered_indices and title:
                new_items.append({
                    'id': None,  # Reassign later
                    'title': title,
                    'start_index': idx
                })

        if not new_items:
            return toc_items

        # Merge existing and new, then sort
        all_items = toc_items + new_items
        valid_items = [item for item in all_items if item.get('start_index', -1) >= 0]
        invalid_items = [item for item in all_items if item.get('start_index', -1) < 0]
        valid_items.sort(key=lambda x: x['start_index'])
        merged = valid_items + invalid_items

        # Reassign IDs
        section_counter = 0
        for item in merged:
            if item.get('start_index', -1) >= 0:
                item['id'] = f'section-{section_counter}'
                section_counter += 1
            else:
                item['id'] = None

        return merged

    def _assign_sections_to_conversations(
        self,
        conversations: List[Dict[str, Any]],
        messages: List[Dict[str, Any]],
        toc_items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Attach section information to each conversation

        Args:
            conversations: Conversation list
            messages: Original message list
            toc_items: TOC item list

        Returns:
            Conversation list with section information attached
        """
        if not toc_items:
            return conversations

        # Use only valid TOC items
        valid_items = [item for item in toc_items if item.get('id') and item.get('start_index', -1) >= 0]

        # Group sections by index
        index_to_sections = {}
        for item in valid_items:
            idx = item['start_index']
            if idx not in index_to_sections:
                index_to_sections[idx] = []
            index_to_sections[idx].append(item)

        # Assign section information to each conversation
        for conv_idx, conv in enumerate(conversations):
            # Get sections starting at this conversation
            sections_starting_here = index_to_sections.get(conv_idx, [])
            if sections_starting_here:
                conv['section_starts'] = sections_starting_here
                conv['is_section_start'] = True
            else:
                conv['is_section_start'] = False

        return conversations

    def _build_plan_toc(
        self,
        conversations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Build TOC items from approved plans

        Args:
            conversations: Conversation list (including approved_plan_id)

        Returns:
            List of plan TOC items (including conv_idx)
        """
        plan_toc_items = []
        for conv_idx, conv in enumerate(conversations):
            if conv.get('approved_plan_id') and conv.get('approved_plan'):
                # Extract plan title (first heading or initial text)
                title = self._extract_plan_title(conv['approved_plan'])
                plan_toc_items.append({
                    'id': conv['approved_plan_id'],
                    'title': title,
                    'conv_idx': conv_idx,
                    'item_type': 'plan'
                })
        return plan_toc_items

    def _build_unified_toc(
        self,
        toc_items: List[Dict[str, Any]],
        plan_toc_items: List[Dict[str, Any]],
        conversations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Build unified TOC with topics and plans in chronological order

        Args:
            toc_items: Summary-based TOC items (including start_index)
            plan_toc_items: Plan-based TOC items (including conv_idx)
            conversations: Conversation list (for obtaining timestamps)

        Returns:
            List of unified TOC items
        """
        unified = []

        # Add type and sort index to topic items
        for item in toc_items:
            if item.get('id') and item.get('start_index', -1) >= 0:
                idx = item['start_index']
                timestamp = conversations[idx].get('timestamp', '') if idx < len(conversations) else ''
                unified.append({
                    'id': item['id'],
                    'title': item['title'],
                    'sort_idx': idx,
                    'item_type': 'topic',
                    'timestamp': timestamp
                })

        # Add plan items
        for item in plan_toc_items:
            idx = item['conv_idx']
            timestamp = conversations[idx].get('timestamp', '') if idx < len(conversations) else ''
            unified.append({
                'id': item['id'],
                'title': item['title'],
                'sort_idx': idx,
                'item_type': 'plan',
                'timestamp': timestamp
            })

        # Sort by index
        unified.sort(key=lambda x: x['sort_idx'])

        # Set display flag only when date differs from previous item
        prev_date = None
        for item in unified:
            ts = item.get('timestamp', '')
            # Extract date part from timestamp (first 10 chars in YYYY-MM-DD format)
            current_date = ts[:10] if len(ts) >= 10 else ''
            if current_date and current_date != prev_date:
                item['show_date'] = True
                item['date'] = current_date
                prev_date = current_date
            else:
                item['show_date'] = False
            # Extract time part (HH:MM)
            if len(ts) >= 16:
                item['time'] = ts[11:16]
            else:
                item['time'] = ''

        return unified

    def _extract_plan_title(self, plan_text: str) -> str:
        """
        Extract title from plan text

        Args:
            plan_text: Plan text in Markdown

        Returns:
            Title string
        """
        if not plan_text:
            return '(Plan)'

        lines = plan_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Detect Markdown heading
            if line.startswith('#'):
                # Remove # and get title
                title = line.lstrip('#').strip()
                if title:
                    return title
            else:
                # Use first non-empty line if no heading (max 50 chars)
                if len(line) > 50:
                    return line[:50] + '...'
                return line

        return '(Plan)'
