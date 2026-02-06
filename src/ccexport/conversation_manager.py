#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025-2026 @yosagi
"""
Conversation group management module

Provides functionality to detect continued sessions in Claude Code and manage them as conversation groups.
A conversation group represents a set of sessions that are chronologically consecutive and have continuation relationships.
"""

from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
from dataclasses import dataclass

from rich.console import Console

console = Console()


@dataclass
class ConversationGroup:
    """Data class representing a conversation group"""
    group_id: str                    # conv-1, conv-2, ...
    session_ids: List[str]          # List of session IDs
    start_time: datetime            # Start time
    end_time: datetime              # End time
    topic_summary: str = ""         # Topic summary (auto-generated)
    message_count: int = 0          # Total message count
    is_continuous: bool = True      # Whether it's a continued conversation


class ConversationGroupManager:
    """Conversation group management class"""

    def __init__(self, extractor):
        """
        Initialize

        Args:
            extractor: MessageExtractor instance
        """
        self.extractor = extractor
        
    def detect_conversation_groups(self, project_name: str) -> List[ConversationGroup]:
        """
        Detect conversation groups within a project

        Args:
            project_name: Project name

        Returns:
            List of conversation groups
        """
        # Get all sessions in the project
        sessions = self._get_all_sessions(project_name)
        if not sessions:
            return []

        # Sort sessions chronologically
        sorted_sessions = self._sort_sessions_by_time(sessions)

        # Analyze continuation relationships and group sessions
        groups = self._group_sessions_by_continuation(sorted_sessions)

        console.print(f"[cyan]Conversation group detection completed: {len(groups)} groups[/cyan]")
        return groups
    
    def _get_all_sessions(self, project_name: str) -> List[Dict[str, Any]]:
        """Get all session information in a project"""
        try:
            project_path = Path(self.extractor.projects_dir) / project_name
            if not project_path.exists():
                return []

            sessions = []
            for jsonl_file in project_path.glob("*.jsonl"):
                session_id = jsonl_file.stem

                # Get basic session information
                session_info = self._analyze_session(jsonl_file, session_id)
                if session_info:
                    sessions.append(session_info)

            return sessions

        except Exception as e:
            console.print(f"[red]Session retrieval error: {e}[/red]")
            return []
    
    def _analyze_session(self, jsonl_file: Path, session_id: str) -> Optional[Dict[str, Any]]:
        """Analyze individual session"""
        try:
            messages = []
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            message = json.loads(line)
                            messages.append(message)
                        except json.JSONDecodeError:
                            continue

            if not messages:
                return None

            # Exclude files consisting only of summaries
            user_assistant_messages = [m for m in messages if m.get('type') in ['user', 'assistant']]
            if not user_assistant_messages:
                console.print(f"[dim]Session file {session_id[:8]}... consists only of summaries, excluded[/dim]")
                return None

            # Exclude sessions with only API errors
            if self._is_error_only_session(messages):
                console.print(f"[dim]Session file {session_id[:8]}... contains only API errors, excluded[/dim]")
                return None

            # Analyze session information
            session_info = {
                'session_id': session_id,
                'file_path': str(jsonl_file),
                'message_count': len(messages),
                'messages': messages,
                'is_continuation': self._is_continuation_session(messages, session_id),
                'start_time': self._extract_start_time(messages),
                'end_time': self._extract_end_time(messages),
                'first_user_message': self._extract_first_user_message(messages)
            }

            return session_info

        except Exception as e:
            console.print(f"[yellow]Session analysis error ({session_id}): {e}[/yellow]")
            return None
    
    def _is_error_only_session(self, messages: List[Dict[str, Any]]) -> bool:
        """
        Determine if a session consists only of API errors

        Criteria:
        - All assistant messages have isApiErrorMessage: true or contain "API Error" string
        - No meaningful conversation exists
        """
        assistant_messages = [m for m in messages if m.get('type') == 'assistant']

        if not assistant_messages:
            return False

        # Check if all assistant messages are error messages
        for message in assistant_messages:
            # Check isApiErrorMessage flag
            if message.get('isApiErrorMessage', False):
                continue

            # Check message content
            content = self._extract_message_content(message)
            if content and 'API Error:' in content:
                continue

            # Found a normal response that is not an error message
            return False

        # All were error messages
        return True
    
    def _is_continuation_session(self, messages: List[Dict[str, Any]], session_id: str) -> bool:
        """
        Determine if it's a continuation session

        Criteria:
        1. A message with isCompactSummary: true exists
        2. "This session is being continued" text exists
        3. A summary-type message exists
        4. The file contains multiple different session IDs (for claude -c continuation)
        5. The first message is summary type and subsequent messages have a different session ID
        """
        # Session ID duplication check (detection of claude -c continuation)
        session_ids = set()
        for message in messages:
            msg_session_id = message.get('sessionId')
            if msg_session_id:
                session_ids.add(msg_session_id)

        # If multiple session IDs are included, it's a continuation session
        if len(session_ids) > 1:
            return True

        # If the first message is summary type and the session ID in the message differs from the filename
        if messages and messages[0].get('type') == 'summary':
            # Check session IDs of 2nd and subsequent messages
            for message in messages[1:10]:  # Check from 2nd to 11th messages
                message_session_id = message.get('sessionId')
                if message_session_id:
                    # If the message's session ID differs from the filename's session ID, it's a continuation session
                    if message_session_id != session_id:
                        return True
                    break

        for message in messages[:5]:  # Check only the first 5 messages
            # Check isCompactSummary flag
            if message.get('isCompactSummary', False):
                return True

            # Check message content
            content = self._extract_message_content(message)
            if content and 'This session is being continued' in content:
                return True
        return False
    
    def _find_continuation_target(self, continuation_session: Dict[str, Any], all_sessions: List[Dict[str, Any]]) -> Optional[str]:
        """
        Identify the source session ID of a continuation session

        Look for a different session ID contained in the continuation session's messages
        """
        messages = continuation_session.get('messages', [])
        continuation_session_id = continuation_session['session_id']

        # Look for other session IDs contained in the file
        for message in messages:
            message_session_id = message.get('sessionId')
            if message_session_id and message_session_id != continuation_session_id:
                return message_session_id

        return None
    
    def _extract_message_content(self, message: Dict[str, Any]) -> str:
        """Extract actual text content from message"""
        try:
            # Process Claude Code format message structure
            if 'message' in message and isinstance(message['message'], dict):
                msg_content = message['message'].get('content', [])
                if isinstance(msg_content, list) and len(msg_content) > 0:
                    first_content = msg_content[0]
                    if isinstance(first_content, dict) and first_content.get('type') == 'text':
                        return first_content.get('text', '')

            # If there's a direct content field
            if 'content' in message:
                content = message['content']
                if isinstance(content, str):
                    return content
                elif isinstance(content, list) and len(content) > 0:
                    if isinstance(content[0], dict):
                        return content[0].get('text', '')

            return ''

        except Exception:
            return ''
    
    def _extract_start_time(self, messages: List[Dict[str, Any]]) -> Optional[datetime]:
        """Extract session start time"""
        for message in messages:
            timestamp = message.get('timestamp')
            if timestamp:
                try:
                    return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except:
                    continue
        return None

    def _extract_end_time(self, messages: List[Dict[str, Any]]) -> Optional[datetime]:
        """Extract session end time"""
        for message in reversed(messages):
            timestamp = message.get('timestamp')
            if timestamp:
                try:
                    return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except:
                    continue
        return None
    
    def _extract_first_user_message(self, messages: List[Dict[str, Any]]) -> str:
        """Extract first valid user message (for topic estimation)"""
        for message in messages:
            if message.get('type') == 'user':
                content = self._extract_message_content(message)
                if content and not self._is_continuation_content(content):
                    return content[:100] + ('...' if len(content) > 100 else '')
        return ''

    def _is_continuation_content(self, content: str) -> bool:
        """Determine if it's a continuation message"""
        continuation_patterns = [
            'This session is being continued',
            'conversation is summarized below',
            'Analysis:',
            'Please continue the conversation'
        ]
        return any(pattern in content for pattern in continuation_patterns)
    
    def _sort_sessions_by_time(self, sessions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort sessions chronologically (continuation sessions placed later even at the same time)"""
        def get_sort_key(session):
            start_time = session.get('start_time')
            is_continuation = session.get('is_continuation', False)

            if start_time:
                # Fine-tune so continuation sessions are processed later even at the same time
                adjustment = timedelta(microseconds=1) if is_continuation else timedelta(0)
                return start_time + adjustment
            return datetime.min.replace(tzinfo=timezone.utc)

        return sorted(sessions, key=get_sort_key)
    
    def _group_sessions_by_continuation(self, sorted_sessions: List[Dict[str, Any]]) -> List[ConversationGroup]:
        """Group sessions based on continuation relationships"""
        groups = []
        session_to_group = {}  # Mapping from session ID to group index
        group_counter = 1

        for session in sorted_sessions:
            if session['is_continuation']:
                # Continuation session: identify source session and add to that group
                continuation_target = self._find_continuation_target(session, sorted_sessions)
                if continuation_target and continuation_target in session_to_group:
                    # Add to source group
                    group_index = session_to_group[continuation_target]
                    groups[group_index]['sessions'].append(session)
                    session_to_group[session['session_id']] = group_index
                else:
                    # Create new group if source not found
                    group_data = {'sessions': [session]}
                    groups.append(group_data)
                    session_to_group[session['session_id']] = len(groups) - 1
            else:
                # New session: create new group
                group_data = {'sessions': [session]}
                groups.append(group_data)
                session_to_group[session['session_id']] = len(groups) - 1

        # Convert group data to ConversationGroup
        conversation_groups = []
        for i, group_data in enumerate(groups):
            group = self._create_conversation_group(
                f"conv-{i+1}",
                group_data['sessions']
            )
            conversation_groups.append(group)

        return conversation_groups
    
    def _create_conversation_group(self, group_id: str, sessions: List[Dict[str, Any]]) -> ConversationGroup:
        """Create conversation group from session list"""
        session_ids = [s['session_id'] for s in sessions]

        # Calculate time range
        start_times = [s['start_time'] for s in sessions if s['start_time']]
        end_times = [s['end_time'] for s in sessions if s['end_time']]

        start_time = min(start_times) if start_times else datetime.now(timezone.utc)
        end_time = max(end_times) if end_times else start_time

        # Calculate message count (count only user/assistant types)
        message_count = 0
        for s in sessions:
            messages = s.get('messages', [])
            user_assistant_count = len([m for m in messages if m.get('type') in ['user', 'assistant']])
            message_count += user_assistant_count

        # Generate topic summary
        topic_summary = self._generate_topic_summary(sessions)

        # Determine if it's a continuous conversation
        is_continuous = len(sessions) > 1 or any(s['is_continuation'] for s in sessions)

        return ConversationGroup(
            group_id=group_id,
            session_ids=session_ids,
            start_time=start_time,
            end_time=end_time,
            topic_summary=topic_summary,
            message_count=message_count,
            is_continuous=is_continuous
        )
    
    def _generate_topic_summary(self, sessions: List[Dict[str, Any]]) -> str:
        """Generate topic summary for conversation group"""
        # Estimate topic from first user message of first session
        for session in sessions:
            first_message = session.get('first_user_message', '')
            if first_message and not self._is_continuation_content(first_message):
                # Simple keyword extraction
                if 'CLAUDE.md' in first_message:
                    return 'Project initialization / documentation'
                elif 'server.py' in first_message:
                    return 'Server feature fixes / improvements'
                elif 'bug' in first_message or 'error' in first_message:
                    return 'Bug fixes / error handling'
                elif 'test' in first_message:
                    return 'Testing'
                elif 'implement' in first_message:
                    return 'New feature implementation'
                else:
                    # Use first sentence or 50 characters
                    sentences = first_message.split('.')
                    if sentences and len(sentences[0]) > 10:
                        return sentences[0][:50] + ('...' if len(sentences[0]) > 50 else '')

        # Default
        date_str = sessions[0]['start_time'].strftime('%m-%d') if sessions[0]['start_time'] else ''
        return f'{date_str} Work session'
    
    def _has_shared_content(self, session1: Dict[str, Any], session2: Dict[str, Any]) -> bool:
        """
        Check if there's shared message content between two sessions

        Continuation sessions may contain user messages from the previous session
        """
        messages1 = session1.get('messages', [])
        messages2 = session2.get('messages', [])

        # Get first user message of session 2
        first_user_message2 = None
        for msg in messages2[:10]:  # Check first 10 messages
            if msg.get('type') == 'user':
                content = self._extract_message_content(msg)
                if content and not self._is_continuation_content(content):
                    first_user_message2 = content
                    break

        if not first_user_message2:
            return False

        # Check user messages at end of session 1
        for msg in reversed(messages1[-20:]):  # Check last 20 messages
            if msg.get('type') == 'user':
                content = self._extract_message_content(msg)
                if content and content == first_user_message2:
                    return True

        return False