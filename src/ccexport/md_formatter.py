# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025-2026 @yosagi
"""
Markdown formatter

Format extracted messages in Markdown format
"""

from datetime import datetime
from .osc_tap_loader import parse_iso_timestamp, format_local_timestamp


def _to_local_ts(ts_str: str) -> str:
    """Convert ISO 8601 timestamp to local time string"""
    if not ts_str:
        return ''
    try:
        ts = parse_iso_timestamp(ts_str)
        return format_local_timestamp(ts)
    except (ValueError, TypeError):
        # Return truncated original string if parsing fails
        return ts_str[:16]


def _build_summary_map(messages: list, summaries: list) -> dict:
    """Map summaries to message indices"""
    if not summaries:
        return {}

    # Create map from message UUID to index
    uuid_to_index = {}
    for idx, msg in enumerate(messages, 1):
        if msg.get('type') == 'user_instruction_group':
            user_msg = msg.get('user_message', {})
            uuid = user_msg.get('uuid', '')
            if uuid:
                uuid_to_index[uuid] = idx
            # Also map assistant_messages UUID
            for assistant_msg in msg.get('assistant_messages', []):
                a_uuid = assistant_msg.get('uuid', '')
                if a_uuid:
                    uuid_to_index[a_uuid] = idx

    # Map summary to index
    index_to_summary = {}
    for s in summaries:
        summary_text = s.get('summary', '')
        if not summary_text:
            continue
        # Skip error messages
        if summary_text.startswith('API Error:') or summary_text.startswith('Error:'):
            continue

        # Match with resolved_uuid or leafUuid
        resolved_uuid = s.get('resolved_uuid', '')
        leaf_uuid = s.get('leafUuid', '')
        idx = uuid_to_index.get(resolved_uuid) or uuid_to_index.get(leaf_uuid)

        if idx and idx not in index_to_summary:
            # Use only first match (avoid duplicates)
            index_to_summary[idx] = summary_text

    return index_to_summary


def _offset_markdown_header(line: str) -> str:
    """
    Offset Markdown header level

    Adjust so headers in Assistant responses are level 4 or lower:
    - # → #### (+3)
    - ## → #### (+2)
    - ### → ##### (+2)
    - #### → ###### (+2)
    - ##### or higher → ###### (capped at 6)
    """
    if not line.startswith('#'):
        return line

    # Count header level
    level = 0
    for char in line:
        if char == '#':
            level += 1
        else:
            break

    # Not a header if no space after '#'
    if level == 0 or len(line) <= level or line[level] != ' ':
        return line

    # Calculate new level
    if level == 1:
        new_level = 4  # # → ####
    elif level <= 4:
        new_level = level + 2  # ## → ####, ### → #####, #### → ######
    else:
        new_level = 6  # Max 6

    # Generate new header
    return '#' * new_level + line[level:]


def format_as_markdown(messages: list, project_name: str, grouped: bool,
                       summaries: list = None, titles_map: dict = None) -> str:
    """
    Format extracted messages in Markdown format

    Args:
        messages: List of extracted messages
        project_name: Project name
        grouped: Whether messages are grouped
        summaries: List of summary entries (optional)
        titles_map: Title map from osc-tap (index → title, optional)

    Returns:
        Markdown format string
    """
    summary_map = _build_summary_map(messages, summaries)
    if titles_map is None:
        titles_map = {}

    lines = []

    # Title
    lines.append(f"# {project_name} - Export")
    lines.append("")

    # Generated time
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    if grouped and messages and messages[0].get('type') == 'user_instruction_group':
        # Statistics (no header)
        # Aggregate session information
        sessions = set()
        for msg in messages:
            if 'user_message' in msg:
                session_id = msg['user_message'].get('sessionId')
                if session_id:
                    sessions.add(session_id)

        lines.append(f"- **Groups**: {len(messages)}")
        lines.append(f"- **Sessions**: {len(sessions)}")

        # Period
        if messages:
            start_time = messages[0].get('timestamp', '')
            end_time = messages[-1].get('end_timestamp', messages[-1].get('timestamp', ''))
            lines.append(f"- **Period**: {_to_local_ts(start_time)} - {_to_local_ts(end_time)}")

        lines.append("")
        lines.append("---")
        lines.append("")

        for i, group in enumerate(messages, 1):
            user_msg = group.get('user_message', {})
            user_content = user_msg.get('content', '')
            timestamp = group.get('timestamp', '')
            assistant_count = group.get('assistant_message_count', 0)

            # Add summary to section title
            # Priority: 1. summary from summaries, 2. title from titles_map
            summary_text = summary_map.get(i, '')
            if not summary_text:
                # titles_map is 0-indexed, so reference with i-1
                summary_text = titles_map.get(i - 1, '')

            if summary_text:
                lines.append(f"## {i}. {_to_local_ts(timestamp)} - {summary_text}")
            else:
                lines.append(f"## {i}. {_to_local_ts(timestamp)}")
            lines.append("")
            lines.append(f"**User instruction**:")
            lines.append("")
            # Shortened display for skill invocations
            if group.get('is_skill_invocation'):
                skill_name = group.get('skill_name', 'unknown')
                lines.append(f"> [Skill: {skill_name} invoked]")
            else:
                # Display user instruction as quote
                for line in user_content.split('\n'):
                    lines.append(f"> {line}")
            lines.append("")
            lines.append(f"**Assistant responses**: {assistant_count}")
            lines.append("")

            # Assistant response details
            if group.get('assistant_messages'):
                lines.append("**Assistant response**:")
                lines.append("")
                for j, assistant_msg in enumerate(group['assistant_messages'], 1):
                    # Display full content (no truncation)
                    content = assistant_msg.get('content', '')
                    timestamp_detail = assistant_msg.get('timestamp', '')

                    lines.append(f"### {j}. {_to_local_ts(timestamp_detail)}")
                    lines.append("")
                    # Format content appropriately in Markdown
                    # Offset headers in responses to level 4 or lower
                    for content_line in content.split('\n'):
                        lines.append(_offset_markdown_header(content_line))
                    lines.append("")
                lines.append("")

            lines.append("---")
            lines.append("")

    else:
        # Statistics for normal messages
        lines.append(f"- **Messages**: {len(messages)}")

        # Aggregate session information
        sessions = {}
        for msg in messages:
            session_id = msg.get('sessionId', 'unknown')
            if session_id not in sessions:
                sessions[session_id] = 0
            sessions[session_id] += 1

        lines.append(f"- **Sessions**: {len(sessions)}")

        # Period
        if messages:
            start_time = messages[0].get('timestamp', '')
            end_time = messages[-1].get('timestamp', '')
            lines.append(f"- **Period**: {_to_local_ts(start_time)} - {_to_local_ts(end_time)}")

        lines.append("")

        # Statistics by session
        lines.append("## Statistics by session")
        lines.append("")
        for session_id, count in sessions.items():
            lines.append(f"- **{session_id[:8]}...**: {count}")
        lines.append("")

        # Message list
        lines.append("## Messages")
        lines.append("")

        current_session = None
        for i, msg in enumerate(messages, 1):
            session_id = msg.get('sessionId', 'unknown')
            timestamp = msg.get('timestamp', '')
            msg_type = msg.get('type', 'unknown')
            # Display full content for normal messages too
            content = msg.get('content', '')

            # Add header when session changes
            if session_id != current_session:
                if current_session is not None:
                    lines.append("")
                lines.append(f"### Session: {session_id[:8]}...")
                lines.append("")
                current_session = session_id

            # Icon based on message type
            if msg_type == 'human':
                icon = "👤"
            elif msg_type == 'assistant':
                icon = "🤖"
            else:
                icon = "📄"

            lines.append(f"#### {i}. {icon} {_to_local_ts(timestamp)} ({msg_type})")
            lines.append("")
            # Display full content
            for content_line in content.split('\n'):
                lines.append(content_line)
            lines.append("")
            lines.append("---")
            lines.append("")

    return '\n'.join(lines)
