# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025-2026 @yosagi
"""
org-mode formatter

Export Claude Code conversation logs in org-mode format.
"""

import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from .osc_tap_loader import parse_iso_timestamp, format_local_timestamp, format_duration


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


class OrgFormatter:
    """Export in org-mode format"""

    def __init__(self, config=None):
        self.config = config

    def format_extract(
        self,
        messages: List[Dict[str, Any]],
        project_name: str = "Claude Code Export",
        summaries: Optional[List[Dict[str, Any]]] = None,
        titles_map: Optional[Dict[int, str]] = None
    ) -> str:
        """
        Convert messages to org-mode format

        Args:
            messages: Grouped message list
            project_name: Project name
            summaries: List of summary entries (for section titles)
            titles_map: Title map from osc-tap (index → title)

        Returns:
            org-mode format string
        """
        lines = []

        # Collect approved plans first (for link list generation)
        plans = self._collect_approved_plans(messages)

        # Header
        lines.extend(self._format_header(project_name, messages))

        # Link list to approved plans
        if plans:
            lines.extend(self._format_plan_links(plans))

        # Create mapping between summaries and messages
        index_to_summary = self._build_summary_map(messages, summaries) if summaries else {}

        # Conversation section
        lines.append("* Conversations")
        lines.append("")

        for i, msg in enumerate(messages, 1):
            # Priority: 1. summary from summaries, 2. title from titles_map
            summary_text = index_to_summary.get(i, '')
            if not summary_text and titles_map:
                # titles_map is 0-indexed, so reference with i-1
                summary_text = titles_map.get(i - 1, '')
            lines.extend(self._format_conversation(msg, i, summary_text))

        return '\n'.join(lines)

    def _build_summary_map(
        self,
        messages: List[Dict[str, Any]],
        summaries: List[Dict[str, Any]]
    ) -> Dict[int, str]:
        """
        Map summaries to message indices

        Args:
            messages: Grouped message list
            summaries: List of summary entries

        Returns:
            Map of index → summary text
        """
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

    def _collect_approved_plans(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Tuple[str, str, str]]:
        """
        Collect approved plans

        Returns:
            List of (plan_id, timestamp, plan_content)
        """
        plans = []

        for i, msg in enumerate(messages, 1):
            if msg.get('type') != 'user_instruction_group':
                continue

            user_msg = msg.get('user_message', {})
            if user_msg.get('approved_plan'):
                # Generate plan_id using conversation index
                # Use same logic as _format_conversation
                plan_id = f"plan-{i}"
                timestamp = _to_local_ts(user_msg.get('timestamp', ''))
                plan_content = user_msg['approved_plan']

                # Extract plan title (first line or first heading)
                title = self._extract_plan_title(plan_content)

                plans.append((plan_id, timestamp, title))

        return plans

    def _extract_plan_title(self, plan_content: str) -> str:
        """Extract plan title"""
        lines = plan_content.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line:
                # Remove Markdown heading
                if line.startswith('#'):
                    line = re.sub(r'^#+\s*', '', line)
                # Return first non-empty line (max 60 chars)
                if len(line) > 60:
                    return line[:57] + "..."
                return line
        return "Plan"

    def _format_header(
        self,
        project_name: str,
        messages: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate org file header"""
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Collect session IDs
        session_ids = set()
        for msg in messages:
            if msg.get('type') == 'user_instruction_group':
                user_msg = msg.get('user_message', {})
                sid = user_msg.get('sessionId')
                if sid:
                    session_ids.add(sid)

        # Calculate period
        start_time = ""
        end_time = ""
        if messages:
            first_msg = messages[0]
            last_msg = messages[-1]
            if first_msg.get('timestamp'):
                start_time = _to_local_ts(first_msg['timestamp'])
            if last_msg.get('end_timestamp'):
                end_time = _to_local_ts(last_msg['end_timestamp'])
            elif last_msg.get('timestamp'):
                end_time = _to_local_ts(last_msg['timestamp'])

        lines = [
            f"#+TITLE: {project_name} - Conversation Export",
            "#+STARTUP: hideblocks",
            f"#+DATE: {now}",
            "",
            "* Metadata",
            f"- Generated: {now}",
            f"- Conversations: {len(messages)}",
            f"- Sessions: {len(session_ids)}",
        ]

        if start_time and end_time:
            lines.append(f"- Period: {start_time} ~ {end_time}")

        # Total AI time
        total_ms = sum(msg.get('duration_ms', 0) or 0 for msg in messages)
        if total_ms > 0:
            lines.append(f"- Total AI time: {format_duration(total_ms)}")

        lines.append("")

        return lines

    def _format_plan_links(
        self,
        plans: List[Tuple[str, str, str]]
    ) -> List[str]:
        """Generate link list to approved plans"""
        lines = [
            "* Approved Plans",
            "",
        ]

        for plan_id, timestamp, title in plans:
            lines.append(f"- [[#{plan_id}][{title}]] [{timestamp}]")

        lines.append("")
        return lines

    def _format_conversation(
        self,
        msg: Dict[str, Any],
        index: int,
        summary_text: str = ""
    ) -> List[str]:
        """Convert conversation group to org format"""
        lines = []

        if msg.get('type') != 'user_instruction_group':
            return lines

        user_msg = msg.get('user_message', {})
        timestamp = _to_local_ts(user_msg.get('timestamp', ''))
        session_id = user_msg.get('sessionId', '')
        user_content = user_msg.get('content', '')

        # Duration display
        duration_ms = msg.get('duration_ms')
        duration_str = f" ({format_duration(duration_ms)})" if duration_ms is not None else ""

        # User instruction section (datetime + summary + duration)
        if summary_text:
            lines.append(f"** {timestamp}{duration_str} - {summary_text}")
        else:
            lines.append(f"** {timestamp}{duration_str}")
        lines.append(":PROPERTIES:")
        lines.append(f":CUSTOM_ID: msg-{index}")
        lines.append(f":SESSION_ID: {session_id[:8]}..." if session_id else ":SESSION_ID:")
        lines.append(":END:")
        lines.append("")

        # Shortened display for skill invocations
        if msg.get('is_skill_invocation'):
            skill_name = msg.get('skill_name', 'unknown')
            lines.append(f"[Skill: {skill_name} invoked]")
        else:
            # User message content
            lines.extend(self._convert_content(user_content))
        lines.append("")

        # Assistant response
        assistant_content = msg.get('assistant_combined_content', '')
        if assistant_content:
            assistant_timestamp = ''
            assistant_msgs = msg.get('assistant_messages', [])
            if assistant_msgs:
                assistant_timestamp = _to_local_ts(assistant_msgs[0].get('timestamp', ''))

            # Title is datetime only
            lines.append(f"*** {assistant_timestamp}" if assistant_timestamp else "*** Assistant")
            lines.append("")
            lines.extend(self._convert_content(assistant_content))
            lines.append("")

        # Approved plan
        if user_msg.get('approved_plan'):
            plan_content = user_msg['approved_plan']
            # Generate plan_id using conversation index
            plan_id = f"plan-{index}"

            lines.append("*** Approved Plan")
            lines.append(":PROPERTIES:")
            lines.append(f":CUSTOM_ID: {plan_id}")
            lines.append(":END:")
            lines.append("")
            lines.extend(self._convert_content(plan_content))
            lines.append("")

        return lines

    def _extract_title(self, content: str, max_length: int = 50) -> str:
        """Extract title from content"""
        lines = content.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line:
                if len(line) > max_length:
                    return line[:max_length - 3] + "..."
                return line
        return "Message"

    def _convert_content(self, content: str) -> List[str]:
        """
        Convert Markdown content to org-mode format

        Conversions:
        - Code blocks: ```lang → #+BEGIN_SRC lang / #+END_SRC
          - But for org: #+BEGIN_EXAMPLE / #+END_EXAMPLE (avoid nesting)
        - Headings: # → * (but lowered by 2 levels)
        - Others: as-is
        """
        lines = []
        in_code_block = False
        code_lang = ""
        use_example = False

        for line in content.split('\n'):
            # Code block start/end
            if line.strip().startswith('```'):
                if not in_code_block:
                    # Code block start
                    in_code_block = True
                    code_lang = line.strip()[3:].strip()
                    # Use EXAMPLE for org source to avoid nesting
                    if code_lang == "org":
                        use_example = True
                        lines.append("#+BEGIN_EXAMPLE")
                    else:
                        use_example = False
                        lines.append(f"#+BEGIN_SRC {code_lang}" if code_lang else "#+BEGIN_SRC")
                else:
                    # Code block end
                    in_code_block = False
                    if use_example:
                        lines.append("#+END_EXAMPLE")
                    else:
                        lines.append("#+END_SRC")
                    code_lang = ""
                    use_example = False
                continue

            if in_code_block:
                # In code block: add space to * or #+ at line start to prevent org misrecognition
                if line.startswith('*') or line.startswith('#+'):
                    lines.append(' ' + line)
                else:
                    lines.append(line)
            else:
                # Convert Markdown headings to org headings (level 4 or lower)
                if line.strip().startswith('#'):
                    match = re.match(r'^(#+)\s*(.*)', line.strip())
                    if match:
                        level = len(match.group(1))
                        text = match.group(2)
                        # Headings in conversations should be **** or lower
                        org_level = '*' * min(level + 3, 6)
                        lines.append(f"{org_level} {text}")
                        continue

                # Others as-is
                lines.append(line)

        return lines
