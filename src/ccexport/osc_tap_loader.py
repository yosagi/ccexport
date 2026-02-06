"""Loading osc-tap logs and title mapping"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Spinner character pattern
SPINNER_PATTERN = re.compile(r'^[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏⠐⠂✳★☆] ')


def remove_spinner(title: str) -> str:
    """Remove spinner characters from title"""
    return SPINNER_PATTERN.sub('', title)


def parse_iso_timestamp(ts_str: str) -> datetime:
    """Parse ISO 8601 format timestamp (timezone-aware)"""
    # Python 3.11+ fromisoformat can parse +09:00 format directly
    # For 3.10 and earlier, need to convert 'Z' to '+00:00'
    if ts_str.endswith('Z'):
        ts_str = ts_str[:-1] + '+00:00'
    return datetime.fromisoformat(ts_str)


def to_local_time(ts: datetime) -> datetime:
    """Convert UTC timestamp to local time"""
    local_tz = datetime.now().astimezone().tzinfo
    return ts.astimezone(local_tz)


def format_local_timestamp(ts: datetime, include_seconds: bool = False) -> str:
    """Convert timestamp to local time and format as display string"""
    local_ts = to_local_time(ts)
    if include_seconds:
        return local_ts.strftime('%Y-%m-%dT%H:%M:%S')
    return local_ts.strftime('%Y-%m-%dT%H:%M')


def load_osc_logs(titles_dir: Path) -> List[Dict]:
    """Load all osc-tap logs from specified directory"""
    logs = []
    if not titles_dir.exists():
        return logs

    for jsonl_path in sorted(titles_dir.glob('*.jsonl')):
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        entry['_source_file'] = str(jsonl_path)
                        logs.append(entry)
                    except json.JSONDecodeError:
                        continue
        except IOError:
            continue

    return logs


def find_logs_for_session(
    logs: List[Dict],
    session_id: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> List[Dict]:
    """Filter logs related to specified session

    1. Prioritize log files matching by SESSION_ID
    2. Fall back to time range if no match
    """
    # Look for files matching SESSION_ID
    matching_files = set()
    for entry in logs:
        if entry.get('matcher') == 'SESSION_ID' and entry.get('string') == session_id:
            matching_files.add(entry.get('_source_file'))

    if matching_files:
        # Return only logs from SESSION_ID matched files
        return [e for e in logs if e.get('_source_file') in matching_files]

    # Fallback: filter by time range
    if start_time is None and end_time is None:
        return logs

    filtered = []
    for entry in logs:
        try:
            ts = parse_iso_timestamp(entry.get('ts', ''))
            if start_time and ts < start_time:
                continue
            if end_time and ts > end_time:
                continue
            filtered.append(entry)
        except (ValueError, TypeError):
            continue

    return filtered


def extract_titles(logs: List[Dict]) -> List[Dict]:
    """Extract TITLE entries, remove spinners and deduplicate

    Returns:
        List of {'ts': datetime, 'title': str}
    """
    titles = []
    prev_title = None

    for entry in logs:
        if entry.get('matcher') != 'TITLE':
            continue

        raw_title = entry.get('string', '')
        clean_title = remove_spinner(raw_title)

        # Skip empty titles or "Claude Code"
        if not clean_title or clean_title == 'Claude Code':
            continue

        # Skip consecutive duplicates
        if clean_title == prev_title:
            continue

        try:
            ts = parse_iso_timestamp(entry.get('ts', ''))
            titles.append({'ts': ts, 'title': clean_title})
            prev_title = clean_title
        except (ValueError, TypeError):
            continue

    return titles


def build_title_map(
    titles: List[Dict],
    messages: List[Dict],
    timestamp_key: str = 'timestamp'
) -> Dict[int, str]:
    """Create mapping between message indices and titles

    Matching rule:
    Assign title at time T to the user message immediately before T

    Args:
        titles: Result of extract_titles()
        messages: Message list (user_instruction_group format)
        timestamp_key: Timestamp key in messages

    Returns:
        Dictionary of {message index: title}
    """
    if not titles or not messages:
        return {}

    title_map = {}

    # Parse and list message start times
    msg_times = []
    for i, msg in enumerate(messages):
        ts_str = msg.get(timestamp_key)
        if ts_str:
            try:
                ts = parse_iso_timestamp(ts_str)
                msg_times.append((i, ts))
            except (ValueError, TypeError):
                continue

    if not msg_times:
        return {}

    # Assign each title to appropriate message
    for title_entry in titles:
        title_ts = title_entry['ts']
        title_text = title_entry['title']

        # Find message immediately before (or right after) title change time
        best_idx = None
        best_diff = None

        for msg_idx, msg_ts in msg_times:
            # Title change occurs after message submission
            # Choose closest in range where message time <= title time
            diff = (title_ts - msg_ts).total_seconds()
            if diff >= 0:  # Title time is at or after message time
                if best_diff is None or diff < best_diff:
                    best_diff = diff
                    best_idx = msg_idx

        # Accept only matches within 60 seconds
        if best_idx is not None and best_diff is not None and best_diff <= 60:
            # Don't overwrite if already assigned (prioritize first title)
            if best_idx not in title_map:
                title_map[best_idx] = title_text

    return title_map


def get_titles_for_export(
    titles_dir: Path,
    session_id: str,
    messages: List[Dict],
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
) -> Dict[int, str]:
    """Get title map for export (main API)

    Args:
        titles_dir: osc-tap log directory
        session_id: Session ID
        messages: Message list
        start_time: Session start time (ISO 8601)
        end_time: Session end time (ISO 8601)

    Returns:
        Dictionary of {message index: title}
    """
    # Load logs
    all_logs = load_osc_logs(titles_dir)
    if not all_logs:
        return {}

    # Parse timestamps
    start_dt = parse_iso_timestamp(start_time) if start_time else None
    end_dt = parse_iso_timestamp(end_time) if end_time else None

    # Filter logs related to session
    session_logs = find_logs_for_session(all_logs, session_id, start_dt, end_dt)

    # Extract titles
    titles = extract_titles(session_logs)

    # Create mapping
    return build_title_map(titles, messages)
