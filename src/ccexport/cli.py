#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025-2026 @yosagi
"""ccexport - Claude Code conversation export tool

Export conversations to files by specifying session IDs.
Designed to be invoked non-interactively from the work-logger skill.
"""

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import click

from .config import Config
from .extractor import MessageExtractor
from .html_formatter import HTMLExtractFormatter
from .osc_tap_loader import get_titles_for_export


@click.group()
def cli():
    """Claude Code conversation export tool"""
    pass


@cli.command()
@click.option('--session', '-s', required=True, help='Session ID (UUID format)')
@click.option('--output', '-o', required=True, type=click.Path(), help='Output file path')
@click.option('--format', '-f', 'output_format', required=True,
              type=click.Choice(['html', 'md', 'org', 'json']), help='Output format')
@click.option('--project', '-p', default=None, help='Project name (auto-detect if omitted)')
@click.option('--detail', '-d', 'detail_level',
              type=click.Choice(['text', 'normal', 'full']),
              default='normal',
              help='Output detail level (text=no tools, normal=tool summaries, full=tool I/O)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose logging')
@click.option('--titles-dir', type=click.Path(exists=True),
              help='osc-tap log directory (for title display)')
def export(session: str, output: str, output_format: str,
           project: Optional[str], detail_level: str, verbose: bool,
           titles_dir: Optional[str]):
    """Export conversation by specifying session ID"""

    def debug(msg: str):
        if verbose:
            click.echo(f"[DEBUG] {msg}", err=True)

    total_start = time.time()

    # 1. Load configuration
    t0 = time.time()
    config = Config(verbose=verbose)
    extractor = MessageExtractor(config.projects_dir, config)
    debug(f"Config loaded: {time.time() - t0:.3f}s")

    # 2. Identify project
    t0 = time.time()
    if project:
        project_name = project
        full_session_id = session  # Use session ID as-is when project is specified
    else:
        result = extractor.find_project_by_session(session)
        if not result:
            click.echo(f"Error: Session '{session}' not found (or multiple matches)", err=True)
            sys.exit(1)
        project_name, full_session_id, _ = result
        debug(f"Project search: {time.time() - t0:.3f}s -> {project_name} ({full_session_id[:8]}...)")

    # 3. Get continuation session chain
    t0 = time.time()
    session_chain = extractor.get_session_chain(project_name, full_session_id)
    if len(session_chain) > 1:
        debug(f"Continuation chain detected: {len(session_chain)} sessions")
        for i, sid in enumerate(session_chain):
            debug(f"  [{i+1}] {sid[:8]}...")
    debug(f"Chain retrieved: {time.time() - t0:.3f}s")

    # 4. Extract messages from entire chain
    t0 = time.time()
    try:
        all_messages = []
        all_summaries = []

        for sid in session_chain:
            messages = extractor.extract_project_messages(project_name, sid)
            all_messages.extend(messages)

            summaries = extractor.extract_project_summaries(project_name, sid)
            all_summaries.extend(summaries)

        debug(f"Messages extracted: {time.time() - t0:.3f}s ({len(all_messages)} messages)")

        t0 = time.time()
        messages = extractor.group_messages_by_user_instruction(all_messages)
        debug(f"Grouping: {time.time() - t0:.3f}s ({len(messages)} groups)")

        summaries = all_summaries
        debug(f"Summaries: {len(summaries)} items")
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if not messages:
        click.echo("Warning: No messages extracted", err=True)
        sys.exit(1)

    # 4.5. Get title map (osc-tap)
    titles_map = {}
    if titles_dir:
        t0 = time.time()
        # Get session info (start_time, end_time)
        session_info = extractor.get_session_info(project_name, full_session_id)
        start_time = session_info.get('start_time') if session_info else None
        end_time = session_info.get('end_time') if session_info else None

        titles_map = get_titles_for_export(
            Path(titles_dir),
            full_session_id,
            messages,
            start_time,
            end_time
        )
        debug(f"Titles retrieved: {time.time() - t0:.3f}s ({len(titles_map)} items)")

    # 5. Format conversion
    t0 = time.time()
    output_path = Path(output)
    content = _format_messages(messages, project_name, output_format, config,
                               summaries, titles_map, detail_level,
                               output_path=output_path)
    debug(f"Format conversion ({output_format}): {time.time() - t0:.3f}s")

    # 6. Write to file
    t0 = time.time()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    debug(f"File written: {time.time() - t0:.3f}s")

    debug(f"Total: {time.time() - total_start:.3f}s")
    if verbose:
        click.echo(f"Export completed: {output_path}")


@cli.command()
@click.option('--json', 'as_json', is_flag=True, help='Output in JSON format')
def projects(as_json: bool):
    """List projects"""
    config = Config()
    extractor = MessageExtractor(config.projects_dir, config)

    project_list = extractor.list_projects()

    if as_json:
        click.echo(json.dumps(project_list, ensure_ascii=False, indent=2))
    else:
        if not project_list:
            click.echo("No projects found")
            return

        click.echo(f"Projects ({len(project_list)}):")
        for project_name in project_list:
            click.echo(f"  {project_name}")


@cli.command()
@click.option('--project', '-p', required=True, help='Project name')
@click.option('--json', 'as_json', is_flag=True, help='Output in JSON format')
def sessions(project: str, as_json: bool):
    """List sessions in a project"""
    config = Config()
    extractor = MessageExtractor(config.projects_dir, config)

    try:
        session_list = extractor.get_session_info(project)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if as_json:
        click.echo(json.dumps(session_list, ensure_ascii=False, indent=2))
    else:
        if not session_list:
            click.echo(f"No sessions found: {project}")
            return

        click.echo(f"Sessions ({len(session_list)}):")
        for session in session_list:
            session_id = session['session_id']
            start_time = session.get('start_time', '')[:16] if session.get('start_time') else 'N/A'
            msg_count = session.get('message_count', 0)
            click.echo(f"  {session_id[:8]} ...  {start_time}  ({msg_count} messages)")


@cli.command('session-info')
@click.option('--session', '-s', required=True, help='Session ID (prefix match)')
@click.option('--json', 'as_json', is_flag=True, help='Output in JSON format')
@click.option('--verbose', '-v', is_flag=True, help='Include start/end time, turn count, message count (requires JSONL parsing)')
def session_info(session: str, as_json: bool, verbose: bool):
    """Get project info from session ID"""
    config = Config()
    extractor = MessageExtractor(config.projects_dir, config)

    result = extractor.find_project_by_session(session)
    if not result:
        if as_json:
            click.echo(json.dumps({"error": "not_found"}, ensure_ascii=False))
        else:
            click.echo(f"Error: Session '{session}' not found (or multiple matches)", err=True)
        sys.exit(1)

    project_name, full_session_id, jsonl_path = result

    info = {
        "project": project_name,
        "session_id": full_session_id,
        "jsonl_path": jsonl_path
    }

    if verbose:
        session_detail = extractor.get_session_info(project_name, full_session_id)
        if session_detail:
            info["start_time"] = session_detail.get("start_time")
            info["end_time"] = session_detail.get("end_time")
            info["turn_count"] = session_detail.get("turn_count", 0)
            info["message_count"] = session_detail.get("message_count", 0)

    if as_json:
        click.echo(json.dumps(info, ensure_ascii=False, indent=2))
    else:
        click.echo(f"Project: {project_name}")
        click.echo(f"Session ID: {full_session_id}")
        click.echo(f"JSONL path: {jsonl_path}")
        if verbose and "start_time" in info:
            click.echo(f"Start time: {info['start_time']}")
            click.echo(f"End time: {info['end_time']}")
            click.echo(f"Turn count: {info['turn_count']}")
            click.echo(f"Message count: {info['message_count']}")


DEFAULT_NAME_TEMPLATE = "{project}_{date}_{session}.{ext}"


def _compute_logical_date(iso_timestamp: str, boundary_hour: int, boundary_minute: int) -> str:
    """Compute logical date from a timestamp with day boundary offset.

    If the local time is before the boundary (e.g. 03:00), the date is
    shifted back by one day. This matches work-logger's behavior where
    late-night sessions belong to the previous day.

    Args:
        iso_timestamp: ISO 8601 timestamp string (e.g. "2026-02-27T01:30:00+09:00")
        boundary_hour: Hour of day boundary (0-23)
        boundary_minute: Minute of day boundary (0-59)

    Returns:
        Date string in YYYY-MM-DD format
    """
    from .osc_tap_loader import to_local_time

    ts = datetime.fromisoformat(iso_timestamp.replace('Z', '+00:00'))
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    local_ts = to_local_time(ts)

    if local_ts.hour < boundary_hour or (
        local_ts.hour == boundary_hour and local_ts.minute < boundary_minute
    ):
        local_ts = local_ts.replace(hour=0, minute=0) - timedelta(days=1)

    return local_ts.strftime('%Y-%m-%d')


@cli.command()
@click.option('--project', '-p', required=True, multiple=True,
              help='Project name (can be specified multiple times)')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output directory')
@click.option('--format', '-f', 'output_format', required=True,
              type=click.Choice(['html', 'md', 'org', 'json']),
              help='Output format')
@click.option('--since', default=None,
              help='Export sessions from this date onward (YYYY-MM-DD)')
@click.option('--name-template', default=None,
              help=f'Filename template (default: {DEFAULT_NAME_TEMPLATE}). '
                   'Variables: {project}, {date}, {session}, {ext}')
@click.option('--detail', '-d', 'detail_level',
              type=click.Choice(['text', 'normal', 'full']),
              default='normal',
              help='Output detail level (text=no tools, normal=tool summaries, full=tool I/O)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose logging')
@click.option('--titles-dir', type=click.Path(exists=True),
              help='osc-tap log directory (for title display)')
@click.option('--day-boundary', default='00:00',
              help='Day boundary time in HH:MM (default: 00:00). '
                   'Sessions ending before this time use the previous date.')
def batch(project: tuple[str, ...], output: str, output_format: str,
          since: Optional[str], name_template: Optional[str],
          detail_level: str,
          verbose: bool,
          titles_dir: Optional[str], day_boundary: str):
    """Batch export all sessions in a project (multiple projects supported)"""
    total_start = time.time()

    # Parse day boundary
    try:
        parts = day_boundary.split(':')
        boundary_hour = int(parts[0])
        boundary_minute = int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        click.echo(f"Invalid --day-boundary format: {day_boundary} (expected HH:MM)", err=True)
        sys.exit(1)

    # Load configuration
    config = Config(verbose=verbose)
    extractor = MessageExtractor(config.projects_dir, config)

    # Collect sessions from all specified projects
    all_sessions: list[tuple[str, dict]] = []  # (project_name, session_info)
    for proj in project:
        try:
            session_list = extractor.get_session_info(proj)
        except FileNotFoundError as e:
            click.echo(f"Warning: {e} (skipping)", err=True)
            continue

        if not session_list:
            click.echo(f"No sessions found for project: {proj} (skipping)", err=True)
            continue

        # Filter by --since
        if since:
            session_list = [
                s for s in session_list
                if (s.get('start_time') or '') >= since
            ]

        for s in session_list:
            all_sessions.append((proj, s))

    if not all_sessions:
        click.echo("No sessions found for any specified project", err=True)
        sys.exit(1)

    # Prepare output directory
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filename template
    template = name_template or DEFAULT_NAME_TEMPLATE
    titles_path = Path(titles_dir) if titles_dir else None

    # Export each session
    exported = 0
    failed = 0
    total = len(all_sessions)

    for i, (proj, session) in enumerate(all_sessions, 1):
        session_id = session['session_id']
        end_time = session.get('end_time') or session.get('start_time') or ''
        if end_time:
            logical_date = _compute_logical_date(end_time, boundary_hour, boundary_minute)
        else:
            logical_date = 'unknown'
        short_id = session_id[:8]

        # Use basename of project path for filename (avoid absolute path in filename)
        project_name = Path(proj).name

        filename = template.format(
            project=project_name,
            date=logical_date,
            session=short_id,
            ext=output_format,
        )
        output_path = output_dir / filename

        click.echo(f"[{i}/{total}] Exporting {project_name}/{short_id} ... {filename}", err=True)

        try:
            _export_single_session(
                extractor=extractor,
                project_name=proj,
                session_id=session_id,
                output_path=output_path,
                output_format=output_format,
                config=config,
                titles_dir=titles_path,
                verbose=verbose,
                detail_level=detail_level,
            )
            exported += 1
        except Exception as e:
            click.echo(f"  Error: {e}", err=True)
            failed += 1

    # Summary
    elapsed = time.time() - total_start
    projects_str = ", ".join(Path(p).name for p in project)
    summary = f"Done: {exported}/{total} sessions exported from [{projects_str}]"
    if failed:
        summary += f" ({failed} failed)"
    summary += f" [{elapsed:.1f}s]"
    click.echo(summary, err=True)


def _export_single_session(
    extractor: MessageExtractor,
    project_name: str,
    session_id: str,
    output_path: Path,
    output_format: str,
    config: Config,
    titles_dir: Optional[Path] = None,
    verbose: bool = False,
    detail_level: str = 'normal',
) -> None:
    """Export a single session to a file.

    Raises on failure (FileNotFoundError, ValueError, etc.)
    """
    def debug(msg: str):
        if verbose:
            click.echo(f"[DEBUG] {msg}", err=True)

    # Get continuation session chain
    session_chain = extractor.get_session_chain(project_name, session_id)
    if len(session_chain) > 1:
        debug(f"Continuation chain: {len(session_chain)} sessions")

    # Extract messages from chain
    all_messages = []
    all_summaries = []
    for sid in session_chain:
        messages = extractor.extract_project_messages(project_name, sid)
        all_messages.extend(messages)
        summaries = extractor.extract_project_summaries(project_name, sid)
        all_summaries.extend(summaries)

    messages = extractor.group_messages_by_user_instruction(all_messages)
    if not messages:
        raise ValueError(f"No messages extracted for session {session_id[:8]}")

    debug(f"Messages: {len(all_messages)} raw, {len(messages)} groups")

    # Get title map (osc-tap)
    titles_map = {}
    if titles_dir:
        session_info = extractor.get_session_info(project_name, session_id)
        start_time = session_info.get('start_time') if session_info else None
        end_time = session_info.get('end_time') if session_info else None
        titles_map = get_titles_for_export(
            titles_dir, session_id, messages, start_time, end_time
        )
        debug(f"Titles: {len(titles_map)} items")

    # Format and write
    content = _format_messages(
        messages, project_name, output_format, config,
        all_summaries, titles_map, detail_level,
        output_path=output_path
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)


def _format_messages(messages: list, project_name: str,
                    output_format: str, config: Config,
                    summaries: list = None,
                    titles_map: dict = None,
                    detail_level: str = 'normal',
                    output_path: Path = None) -> str:
    """Format messages"""
    if titles_map is None:
        titles_map = {}

    if output_format == 'html':
        formatter = HTMLExtractFormatter(config)
        return formatter.format_extract(
            messages,
            project_name=project_name,
            collapse_code_blocks=True,
            min_lines_to_collapse=10,
            summaries=summaries,
            titles_map=titles_map,
            detail_level=detail_level
        )
    elif output_format == 'md':
        from .md_formatter import format_as_markdown
        return format_as_markdown(messages, project_name, grouped=True,
                                  summaries=summaries, titles_map=titles_map,
                                  detail_level=detail_level,
                                  output_path=output_path)
    elif output_format == 'org':
        from .org_formatter import OrgFormatter
        formatter = OrgFormatter(config)
        return formatter.format_extract(messages, project_name,
                                        summaries=summaries, titles_map=titles_map,
                                        detail_level=detail_level,
                                        output_path=output_path)
    elif output_format == 'json':
        output = {
            'project_name': project_name,
            'messages': messages,
            'summaries': summaries or [],
            'titles': titles_map
        }
        return json.dumps(output, ensure_ascii=False, indent=2)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


if __name__ == '__main__':
    cli()
