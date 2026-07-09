#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025-2026 @yosagi
"""ccexport - Claude Code conversation export tool

Export conversations to files by specifying session IDs.
Designed to be invoked non-interactively from the work-logger skill.
"""

import json
import socket
import sys
import time
from datetime import datetime, timedelta, timezone
from importlib.metadata import version as _pkg_version, PackageNotFoundError
from pathlib import Path
from typing import Optional

import click


try:
    CCEXPORT_VERSION = _pkg_version("ccexport")
except PackageNotFoundError:
    CCEXPORT_VERSION = "unknown"

from .config import Config
from .extractor import MessageExtractor
from .html_formatter import HTMLExtractFormatter
from .osc_tap_loader import get_titles_for_export


@click.group()
@click.version_option(package_name="ccexport")
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

        # Load subagent data
        all_subagents = {}
        tool_use_map = {}
        if detail_level != 'text':
            t0 = time.time()
            all_subagents = extractor.load_subagents(project_name, full_session_id)
            if all_subagents:
                tool_use_map = extractor._build_tool_use_to_agent_map(
                    project_name, full_session_id)
                extractor.enrich_subagent_data(messages, all_subagents, tool_use_map)
                debug(f"Subagents: {time.time() - t0:.3f}s ({len(all_subagents)} agents, {len(tool_use_map)} matched)")

        # Compute per-group token usage (also folds in subagent totals)
        extractor.compute_token_usage(messages, all_subagents, tool_use_map)

        # Origin-mode estimate sanity check.
        est_stats = extractor.summarize_origin_estimates(messages)
        if est_stats:
            o = est_stats.get('output_ratio')
            if o:
                click.echo(
                    f"Origin output-side ratio (est / output_tokens) over "
                    f"{o['count']} API calls: "
                    f"min={o['min']:.2f} p25={o['p25']:.2f} "
                    f"median={o['median']:.2f} p75={o['p75']:.2f} "
                    f"p95={o['p95']:.2f} max={o['max']:.2f}",
                    err=True,
                )
            b = est_stats.get('bar_scale')
            if b:
                click.echo(
                    f"Origin bar scale (budget / 5-cat sum) over "
                    f"{b['count']} API calls: "
                    f"min={b['min']:.2f} p25={b['p25']:.2f} "
                    f"median={b['median']:.2f} p75={b['p75']:.2f} "
                    f"p95={b['p95']:.2f} max={b['max']:.2f}",
                    err=True,
                )
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
        session_cwd = session_info.get('cwd') if session_info else None

        titles_map = get_titles_for_export(
            Path(titles_dir),
            full_session_id,
            messages,
            start_time,
            end_time,
            cwd=session_cwd
        )
        debug(f"Titles retrieved: {time.time() - t0:.3f}s ({len(titles_map)} items)")

    # 4.7. Header metadata (JSON only)
    header_meta = None
    if output_format == 'json':
        t0 = time.time()
        header_meta = _build_header_meta(extractor, project_name,
                                         session_chain, messages)
        debug(f"Header meta: {time.time() - t0:.3f}s")

    # 5. Format conversion
    t0 = time.time()
    output_path = Path(output)
    content = _format_messages(messages, project_name, output_format, config,
                               summaries, titles_map, detail_level,
                               output_path=output_path,
                               all_subagents=all_subagents,
                               header_meta=header_meta)
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
@click.option('--quick', '-q', is_flag=True, help='List session IDs only (no JSONL parsing, fast)')
def sessions(project: str, as_json: bool, quick: bool):
    """List sessions in a project"""
    config = Config()
    extractor = MessageExtractor(config.projects_dir, config)

    if quick:
        try:
            session_list = extractor.list_session_ids(project)
        except FileNotFoundError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)

        if as_json:
            click.echo(json.dumps(session_list, ensure_ascii=False))
        else:
            if not session_list:
                click.echo(f"No sessions found: {project}")
                return
            click.echo(f"Sessions ({len(session_list)}):")
            for s in session_list:
                start = s['start_time'][:16] if s.get('start_time') else 'N/A'
                click.echo(f"  {s['session_id'][:8]} ...  {start}")
        return

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
            info["session_type"] = session_detail.get("session_type", "interactive")
            if session_detail.get("has_duration_data"):
                info["total_duration_ms"] = session_detail.get("total_duration_ms", 0)
            else:
                info["total_duration_ms"] = None

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
            click.echo(f"Session type: {info['session_type']}")
            duration = info['total_duration_ms']
            click.echo(f"Total duration: {duration}ms" if duration is not None else "Total duration: N/A")


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
@click.option('--project', '-p', multiple=True,
              help='Project name (can be specified multiple times)')
@click.option('--all', 'all_projects', is_flag=True,
              help='Export all projects')
@click.option('--output', '-o', type=click.Path(),
              help='Output directory (required unless --registry)')
@click.option('--format', '-f', 'output_format',
              type=click.Choice(['html', 'md', 'org', 'json']),
              help='Output format (required unless --registry)')
@click.option('--registry', 'registry_path', type=click.Path(),
              help='Registry root path (enables registry output mode)')
@click.option('--hostname', default=None,
              help='Hostname for registry path (default: system hostname)')
@click.option('--force', is_flag=True,
              help='Force re-export even if up-to-date files exist')
@click.option('--since', default=None,
              help='Export sessions from this date onward (YYYY-MM-DD)')
@click.option('--name-template', default=None,
              help=f'Filename template (default: {DEFAULT_NAME_TEMPLATE}). '
                   'Variables: {{project}}, {{date}}, {{session}}, {{ext}}')
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
def batch(project: tuple[str, ...], all_projects: bool,
          output: Optional[str], output_format: Optional[str],
          registry_path: Optional[str], hostname: Optional[str],
          force: bool,
          since: Optional[str], name_template: Optional[str],
          detail_level: str,
          verbose: bool,
          titles_dir: Optional[str], day_boundary: str):
    """Batch export all sessions in a project (multiple projects supported)"""
    total_start = time.time()

    # Validate option combinations
    if not registry_path and (not output or not output_format):
        click.echo("Error: --output and --format are required unless --registry is specified", err=True)
        sys.exit(1)
    if not project and not all_projects:
        click.echo("Error: --project or --all is required", err=True)
        sys.exit(1)

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

    # Resolve project list
    if all_projects:
        project = tuple(extractor.list_projects())
        if not project:
            click.echo("No projects found", err=True)
            sys.exit(1)

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

    # Dispatch to registry or normal mode
    if registry_path:
        exported, failed, skipped, total = _batch_registry(
            extractor=extractor,
            config=config,
            all_sessions=all_sessions,
            registry_root=Path(registry_path),
            hostname=hostname or socket.gethostname(),
            force=force,
            verbose=verbose,
        )
    else:
        exported, failed, skipped, total = _batch_normal(
            extractor=extractor,
            config=config,
            all_sessions=all_sessions,
            output_dir=Path(output),
            output_format=output_format,
            name_template=name_template,
            titles_dir=Path(titles_dir) if titles_dir else None,
            detail_level=detail_level,
            verbose=verbose,
            boundary_hour=boundary_hour,
            boundary_minute=boundary_minute,
        )

    # Summary
    elapsed = time.time() - total_start
    projects_str = ", ".join(sorted(set(Path(p).name for p in project)))
    summary = f"Done: {exported}/{total} sessions exported from [{projects_str}]"
    if skipped:
        summary += f" ({skipped} skipped)"
    if failed:
        summary += f" ({failed} failed)"
    summary += f" [{elapsed:.1f}s]"
    click.echo(summary, err=True)


def _batch_normal(
    extractor: MessageExtractor,
    config: Config,
    all_sessions: list[tuple[str, dict]],
    output_dir: Path,
    output_format: str,
    name_template: Optional[str],
    titles_dir: Optional[Path],
    detail_level: str,
    verbose: bool,
    boundary_hour: int,
    boundary_minute: int,
) -> tuple[int, int, int, int]:
    """Normal batch export mode (flat directory with template-based filenames)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    template = name_template or DEFAULT_NAME_TEMPLATE

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
                titles_dir=titles_dir,
                verbose=verbose,
                detail_level=detail_level,
            )
            exported += 1
        except Exception as e:
            click.echo(f"  Error: {e}", err=True)
            failed += 1

    return exported, failed, 0, total


def _encode_project_dirname(cwd: str) -> str:
    """Encode project path for registry directory name.

    Strip $HOME/ prefix, then replace '/' and '.' with '-'.
    e.g. /home/yos/work/ccdash -> work-ccdash
         /home/yos/.config/wezterm -> -config-wezterm
    """
    home = str(Path.home())
    path = cwd
    if path.startswith(home + '/'):
        path = path[len(home) + 1:]
    elif path.startswith(home):
        path = path[len(home):]
    return path.replace('/', '-').replace('.', '-')


def _should_skip_registry(log_path: Path, force: bool) -> bool:
    """Check if a registry _log.json is up-to-date and can be skipped."""
    if force:
        return False
    if not log_path.exists():
        return False
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('ccexport_version') == CCEXPORT_VERSION
    except Exception:
        return False


def _write_meta_json(meta_path: Path, meta: dict) -> None:
    """Write _meta.json, preserving existing digest field if present."""
    if meta_path.exists():
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            if 'digest' in existing:
                meta['digest'] = existing['digest']
        except Exception:
            pass

    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.write('\n')


def _batch_registry(
    extractor: MessageExtractor,
    config: Config,
    all_sessions: list[tuple[str, dict]],
    registry_root: Path,
    hostname: str,
    force: bool,
    verbose: bool,
) -> tuple[int, int, int, int]:
    """Registry batch export mode (structured directory with _log.json + _meta.json)."""
    exported = 0
    failed = 0
    skipped = 0
    total = len(all_sessions)

    for i, (proj, session) in enumerate(all_sessions, 1):
        session_id = session['session_id']
        short_id = session_id[:8]
        project_basename = Path(proj).name

        cwd = session.get('cwd') or proj
        project_dirname = _encode_project_dirname(cwd)
        sessions_dir = registry_root / hostname / project_dirname / 'sessions'
        log_path = sessions_dir / f"{session_id}_log.json"
        meta_path = sessions_dir / f"{session_id}_meta.json"

        if _should_skip_registry(log_path, force):
            if verbose:
                click.echo(f"[{i}/{total}] Skipping {project_basename}/{short_id} (up-to-date)", err=True)
            skipped += 1
            continue

        click.echo(f"[{i}/{total}] Exporting {project_basename}/{short_id} → {project_dirname}/sessions/", err=True)

        try:
            sessions_dir.mkdir(parents=True, exist_ok=True)

            _export_single_session(
                extractor=extractor,
                project_name=proj,
                session_id=session_id,
                output_path=log_path,
                output_format='json',
                config=config,
                verbose=verbose,
                detail_level='full',
            )

            first_message = extractor.get_first_user_message(proj, session_id)

            meta = {
                'session_id': session_id,
                'project': project_basename,
                'start': session.get('start_time'),
                'end': session.get('end_time'),
                'turns': session.get('turn_count', 0),
                'total_duration_ms': session.get('total_duration_ms') if session.get('has_duration_data') else None,
                'session_type': session.get('session_type', 'interactive'),
                'first_message': first_message,
            }
            _write_meta_json(meta_path, meta)

            exported += 1
        except Exception as e:
            click.echo(f"  Error: {e}", err=True)
            failed += 1

    return exported, failed, skipped, total


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

    # Token usage per group (batch path does not load subagents)
    extractor.compute_token_usage(messages)

    # Get title map (osc-tap)
    titles_map = {}
    if titles_dir:
        session_info = extractor.get_session_info(project_name, session_id)
        start_time = session_info.get('start_time') if session_info else None
        end_time = session_info.get('end_time') if session_info else None
        session_cwd = session_info.get('cwd') if session_info else None
        titles_map = get_titles_for_export(
            titles_dir, session_id, messages, start_time, end_time,
            cwd=session_cwd
        )
        debug(f"Titles: {len(titles_map)} items")

    # Header metadata (JSON only)
    header_meta = None
    if output_format == 'json':
        header_meta = _build_header_meta(extractor, project_name,
                                         session_chain, messages)

    # Format and write
    content = _format_messages(
        messages, project_name, output_format, config,
        all_summaries, titles_map, detail_level,
        output_path=output_path,
        header_meta=header_meta
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)


FIRST_MESSAGE_MAX_CHARS = 500


def _build_header_meta(extractor: MessageExtractor, project_name: str,
                       session_chain: list, groups: list) -> dict:
    """Collect lightweight session metadata for the JSON header.

    Aggregates start/end/duration over the continuation chain; turns is the
    number of user-instruction groups in the exported output.
    """
    start = None
    end = None
    total_duration_ms = 0
    has_duration = False
    session_type = 'interactive'

    for sid in session_chain:
        info = extractor.get_session_info(project_name, sid)
        if not info:
            continue
        s, e = info.get('start_time'), info.get('end_time')
        if s and (start is None or s < start):
            start = s
        if e and (end is None or e > end):
            end = e
        if info.get('has_duration_data'):
            has_duration = True
            total_duration_ms += info.get('total_duration_ms', 0)
        # The requested session is the last in the chain
        if sid == session_chain[-1]:
            session_type = info.get('session_type', 'interactive')

    first_message = extractor.get_first_user_message(project_name, session_chain[0])
    if first_message and len(first_message) > FIRST_MESSAGE_MAX_CHARS:
        first_message = first_message[:FIRST_MESSAGE_MAX_CHARS]

    return {
        'start': start,
        'end': end,
        'turns': len(groups),
        'total_duration_ms': total_duration_ms if has_duration else None,
        'session_type': session_type,
        'first_message': first_message,
    }


def _format_messages(messages: list, project_name: str,
                    output_format: str, config: Config,
                    summaries: list = None,
                    titles_map: dict = None,
                    detail_level: str = 'normal',
                    output_path: Path = None,
                    all_subagents: dict = None,
                    header_meta: dict = None) -> str:
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
            detail_level=detail_level,
            all_subagents=all_subagents
        )
    elif output_format == 'md':
        from .md_formatter import format_as_markdown
        return format_as_markdown(messages, project_name, grouped=True,
                                  summaries=summaries, titles_map=titles_map,
                                  detail_level=detail_level,
                                  output_path=output_path,
                                  all_subagents=all_subagents)
    elif output_format == 'org':
        from .org_formatter import OrgFormatter
        formatter = OrgFormatter(config)
        return formatter.format_extract(messages, project_name,
                                        summaries=summaries, titles_map=titles_map,
                                        detail_level=detail_level,
                                        output_path=output_path,
                                        all_subagents=all_subagents)
    elif output_format == 'json':
        output = {
            'ccexport_version': CCEXPORT_VERSION,
            'project_name': project_name,
            'titles': titles_map,
            'header_meta': header_meta or {},
            'summaries': summaries or [],
            'messages': messages
        }
        return json.dumps(output, ensure_ascii=False, indent=2)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


if __name__ == '__main__':
    cli()
