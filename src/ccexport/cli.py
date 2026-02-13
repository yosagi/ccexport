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
@click.option('--verbose', '-v', is_flag=True, help='Verbose logging')
@click.option('--config-file', type=click.Path(exists=True), help='Config file path')
@click.option('--titles-dir', type=click.Path(exists=True),
              help='osc-tap log directory (for title display)')
def export(session: str, output: str, output_format: str,
           project: Optional[str], verbose: bool, config_file: Optional[str],
           titles_dir: Optional[str]):
    """Export conversation by specifying session ID"""

    def debug(msg: str):
        if verbose:
            click.echo(f"[DEBUG] {msg}", err=True)

    total_start = time.time()

    # 1. Load configuration
    t0 = time.time()
    config = Config(config_file=config_file, verbose=verbose)
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
    content = _format_messages(messages, project_name, output_format, config, summaries, titles_map)
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
@click.option('--config-file', type=click.Path(exists=True), help='Config file path')
def projects(as_json: bool, config_file: Optional[str]):
    """List projects"""
    config = Config(config_file=config_file)
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
@click.option('--config-file', type=click.Path(exists=True), help='Config file path')
def sessions(project: str, as_json: bool, config_file: Optional[str]):
    """List sessions in a project"""
    config = Config(config_file=config_file)
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
@click.option('--config-file', type=click.Path(exists=True), help='Config file path')
def session_info(session: str, as_json: bool, config_file: Optional[str]):
    """Get project info from session ID"""
    config = Config(config_file=config_file)
    extractor = MessageExtractor(config.projects_dir, config)

    result = extractor.find_project_by_session(session)
    if not result:
        if as_json:
            click.echo(json.dumps({"error": "not_found"}, ensure_ascii=False))
        else:
            click.echo(f"Error: Session '{session}' not found (or multiple matches)", err=True)
        sys.exit(1)

    project_name, full_session_id, jsonl_path = result

    if as_json:
        info = {
            "project": project_name,
            "session_id": full_session_id,
            "jsonl_path": jsonl_path
        }
        click.echo(json.dumps(info, ensure_ascii=False, indent=2))
    else:
        click.echo(f"Project: {project_name}")
        click.echo(f"Session ID: {full_session_id}")
        click.echo(f"JSONL path: {jsonl_path}")


DEFAULT_NAME_TEMPLATE = "{project}_{date}_{session}.{ext}"


@cli.command()
@click.option('--project', '-p', required=True, help='Project name')
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
@click.option('--verbose', '-v', is_flag=True, help='Verbose logging')
@click.option('--config-file', type=click.Path(exists=True),
              help='Config file path')
@click.option('--titles-dir', type=click.Path(exists=True),
              help='osc-tap log directory (for title display)')
def batch(project: str, output: str, output_format: str,
          since: Optional[str], name_template: Optional[str],
          verbose: bool, config_file: Optional[str],
          titles_dir: Optional[str]):
    """Batch export all sessions in a project"""
    total_start = time.time()

    # Load configuration
    config = Config(config_file=config_file, verbose=verbose)
    extractor = MessageExtractor(config.projects_dir, config)

    # Get all sessions for the project
    try:
        session_list = extractor.get_session_info(project)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if not session_list:
        click.echo(f"No sessions found for project: {project}", err=True)
        sys.exit(1)

    # Filter by --since
    if since:
        session_list = [
            s for s in session_list
            if (s.get('start_time') or '') >= since
        ]
        if not session_list:
            click.echo(f"No sessions found since {since}", err=True)
            sys.exit(1)

    # Prepare output directory
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filename template
    template = name_template or DEFAULT_NAME_TEMPLATE
    titles_path = Path(titles_dir) if titles_dir else None

    # Use basename of project path for filename (avoid absolute path in filename)
    project_name = Path(project).name

    # Export each session
    exported = 0
    failed = 0
    total = len(session_list)

    for i, session in enumerate(session_list, 1):
        session_id = session['session_id']
        start_date = (session.get('start_time') or '')[:10] or 'unknown'
        short_id = session_id[:8]

        filename = template.format(
            project=project_name,
            date=start_date,
            session=short_id,
            ext=output_format,
        )
        output_path = output_dir / filename

        click.echo(f"[{i}/{total}] Exporting {short_id} ... {filename}", err=True)

        try:
            _export_single_session(
                extractor=extractor,
                project_name=project,
                session_id=session_id,
                output_path=output_path,
                output_format=output_format,
                config=config,
                titles_dir=titles_path,
                verbose=verbose,
            )
            exported += 1
        except Exception as e:
            click.echo(f"  Error: {e}", err=True)
            failed += 1

    # Summary
    elapsed = time.time() - total_start
    summary = f"Done: {exported}/{total} sessions exported"
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
        all_summaries, titles_map
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)


def _format_messages(messages: list, project_name: str,
                    output_format: str, config: Config,
                    summaries: list = None,
                    titles_map: dict = None) -> str:
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
            titles_map=titles_map
        )
    elif output_format == 'md':
        from .md_formatter import format_as_markdown
        return format_as_markdown(messages, project_name, grouped=True,
                                  summaries=summaries, titles_map=titles_map)
    elif output_format == 'org':
        from .org_formatter import OrgFormatter
        formatter = OrgFormatter(config)
        return formatter.format_extract(messages, project_name,
                                        summaries=summaries, titles_map=titles_map)
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
