# ccexport

Export Claude Code conversation logs to html/md/org/json.

## Quick Start

### Install

```bash
pipx install .
```

### Usage

Export by session ID (prefix match supported)

```bash
ccexport export -s 0d73debd -f html -o export.html
```

Output formats: `html`, `md`, `org`, `json`

Specify project explicitly

```bash
ccexport export -s 0d73debd -f org -o export.org -p /path/to/project
```

Verbose output (timing and debug info)

```bash
ccexport export -s 0d73debd -f html -o export.html -v
```

### Listing Sessions and Projects

List sessions for a project

```bash
ccexport sessions -p /path/to/project
  4df81284...  2026-01-29T04:55  (228 messages)
  0865367a...  2026-01-30T09:14  (435 messages)
  7d73caac...  2026-01-31T05:51  (538 messages)
  1b171b00...  2026-02-02T02:28  (351 messages)
```

List all projects

```bash
ccexport projects
Projects (NN):
  /path/to/project
```

Look up project info by session ID

```bash
ccexport session-info -s 0d73debd
Project: /path/to/project
Session ID: 0d73debd-xxxx-xxxx-xxxx-xxxxxxxxxxxx
JSONL path: /home/user/.claude/projects/...
```

All listing commands support `--json` for JSON output.

### osc-tap Integration

Use terminal titles captured by [osc-tap](https://github.com/...) as section headings.

```bash
ccexport export -s 0d73debd -f org -o export.org --titles-dir ~/.local/share/osc-tap/
```

## Requirements

- Python >= 3.10
- Claude Code session logs (JSONL files under `~/.claude/projects/`)
