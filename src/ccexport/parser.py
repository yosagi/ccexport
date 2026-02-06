# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025-2026 @yosagi
"""
DAG-based JSONL parser module

Parses Claude Code JSONL files and builds a DAG (Directed Acyclic Graph)
structure based on uuid/parentUuid.

Reference: https://piebald.ai/blog/messages-as-commits-claude-codes-git-like-dag-of-onversations
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set


@dataclass
class MessageNode:
    """DAG node (message)"""
    uuid: str
    parent_uuid: Optional[str]
    type: str  # 'user', 'assistant', 'system', etc.
    timestamp: str
    session_id: str
    raw_data: Dict[str, Any]  # Original JSON data (unfiltered)

    # For DAG traversal (set later)
    children: List['MessageNode'] = field(default_factory=list)
    parent: Optional['MessageNode'] = None

    # Processed content added later as decoration
    processed_content: Optional[str] = None

    def __hash__(self):
        return hash(self.uuid)

    def __eq__(self, other):
        if isinstance(other, MessageNode):
            return self.uuid == other.uuid
        return False


@dataclass
class SummaryEntry:
    """Summary entry"""
    summary: str
    leaf_uuid: str
    raw_data: Dict[str, Any]


@dataclass
class SessionDAG:
    """DAG for entire project"""
    nodes: Dict[str, MessageNode]  # uuid → node
    roots: List[MessageNode]       # Nodes with parentUuid=null
    leaves: List[MessageNode]      # Nodes without children
    summaries: List[SummaryEntry]  # Summary entries

    def get_node(self, uuid: str) -> Optional[MessageNode]:
        """Get node by UUID"""
        return self.nodes.get(uuid)

    def get_ancestors(self, uuid: str) -> List[MessageNode]:
        """Get all ancestors from node with specified UUID (excluding self)"""
        ancestors = []
        node = self.nodes.get(uuid)
        if not node:
            return ancestors

        current = node.parent
        while current:
            ancestors.append(current)
            current = current.parent

        return ancestors

    def get_path_to_root(self, uuid: str) -> List[MessageNode]:
        """Get path from node with specified UUID to root (including self)"""
        path = []
        node = self.nodes.get(uuid)
        if not node:
            return path

        current = node
        while current:
            path.append(current)
            current = current.parent

        return path

    def get_path_from_root(self, uuid: str) -> List[MessageNode]:
        """Get path from root to node with specified UUID (chronological order)"""
        path = self.get_path_to_root(uuid)
        path.reverse()
        return path

    def find_nearest_ancestor_by_type(
        self,
        uuid: str,
        target_type: str,
        exclude_filter: Optional[callable] = None
    ) -> Optional[MessageNode]:
        """
        Find nearest ancestor of specified type from UUID

        Args:
            uuid: UUID of starting node
            target_type: Node type to search for ('user', 'assistant', etc.)
            exclude_filter: Exclusion filter function (nodes returning True are excluded)

        Returns:
            Found node, or None if not found
        """
        node = self.nodes.get(uuid)
        if not node:
            return None

        current = node.parent
        while current:
            if current.type == target_type:
                if exclude_filter is None or not exclude_filter(current):
                    return current
            current = current.parent

        return None


class SessionParser:
    """Class for parsing JSONL files and building DAG"""

    def parse_project(self, project_path: Path) -> SessionDAG:
        """
        Parse all JSONL files in project directory and build DAG

        Args:
            project_path: Path to project directory

        Returns:
            Built SessionDAG
        """
        jsonl_files = list(project_path.glob('*.jsonl'))
        return self.parse_files(jsonl_files)

    def parse_files(self, jsonl_files: List[Path]) -> SessionDAG:
        """
        Parse multiple JSONL files and build DAG

        Args:
            jsonl_files: List of JSONL files

        Returns:
            Built SessionDAG
        """
        all_nodes: List[MessageNode] = []
        all_summaries: List[SummaryEntry] = []

        for jsonl_file in jsonl_files:
            try:
                nodes, summaries = self._parse_single_file(jsonl_file)
                all_nodes.extend(nodes)
                all_summaries.extend(summaries)
            except Exception as e:
                print(f"File parse error {jsonl_file}: {e}")
                continue

        return self._build_dag(all_nodes, all_summaries)

    def _parse_single_file(
        self,
        file_path: Path
    ) -> Tuple[List[MessageNode], List[SummaryEntry]]:
        """
        Parse single JSONL file

        Args:
            file_path: Path to JSONL file

        Returns:
            (List of MessageNodes, List of SummaryEntries)
        """
        nodes: List[MessageNode] = []
        summaries: List[SummaryEntry] = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    entry_type = data.get('type')

                    if entry_type == 'summary':
                        # Keep summary entries separately
                        summaries.append(SummaryEntry(
                            summary=data.get('summary', ''),
                            leaf_uuid=data.get('leafUuid', ''),
                            raw_data=data
                        ))
                    elif entry_type in ('user', 'assistant', 'system'):
                        # Create message node
                        uuid = data.get('uuid', '')
                        if uuid:  # Ignore entries without UUID
                            nodes.append(MessageNode(
                                uuid=uuid,
                                parent_uuid=data.get('parentUuid'),
                                type=entry_type,
                                timestamp=data.get('timestamp', ''),
                                session_id=data.get('sessionId', ''),
                                raw_data=data
                            ))
                    # Ignore other types (e.g., file-history-snapshot)

                except json.JSONDecodeError as e:
                    import sys; print(f"JSON parse error {file_path}:{line_num}: {e}", file=sys.stderr)
                    continue

        return nodes, summaries

    def _build_dag(
        self,
        nodes: List[MessageNode],
        summaries: List[SummaryEntry]
    ) -> SessionDAG:
        """
        Build DAG from node list

        If multiple nodes with the same UUID exist, use the first one (deduplication)

        Args:
            nodes: List of MessageNodes
            summaries: List of SummaryEntries

        Returns:
            Built SessionDAG
        """
        # Index nodes by UUID (deduplication)
        node_dict: Dict[str, MessageNode] = {}
        for node in nodes:
            if node.uuid not in node_dict:
                node_dict[node.uuid] = node

        # Build parent-child relationships
        roots: List[MessageNode] = []
        child_uuids: Set[str] = set()

        for node in node_dict.values():
            if node.parent_uuid and node.parent_uuid in node_dict:
                parent = node_dict[node.parent_uuid]
                node.parent = parent
                parent.children.append(node)
                child_uuids.add(node.uuid)
            else:
                # Root if parentUuid is null or not found
                roots.append(node)

        # Identify leaf nodes (nodes without children)
        leaves: List[MessageNode] = [
            node for node in node_dict.values()
            if not node.children
        ]

        # Sort roots and leaves by timestamp
        roots.sort(key=lambda n: n.timestamp)
        leaves.sort(key=lambda n: n.timestamp)

        return SessionDAG(
            nodes=node_dict,
            roots=roots,
            leaves=leaves,
            summaries=summaries
        )
