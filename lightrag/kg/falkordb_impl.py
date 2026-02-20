import os
import re
from collections import deque
from dataclasses import dataclass
from typing import final

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from ..utils import logger
from ..base import BaseGraphStorage
from ..types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge
from ..kg.shared_storage import get_data_init_lock
import pipmaster as pm

if not pm.is_installed("falkordb"):
    pm.install("falkordb")

from falkordb.asyncio import FalkorDB  # type: ignore
from redis.asyncio import BlockingConnectionPool  # type: ignore

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)

RETRY_EXCEPTIONS = (
    ConnectionResetError,
    ConnectionError,
    OSError,
    TimeoutError,
)

READ_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(RETRY_EXCEPTIONS),
    reraise=True,
)

WRITE_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(RETRY_EXCEPTIONS),
    reraise=True,
)


@final
@dataclass
class FalkorDBStorage(BaseGraphStorage):
    def __init__(self, namespace, global_config, embedding_func, workspace=None):
        falkordb_workspace = os.environ.get("FALKORDB_WORKSPACE")
        original_workspace = workspace
        if falkordb_workspace and falkordb_workspace.strip():
            workspace = falkordb_workspace

        if not workspace or not str(workspace).strip():
            workspace = "base"

        super().__init__(
            namespace=namespace,
            workspace=workspace,
            global_config=global_config,
            embedding_func=embedding_func,
        )

        if falkordb_workspace and falkordb_workspace.strip():
            logger.info(
                f"Using FALKORDB_WORKSPACE environment variable: '{falkordb_workspace}' "
                f"(overriding '{original_workspace}/{namespace}')"
            )

        self._db = None
        self._graph = None
        self._pool = None

    def _get_workspace_label(self) -> str:
        return self.workspace

    def _get_graph_name(self) -> str:
        safe_ns = re.sub(r"[^A-Za-z0-9_]", "_", self.namespace)
        safe_ws = re.sub(r"[^A-Za-z0-9_]", "_", self.workspace)
        return f"lightrag_{safe_ns}_{safe_ws}"

    def _is_chinese_text(self, text: str) -> bool:
        cjk_pattern = re.compile(
            r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]|[\U00020000-\U0002fa1f]"
        )
        return bool(cjk_pattern.search(text))

    async def initialize(self):
        async with get_data_init_lock():
            HOST = os.environ.get("FALKORDB_HOST", "localhost")
            PORT = int(os.environ.get("FALKORDB_PORT", "6379"))
            USERNAME = os.environ.get("FALKORDB_USERNAME")
            PASSWORD = os.environ.get("FALKORDB_PASSWORD")
            MAX_CONNECTIONS = int(os.environ.get("FALKORDB_MAX_CONNECTIONS", "16"))

            pool_kwargs = dict(
                host=HOST,
                port=PORT,
                max_connections=MAX_CONNECTIONS,
                timeout=None,
                decode_responses=True,
            )
            if USERNAME:
                pool_kwargs["username"] = USERNAME
            if PASSWORD:
                pool_kwargs["password"] = PASSWORD

            self._pool = BlockingConnectionPool(**pool_kwargs)
            self._db = FalkorDB(connection_pool=self._pool)

            graph_name = self._get_graph_name()
            self._graph = self._db.select_graph(graph_name)

            try:
                await self._graph.ro_query("RETURN 1")
                logger.info(
                    f"[{self.workspace}] Connected to FalkorDB at {HOST}:{PORT}, graph '{graph_name}'"
                )
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Failed to connect to FalkorDB at {HOST}:{PORT}: {e}"
                )
                raise

            workspace_label = self._get_workspace_label()
            await self._ensure_index(workspace_label)
            await self._ensure_fulltext_index(workspace_label)

    async def _ensure_index(self, workspace_label: str):
        try:
            await self._graph.query(
                f"CREATE INDEX FOR (n:`{workspace_label}`) ON (n.entity_id)"
            )
            logger.info(
                f"[{self.workspace}] Created index on entity_id for label '{workspace_label}'"
            )
        except Exception as e:
            if "already indexed" in str(e).lower() or "equivalent index" in str(e).lower():
                logger.debug(
                    f"[{self.workspace}] Index on entity_id for '{workspace_label}' already exists"
                )
            else:
                logger.warning(
                    f"[{self.workspace}] Could not create index on entity_id: {e}"
                )

    async def _ensure_fulltext_index(self, workspace_label: str):
        try:
            await self._graph.query(
                f"CALL db.idx.fulltext.createNodeIndex('{workspace_label}', 'entity_id')"
            )
            logger.info(
                f"[{self.workspace}] Created full-text index on entity_id for label '{workspace_label}'"
            )
        except Exception as e:
            if "already indexed" in str(e).lower() or "equivalent index" in str(e).lower():
                logger.debug(
                    f"[{self.workspace}] Full-text index on entity_id for '{workspace_label}' already exists"
                )
            else:
                logger.warning(
                    f"[{self.workspace}] Could not create full-text index on entity_id: {e}"
                )

    async def finalize(self):
        if self._pool is not None:
            await self._pool.aclose()
            self._pool = None
            self._db = None
            self._graph = None

    async def __aexit__(self, exc_type, exc, tb):
        await self.finalize()

    async def index_done_callback(self) -> None:
        pass

    @READ_RETRY
    async def has_node(self, node_id: str) -> bool:
        workspace_label = self._get_workspace_label()
        query = (
            f"MATCH (n:`{workspace_label}` {{entity_id: $entity_id}}) "
            "RETURN count(n) > 0 AS node_exists"
        )
        try:
            result = await self._graph.ro_query(query, {"entity_id": node_id})
            if result.result_set:
                return bool(result.result_set[0][0])
            return False
        except Exception as e:
            logger.error(f"[{self.workspace}] Error checking node existence for {node_id}: {e}")
            raise

    @READ_RETRY
    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        workspace_label = self._get_workspace_label()
        query = (
            f"MATCH (a:`{workspace_label}` {{entity_id: $src}})"
            f"-[r]-(b:`{workspace_label}` {{entity_id: $tgt}}) "
            "RETURN COUNT(r) > 0 AS edgeExists"
        )
        try:
            result = await self._graph.ro_query(
                query, {"src": source_node_id, "tgt": target_node_id}
            )
            if result.result_set:
                return bool(result.result_set[0][0])
            return False
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error checking edge existence between "
                f"{source_node_id} and {target_node_id}: {e}"
            )
            raise

    @READ_RETRY
    async def get_node(self, node_id: str) -> dict[str, str] | None:
        workspace_label = self._get_workspace_label()
        query = f"MATCH (n:`{workspace_label}` {{entity_id: $entity_id}}) RETURN n"
        try:
            result = await self._graph.ro_query(query, {"entity_id": node_id})
            rows = result.result_set
            if len(rows) > 1:
                logger.warning(
                    f"[{self.workspace}] Multiple nodes found with entity_id '{node_id}'. Using first."
                )
            if rows:
                return dict(rows[0][0].properties)
            return None
        except Exception as e:
            logger.error(f"[{self.workspace}] Error getting node for {node_id}: {e}")
            raise

    @READ_RETRY
    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        workspace_label = self._get_workspace_label()
        query = (
            "UNWIND $node_ids AS id "
            f"MATCH (n:`{workspace_label}` {{entity_id: id}}) "
            "RETURN n.entity_id AS entity_id, n"
        )
        try:
            result = await self._graph.ro_query(query, {"node_ids": node_ids})
            return {row[0]: dict(row[1].properties) for row in result.result_set}
        except Exception as e:
            logger.error(f"[{self.workspace}] Error in get_nodes_batch: {e}")
            raise

    @READ_RETRY
    async def node_degree(self, node_id: str) -> int:
        workspace_label = self._get_workspace_label()
        query = (
            f"MATCH (n:`{workspace_label}` {{entity_id: $entity_id}}) "
            "OPTIONAL MATCH (n)-[r]-() "
            "RETURN COUNT(r) AS degree"
        )
        try:
            result = await self._graph.ro_query(query, {"entity_id": node_id})
            if result.result_set:
                return int(result.result_set[0][0] or 0)
            logger.warning(f"[{self.workspace}] No node found with entity_id '{node_id}'")
            return 0
        except Exception as e:
            logger.error(f"[{self.workspace}] Error getting node degree for {node_id}: {e}")
            raise

    @READ_RETRY
    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        workspace_label = self._get_workspace_label()
        query = (
            "UNWIND $node_ids AS id "
            f"MATCH (n:`{workspace_label}` {{entity_id: id}}) "
            "OPTIONAL MATCH (n)-[r]-() "
            "RETURN n.entity_id AS entity_id, COUNT(r) AS degree"
        )
        try:
            result = await self._graph.ro_query(query, {"node_ids": node_ids})
            degrees = {row[0]: int(row[1] or 0) for row in result.result_set}
            for nid in node_ids:
                if nid not in degrees:
                    logger.warning(f"[{self.workspace}] No node found with entity_id '{nid}'")
                    degrees[nid] = 0
            return degrees
        except Exception as e:
            logger.error(f"[{self.workspace}] Error in node_degrees_batch: {e}")
            raise

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        src_degree = await self.node_degree(src_id)
        trg_degree = await self.node_degree(tgt_id)
        return int(src_degree or 0) + int(trg_degree or 0)

    @READ_RETRY
    async def edge_degrees_batch(
        self, edge_pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], int]:
        unique_node_ids = {src for src, _ in edge_pairs}
        unique_node_ids.update({tgt for _, tgt in edge_pairs})
        degrees = await self.node_degrees_batch(list(unique_node_ids))
        return {
            (src, tgt): degrees.get(src, 0) + degrees.get(tgt, 0)
            for src, tgt in edge_pairs
        }

    @READ_RETRY
    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        workspace_label = self._get_workspace_label()
        query = (
            f"MATCH (start:`{workspace_label}` {{entity_id: $src}})"
            f"-[r]-(end:`{workspace_label}` {{entity_id: $tgt}}) "
            "RETURN properties(r) AS edge_properties"
        )
        try:
            result = await self._graph.ro_query(
                query, {"src": source_node_id, "tgt": target_node_id}
            )
            rows = result.result_set
            if len(rows) > 1:
                logger.warning(
                    f"[{self.workspace}] Multiple edges found between "
                    f"'{source_node_id}' and '{target_node_id}'. Using first."
                )
            if rows:
                edge_props = dict(rows[0][0]) if rows[0][0] else {}
                for key, default in {
                    "weight": 1.0,
                    "source_id": None,
                    "description": None,
                    "keywords": None,
                }.items():
                    if key not in edge_props:
                        edge_props[key] = default
                return edge_props
            return None
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error in get_edge between "
                f"{source_node_id} and {target_node_id}: {e}"
            )
            raise

    @READ_RETRY
    async def get_edges_batch(
        self, pairs: list[dict[str, str]]
    ) -> dict[tuple[str, str], dict]:
        workspace_label = self._get_workspace_label()
        query = (
            "UNWIND $pairs AS pair "
            f"MATCH (start:`{workspace_label}` {{entity_id: pair.src}})"
            f"-[r:DIRECTED]-(end:`{workspace_label}` {{entity_id: pair.tgt}}) "
            "RETURN pair.src AS src_id, pair.tgt AS tgt_id, collect(properties(r)) AS edges"
        )
        try:
            result = await self._graph.ro_query(query, {"pairs": pairs})
            edges_dict = {}
            for row in result.result_set:
                src, tgt, edges = row[0], row[1], row[2]
                if edges:
                    edge_props = dict(edges[0])
                    for key, default in {
                        "weight": 1.0,
                        "source_id": None,
                        "description": None,
                        "keywords": None,
                    }.items():
                        if key not in edge_props:
                            edge_props[key] = default
                    edges_dict[(src, tgt)] = edge_props
                else:
                    edges_dict[(src, tgt)] = {
                        "weight": 1.0,
                        "source_id": None,
                        "description": None,
                        "keywords": None,
                    }
            return edges_dict
        except Exception as e:
            logger.error(f"[{self.workspace}] Error in get_edges_batch: {e}")
            raise

    @READ_RETRY
    async def get_node_edges(
        self, source_node_id: str
    ) -> list[tuple[str, str]] | None:
        workspace_label = self._get_workspace_label()
        query = (
            f"MATCH (n:`{workspace_label}` {{entity_id: $entity_id}}) "
            f"OPTIONAL MATCH (n)-[r]-(connected:`{workspace_label}`) "
            "WHERE connected.entity_id IS NOT NULL "
            "RETURN n, r, connected"
        )
        try:
            result = await self._graph.ro_query(query, {"entity_id": source_node_id})
            edges = []
            for row in result.result_set:
                source_node = row[0]
                connected_node = row[2]
                if not source_node or not connected_node:
                    continue
                source_label = source_node.properties.get("entity_id")
                target_label = connected_node.properties.get("entity_id")
                if source_label and target_label:
                    edges.append((source_label, target_label))
            return edges
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error getting edges for node {source_node_id}: {e}"
            )
            raise

    @READ_RETRY
    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        workspace_label = self._get_workspace_label()
        query = (
            "UNWIND $node_ids AS id "
            f"MATCH (n:`{workspace_label}` {{entity_id: id}}) "
            f"OPTIONAL MATCH (n)-[r]-(connected:`{workspace_label}`) "
            "RETURN id AS queried_id, n.entity_id AS node_entity_id, "
            "connected.entity_id AS connected_entity_id, "
            "startNode(r).entity_id AS start_entity_id"
        )
        try:
            result = await self._graph.ro_query(query, {"node_ids": node_ids})
            edges_dict = {node_id: [] for node_id in node_ids}
            for row in result.result_set:
                queried_id = row[0]
                node_entity_id = row[1]
                connected_entity_id = row[2]
                start_entity_id = row[3]
                if not node_entity_id or not connected_entity_id:
                    continue
                if start_entity_id == node_entity_id:
                    edges_dict[queried_id].append((node_entity_id, connected_entity_id))
                else:
                    edges_dict[queried_id].append((connected_entity_id, node_entity_id))
            return edges_dict
        except Exception as e:
            logger.error(f"[{self.workspace}] Error in get_nodes_edges_batch: {e}")
            raise

    @WRITE_RETRY
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        workspace_label = self._get_workspace_label()
        properties = node_data
        entity_type = properties.get("entity_type", "UNKNOWN")
        if "entity_id" not in properties:
            raise ValueError(
                "FalkorDB: node properties must contain an 'entity_id' field"
            )
        try:
            query = (
                f"MERGE (n:`{workspace_label}` {{entity_id: $entity_id}}) "
                "SET n += $properties "
                f"SET n:`{entity_type}`"
            )
            await self._graph.query(
                query, {"entity_id": node_id, "properties": properties}
            )
        except Exception as e:
            logger.error(f"[{self.workspace}] Error during upsert_node: {e}")
            raise

    @WRITE_RETRY
    async def upsert_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        edge_data: dict[str, str],
    ) -> None:
        workspace_label = self._get_workspace_label()
        try:
            query = (
                f"MATCH (source:`{workspace_label}` {{entity_id: $src}}) "
                "WITH source "
                f"MATCH (target:`{workspace_label}` {{entity_id: $tgt}}) "
                "MERGE (source)-[r:DIRECTED]-(target) "
                "SET r += $properties "
                "RETURN r, source, target"
            )
            await self._graph.query(
                query,
                {
                    "src": source_node_id,
                    "tgt": target_node_id,
                    "properties": edge_data,
                },
            )
        except Exception as e:
            logger.error(f"[{self.workspace}] Error during upsert_edge: {e}")
            raise

    @WRITE_RETRY
    async def delete_node(self, node_id: str) -> None:
        workspace_label = self._get_workspace_label()
        query = (
            f"MATCH (n:`{workspace_label}` {{entity_id: $entity_id}}) DETACH DELETE n"
        )
        try:
            await self._graph.query(query, {"entity_id": node_id})
            logger.debug(f"[{self.workspace}] Deleted node with entity_id '{node_id}'")
        except Exception as e:
            logger.error(f"[{self.workspace}] Error during node deletion: {e}")
            raise

    @WRITE_RETRY
    async def remove_nodes(self, nodes: list[str]):
        for node in nodes:
            await self.delete_node(node)

    @WRITE_RETRY
    async def remove_edges(self, edges: list[tuple[str, str]]):
        workspace_label = self._get_workspace_label()
        for source, target in edges:
            query = (
                f"MATCH (source:`{workspace_label}` {{entity_id: $src}})"
                f"-[r]-(target:`{workspace_label}` {{entity_id: $tgt}}) "
                "DELETE r"
            )
            try:
                await self._graph.query(query, {"src": source, "tgt": target})
                logger.debug(
                    f"[{self.workspace}] Deleted edge from '{source}' to '{target}'"
                )
            except Exception as e:
                logger.error(f"[{self.workspace}] Error during edge deletion: {e}")
                raise

    async def get_all_labels(self) -> list[str]:
        workspace_label = self._get_workspace_label()
        query = (
            f"MATCH (n:`{workspace_label}`) "
            "WHERE n.entity_id IS NOT NULL "
            "RETURN DISTINCT n.entity_id AS label ORDER BY label"
        )
        try:
            result = await self._graph.ro_query(query)
            return [row[0] for row in result.result_set]
        except Exception as e:
            logger.error(f"[{self.workspace}] Error getting all labels: {e}")
            raise

    async def get_all_nodes(self) -> list[dict]:
        workspace_label = self._get_workspace_label()
        query = f"MATCH (n:`{workspace_label}`) RETURN n"
        try:
            result = await self._graph.ro_query(query)
            nodes = []
            for row in result.result_set:
                node_dict = dict(row[0].properties)
                node_dict["id"] = node_dict.get("entity_id")
                nodes.append(node_dict)
            return nodes
        except Exception as e:
            logger.error(f"[{self.workspace}] Error getting all nodes: {e}")
            raise

    async def get_all_edges(self) -> list[dict]:
        workspace_label = self._get_workspace_label()
        query = (
            f"MATCH (a:`{workspace_label}`)-[r]-(b:`{workspace_label}`) "
            "RETURN DISTINCT a.entity_id AS source, b.entity_id AS target, "
            "properties(r) AS properties"
        )
        try:
            result = await self._graph.ro_query(query)
            edges = []
            for row in result.result_set:
                edge_properties = dict(row[2]) if row[2] else {}
                edge_properties["source"] = row[0]
                edge_properties["target"] = row[1]
                edges.append(edge_properties)
            return edges
        except Exception as e:
            logger.error(f"[{self.workspace}] Error getting all edges: {e}")
            raise

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        workspace_label = self._get_workspace_label()
        query = (
            f"MATCH (n:`{workspace_label}`) "
            "WHERE n.entity_id IS NOT NULL "
            "OPTIONAL MATCH (n)-[r]-() "
            "WITH n.entity_id AS label, count(r) AS degree "
            "ORDER BY degree DESC, label ASC "
            "LIMIT $limit "
            "RETURN label"
        )
        try:
            result = await self._graph.ro_query(query, {"limit": limit})
            labels = [row[0] for row in result.result_set]
            logger.debug(
                f"[{self.workspace}] Retrieved {len(labels)} popular labels (limit: {limit})"
            )
            return labels
        except Exception as e:
            logger.error(f"[{self.workspace}] Error getting popular labels: {e}")
            raise

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        """Search labels with fuzzy matching. Uses FalkorDB full-text index with CONTAINS fallback."""
        workspace_label = self._get_workspace_label()
        query_strip = query.strip()
        if not query_strip:
            return []
        query_lower = query_strip.lower()
        is_chinese = self._is_chinese_text(query_strip)
        try:
            if is_chinese:
                cypher = (
                    f"CALL db.idx.fulltext.queryNodes('{workspace_label}', $sq) "
                    "YIELD node, score "
                    "WITH node.entity_id AS label, score, "
                    "CASE WHEN node.entity_id = $qs THEN score + 1000 "
                    "WHEN node.entity_id CONTAINS $qs THEN score + 500 "
                    "ELSE score END AS final_score "
                    "RETURN label ORDER BY final_score DESC, label ASC LIMIT $limit"
                )
                result = await self._graph.ro_query(
                    cypher, {"sq": query_strip, "qs": query_strip, "limit": limit}
                )
            else:
                cypher = (
                    f"CALL db.idx.fulltext.queryNodes('{workspace_label}', $sq) "
                    "YIELD node, score "
                    "WITH node.entity_id AS label, toLower(node.entity_id) AS ll, score "
                    "WITH label, ll, score, "
                    "CASE WHEN ll = $ql THEN score + 1000 "
                    "WHEN ll STARTS WITH $ql THEN score + 500 "
                    "ELSE score END AS final_score "
                    "RETURN label ORDER BY final_score DESC, label ASC LIMIT $limit"
                )
                result = await self._graph.ro_query(
                    cypher, {"sq": f"{query_strip}*", "ql": query_lower, "limit": limit}
                )
            labels = [row[0] for row in result.result_set]
            logger.debug(
                f"[{self.workspace}] Full-text search for '{query}' returned {len(labels)} results"
            )
            return labels
        except Exception as e:
            logger.warning(
                f"[{self.workspace}] Full-text search failed: {e}. Falling back to CONTAINS."
            )
        try:
            if is_chinese:
                cypher = (
                    f"MATCH (n:`{workspace_label}`) "
                    "WHERE n.entity_id IS NOT NULL AND n.entity_id CONTAINS $qs "
                    "WITH n.entity_id AS label, "
                    "CASE WHEN n.entity_id = $qs THEN 1000 "
                    "WHEN n.entity_id STARTS WITH $qs THEN 500 "
                    "ELSE 100 - size(n.entity_id) END AS score "
                    "ORDER BY score DESC, label ASC LIMIT $limit RETURN label"
                )
                result = await self._graph.ro_query(
                    cypher, {"qs": query_strip, "limit": limit}
                )
            else:
                cypher = (
                    f"MATCH (n:`{workspace_label}`) "
                    "WHERE n.entity_id IS NOT NULL AND toLower(n.entity_id) CONTAINS $ql "
                    "WITH n.entity_id AS label, toLower(n.entity_id) AS ll, "
                    "CASE WHEN toLower(n.entity_id) = $ql THEN 1000 "
                    "WHEN toLower(n.entity_id) STARTS WITH $ql THEN 500 "
                    "ELSE 100 - size(n.entity_id) END AS score "
                    "ORDER BY score DESC, label ASC LIMIT $limit RETURN label"
                )
                result = await self._graph.ro_query(
                    cypher, {"ql": query_lower, "limit": limit}
                )
            labels = [row[0] for row in result.result_set]
            logger.debug(
                f"[{self.workspace}] Fallback CONTAINS search for '{query}' returned {len(labels)} results"
            )
            return labels
        except Exception as e:
            logger.error(f"[{self.workspace}] Fallback search also failed: {e}")
            raise

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int = 1000,
    ) -> KnowledgeGraph:
        """
        Retrieve a subgraph rooted at node_label using iterative BFS.
        Pure-Cypher implementation — no APOC required.
        """
        result = KnowledgeGraph()
        workspace_label = self._get_workspace_label()

        try:
            check_query = (
                f"MATCH (n:`{workspace_label}` {{entity_id: $entity_id}}) "
                "RETURN n LIMIT 1"
            )
            check_result = await self._graph.ro_query(
                check_query, {"entity_id": node_label}
            )
            if not check_result.result_set:
                logger.warning(
                    f"[{self.workspace}] Node '{node_label}' not found in graph"
                )
                return result

            visited_nodes: set[str] = set()
            visited_edges: set[tuple[str, str]] = set()
            queue: deque[tuple[str, int]] = deque([(node_label, 0)])
            visited_nodes.add(node_label)

            while queue and len(visited_nodes) < max_nodes:
                current_id, depth = queue.popleft()

                node_query = (
                    f"MATCH (n:`{workspace_label}` {{entity_id: $entity_id}}) RETURN n"
                )
                node_result = await self._graph.ro_query(
                    node_query, {"entity_id": current_id}
                )
                if node_result.result_set:
                    node_obj = node_result.result_set[0][0]
                    props = dict(node_obj.properties)
                    result.nodes.append(
                        KnowledgeGraphNode(
                            id=props.get("entity_id", current_id),
                            labels=[workspace_label],
                            properties=props,
                        )
                    )

                if depth >= max_depth:
                    continue

                neighbors_query = (
                    f"MATCH (n:`{workspace_label}` {{entity_id: $entity_id}})"
                    f"-[r]-(neighbor:`{workspace_label}`) "
                    "WHERE neighbor.entity_id IS NOT NULL "
                    "RETURN neighbor.entity_id AS neighbor_id, "
                    "properties(r) AS rel_props, "
                    "startNode(r).entity_id AS start_id, "
                    "endNode(r).entity_id AS end_id"
                )
                neighbors_result = await self._graph.ro_query(
                    neighbors_query, {"entity_id": current_id}
                )

                for row in neighbors_result.result_set:
                    neighbor_id = row[0]
                    rel_props = dict(row[1]) if row[1] else {}
                    start_id = row[2]
                    end_id = row[3]

                    if len(visited_nodes) >= max_nodes:
                        break

                    edge_key = (
                        (start_id, end_id)
                        if start_id == current_id
                        else (end_id, start_id)
                    )
                    if edge_key not in visited_edges:
                        visited_edges.add(edge_key)
                        result.edges.append(
                            KnowledgeGraphEdge(
                                id=f"{edge_key[0]}-{edge_key[1]}",
                                type="DIRECTED",
                                source=edge_key[0],
                                target=edge_key[1],
                                properties=rel_props,
                            )
                        )

                    if neighbor_id not in visited_nodes:
                        visited_nodes.add(neighbor_id)
                        queue.append((neighbor_id, depth + 1))

            logger.info(
                f"[{self.workspace}] Knowledge graph for '{node_label}' "
                f"(depth={max_depth}): {len(result.nodes)} nodes, {len(result.edges)} edges"
            )
            return result

        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error getting knowledge graph for '{node_label}': {e}"
            )
            raise

    async def drop(self) -> dict[str, str]:
        """Delete all nodes and edges in this workspace's graph."""
        workspace_label = self._get_workspace_label()
        try:
            count_query = (
                f"MATCH (n:`{workspace_label}`) "
                "OPTIONAL MATCH (n)-[r]-() "
                "RETURN count(DISTINCT n) AS node_count, count(DISTINCT r) AS edge_count"
            )
            count_result = await self._graph.ro_query(count_query)
            node_count = 0
            edge_count = 0
            if count_result.result_set:
                node_count = int(count_result.result_set[0][0] or 0)
                edge_count = int(count_result.result_set[0][1] or 0)

            delete_query = (
                f"MATCH (n:`{workspace_label}`) DETACH DELETE n"
            )
            await self._graph.query(delete_query)

            logger.info(
                f"[{self.workspace}] Dropped workspace '{workspace_label}': "
                f"{node_count} nodes and {edge_count} edges deleted"
            )
            return {
                "status": "success",
                "message": (
                    f"Dropped {node_count} nodes and {edge_count} edges "
                    f"from workspace '{workspace_label}'"
                ),
            }
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error dropping workspace '{workspace_label}': {e}"
            )
            return {"status": "error", "message": str(e)}
