"""Live MCP client for Project 1 standards and video evidence."""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from typing import Any

from .scenario_retrieval import RetrievedContext
from .settings import Settings, settings


class Project1McpUnavailable(RuntimeError):
    """Controlled error used to trigger an explicit retrieval fallback."""


class Project1McpClient:
    def __init__(self, config: Settings = settings) -> None:
        self.config = config

    def _validate(self) -> None:
        if not self.config.project1_mcp_enabled:
            raise Project1McpUnavailable("Project 1 MCP retrieval is disabled.")
        server_path = self.config.project1_mcp_project_dir / self.config.project1_mcp_server
        if not self.config.project1_mcp_project_dir.exists():
            raise Project1McpUnavailable("Project 1 MCP project directory is unavailable.")
        if not self.config.project1_mcp_python.exists():
            raise Project1McpUnavailable("Project 1 MCP Python environment is unavailable.")
        if not server_path.exists():
            raise Project1McpUnavailable("Project 1 MCP server is unavailable.")

    def call_tools(self, requests: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
        """Execute several tools in one stdio server session."""
        self._validate()
        try:
            return asyncio.run(self._call_tools(requests))
        except Project1McpUnavailable:
            raise
        except Exception as exc:
            raise Project1McpUnavailable("Project 1 MCP retrieval failed.") from exc

    async def _call_tools(
        self, requests: list[tuple[str, dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            raise Project1McpUnavailable(
                "Project 1 MCP client dependency is not installed."
            ) from exc

        parameters = StdioServerParameters(
            command=str(self.config.project1_mcp_python),
            args=[self.config.project1_mcp_server],
            cwd=str(self.config.project1_mcp_project_dir),
        )
        timeout = timedelta(seconds=self.config.project1_mcp_timeout)
        payloads: list[dict[str, Any]] = []
        async with stdio_client(parameters) as streams:
            async with ClientSession(*streams, read_timeout_seconds=timeout) as session:
                await session.initialize()
                for tool_name, arguments in requests:
                    result = await session.call_tool(
                        tool_name, arguments, read_timeout_seconds=timeout
                    )
                    if result.isError:
                        raise Project1McpUnavailable(
                            f"Project 1 MCP tool returned an error: {tool_name}"
                        )
                    payloads.append(self._result_payload(result))
        return payloads

    @staticmethod
    def _result_payload(result: Any) -> dict[str, Any]:
        if isinstance(result.structuredContent, dict):
            return result.structuredContent
        for item in result.content:
            text = getattr(item, "text", None)
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        return {"status": "ok", "results": []}

    def status(self) -> dict[str, Any]:
        if not self.config.project1_mcp_enabled:
            return {"enabled": False, "connected": False, "status": "disabled"}
        try:
            payload = self.call_tools([("get_knowledge_base_status", {})])[0]
        except Project1McpUnavailable as exc:
            return {
                "enabled": True,
                "connected": False,
                "status": "unavailable",
                "message": str(exc),
            }
        return {
            "enabled": True,
            "connected": payload.get("status") == "ok",
            "status": payload.get("status", "unknown"),
            "databases": payload.get("databases", {}),
        }

    def retrieve(
        self,
        *,
        query_plan: dict[str, list[str]],
        has_failure_evidence: bool,
    ) -> tuple[list[RetrievedContext], list[RetrievedContext]]:
        """Retrieve standards and video evidence through one live MCP session."""
        if not has_failure_evidence:
            return [], []
        standard_specs = [
            ("ISO 21448", "sotif", "STD-SOTIF"),
            ("ISO 8800", "iso_8800", "STD-8800"),
            ("ISO 26262", "iso_26262", "STD-26262"),
        ]
        requests: list[tuple[str, dict[str, Any]]] = []
        descriptors: list[tuple[str, str, str]] = []
        for standard, query_key, prefix in standard_specs:
            requests.append(
                (
                    "search_safety_standards",
                    {
                        "query": " ".join(query_plan[query_key]),
                        "k": max(1, min(self.config.project1_results_per_source, 10)),
                        "standard": standard,
                        "embedding_backend": self.config.project1_mcp_embedding_backend,
                    },
                )
            )
            descriptors.append(("standards_guidance", standard, prefix))
        if self.config.project1_mcp_video_enabled:
            requests.append(
                (
                    "search_video_evidence",
                    {
                        "query": " ".join(query_plan["failure_mechanism"]),
                        "k": max(1, min(self.config.project1_results_per_source, 10)),
                        "failure_cases_only": False,
                    },
                )
            )
            descriptors.append(("failure_mechanisms", "Project 1 video evidence", "MCP-VID"))

        payloads = self.call_tools(requests)
        standards: list[RetrievedContext] = []
        videos: list[RetrievedContext] = []
        for payload, (layer, title, prefix) in zip(payloads, descriptors):
            if payload.get("status") != "ok":
                continue
            for index, item in enumerate(payload.get("results", []), start=1):
                metadata = dict(item.get("metadata") or {})
                context = RetrievedContext(
                    evidence_id=f"{prefix}-{index}",
                    title=str(metadata.get("standard") or metadata.get("title") or title),
                    source_path=None,
                    layer=layer,
                    score=round(1.0 / index, 4),
                    matched_terms=[],
                    excerpt=str(item.get("content") or ""),
                    retrieval_reason="Retrieved from the live Project 1 knowledge service.",
                    source_type=str(item.get("source_type") or "project1_mcp"),
                    retrieval_method="project1_mcp",
                    metadata={**metadata, "rank": item.get("rank", index)},
                )
                if layer == "standards_guidance":
                    standards.append(context)
                else:
                    videos.append(context)
        return standards, videos
