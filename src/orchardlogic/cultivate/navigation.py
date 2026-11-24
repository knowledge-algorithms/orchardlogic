"""Task sessions for navigating immutable cultivars: cohorts, ledgers, hopper.

This module is transport-free and catalog-generic. It owns:

* a task session with a pinned catalog;
* addressable cohorts with fixed membership, a recut view per cultivar,
  cohort-scoped aliases, and a provisional decision ledger;
* one exposure record and one hopper shared by all cohorts of the session;
* selective access to executable schemas of collected tools.

It deliberately owns no loaded-tool state and executes nothing.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from orchardlogic.cultivate.induced import (
    CUT_GUARANTEE,
    CUT_OPTIMAL,
    CutSettings,
    InducedView,
    recut,
)
from orchardlogic.cultivate.search import SearchIndex, SearchSyntaxError, build_index, parse_term
from orchardlogic.cultivate.tree import Tree

FILTER_ACTIONS = ("add", "keep_only", "drop", "clear")

COMMIT_WARNING = (
    "New cohort created. Use its returned aliases for continuation. "
    "Source references remain valid only with the source cohort ID."
)
EVENT_SCHEMA_VERSION = "orchard_navigation_event_v1"
SNAPSHOT_SCHEMA_VERSION = "orchard_navigation_snapshot_v1"


class NavigationError(Exception):
    """A rejected navigation request. ``code`` is stable; ``message`` is for people."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def to_dict(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message}}


# --------------------------------------------------------------------------- catalog


@dataclass(frozen=True)
class ToolItem:
    """One catalog item. ``schema`` is the executable definition, shown only on request."""

    tool_id: str
    description: str
    schema: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Cultivar:
    """One immutable hierarchy over the catalog plus its scoring inputs."""

    cultivar_id: str
    tree: Tree
    feature_matrix: np.ndarray
    alias_prefix: str
    settings: CutSettings = field(default_factory=CutSettings)
    labels: Mapping[str, str] = field(default_factory=dict)
    summaries: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Catalog:
    """A pinned set of items and the cultivars that organize them."""

    items: Mapping[str, ToolItem]
    cultivars: Mapping[str, Cultivar]
    name: str = "catalog"
    catalog_id: str = ""
    # Optional tool relations (orchard_relations_v1 "tools" mapping): per tool,
    # "needs" (required inputs and the tools that produce them) and "feeds" (tools
    # its outputs can fill). Empty = relation features are off.
    relations: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        item_ids = set(self.items)
        prefixes: set[str] = set()
        for cultivar in self.cultivars.values():
            if set(cultivar.tree.item_ids) != item_ids:
                raise ValueError(
                    f"cultivar {cultivar.cultivar_id!r} does not cover the catalog items"
                )
            if np.asarray(cultivar.feature_matrix).shape[0] != len(item_ids):
                raise ValueError(
                    f"cultivar {cultivar.cultivar_id!r} feature rows do not match items"
                )
            if not cultivar.alias_prefix.isalpha() or cultivar.alias_prefix in prefixes:
                raise ValueError("cultivar alias prefixes must be alphabetic and unique")
            prefixes.add(cultivar.alias_prefix)
        if not self.catalog_id:
            object.__setattr__(self, "catalog_id", self._fingerprint())

    def _fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(json.dumps(sorted(self.items)).encode())
        for cultivar_id in sorted(self.cultivars):
            cultivar = self.cultivars[cultivar_id]
            digest.update(cultivar_id.encode())
            digest.update("|".join(cultivar.tree.linkage_fingerprint()).encode())
            digest.update(json.dumps(cultivar.settings.to_dict(), sort_keys=True).encode())
            digest.update(np.ascontiguousarray(cultivar.feature_matrix).tobytes())
        if self.relations:
            digest.update(json.dumps(self.relations, sort_keys=True).encode())
        return f"{self.name}-{digest.hexdigest()[:12]}"


def _group_needs(needs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge inputs with the same source and producers: 'city, state, zip from X | Y'."""
    grouped: dict[tuple, dict[str, Any]] = {}
    for need in needs:
        key = (need["source"], tuple(p["tool_id"] for p in need.get("producers") or []))
        if key in grouped:
            grouped[key]["params"].append(need["param"])
        else:
            grouped[key] = {"params": [need["param"]], **{k: v for k, v in need.items() if k != "param"}}
    return list(grouped.values())


def default_alias_prefixes(cultivar_ids: Sequence[str]) -> dict[str, str]:
    """Short unique alphabetic prefixes, e.g. domain -> D, function -> F."""
    prefixes: dict[str, str] = {}
    used: set[str] = set()
    for cultivar_id in cultivar_ids:
        letters = [char.upper() for char in cultivar_id if char.isalpha()] or ["T"]
        candidate = letters[0]
        length = 1
        while candidate in used:
            length += 1
            candidate = "".join(letters[:length]) if length <= len(letters) else (
                letters[0] + "X" * (length - len(letters))
            )
        used.add(candidate)
        prefixes[cultivar_id] = candidate
    return prefixes


# --------------------------------------------------------------------------- cohorts


@dataclass
class CohortView:
    """One cultivar's recut view inside a cohort, with cohort-scoped aliases."""

    cultivar_id: str
    view: InducedView
    alias_by_node: dict[str, str] = field(default_factory=dict)
    node_by_alias: dict[str, str] = field(default_factory=dict)
    cut_nodes: dict[str, Mapping[str, Any]] = field(default_factory=dict)

    @classmethod
    def build(cls, cultivar: Cultivar, members: Iterable[str]) -> CohortView:
        view = recut(cultivar.tree, members, cultivar.feature_matrix, cultivar.settings)
        built = cls(cultivar_id=cultivar.cultivar_id, view=view)
        if view.cut is None:
            return built
        # Breadth-first so the top partition gets the lowest numbers. Assigned once
        # at creation, so aliases never depend on what was explored or in what order.
        queue = deque(view.cut["root"]["children"])
        number = 0
        while queue:
            node = queue.popleft()
            number += 1
            alias = f"{cultivar.alias_prefix}{number}"
            node_id = node["canonical_node_id"]
            built.alias_by_node[node_id] = alias
            built.node_by_alias[alias] = node_id
            built.cut_nodes[node_id] = node
            queue.extend(node["children"])
        return built

    def members_of(self, node_id: str) -> list[str]:
        return list(self.view.tree.nodes[node_id]["descendant_item_ids"])


@dataclass
class Cohort:
    cohort_id: str
    parent_id: str | None
    members: frozenset[str]
    views: dict[str, CohortView]
    # Staged filter steps, applied in order to the cohort's members. Each step is
    # {"action": add|keep_only|drop, "ref": canonical text, "key": hashable target}.
    steps: list[dict[str, Any]] = field(default_factory=list)
    revision: int = 0


# --------------------------------------------------------------------------- session


class Session:
    """One task-attempt navigation session over a pinned catalog."""

    def __init__(
        self,
        catalog: Catalog,
        *,
        session_id: str | None = None,
        task_id: str | None = None,
        strict_exposure: bool = True,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
        available_tools: Iterable[str] = (),
    ) -> None:
        self.catalog = catalog
        # Tools the host already provides (e.g. preloaded); the hopper check counts them.
        self.available_tools = frozenset(available_tools)
        self.session_id = session_id or f"S-{uuid.uuid4().hex[:10]}"
        self.task_id = task_id
        self.strict_exposure = strict_exposure
        self.closed = False
        self.cohorts: dict[str, Cohort] = {}
        self.continuation_cohort_id: str | None = None
        self.hopper: dict[str, dict[str, Any]] = {}
        self.hopper_revision = 0
        self.exposed_tools: dict[str, dict[str, Any]] = {}
        self.cluster_exposures: dict[tuple[str, str, str], int] = {}
        self.schema_exposures: dict[str, int] = {}
        self.rejected_ids: list[dict[str, Any]] = []
        self.counters = {
            "calls": 0,
            "cluster_displays": 0,
            "tool_description_displays": 0,
            "schema_displays": 0,
            "commits": 0,
            "recuts": 0,
            "response_chars": 0,
        }
        self.events: list[dict[str, Any]] = []
        self._event_sink = event_sink
        self._responses: dict[str, dict[str, Any]] = {}
        self._full_views: dict[str, CohortView] | None = None
        self._index: SearchIndex | None = None
        self._sequence = 0

    # ------------------------------------------------------------------ plumbing

    def _envelope(self, cohort: Cohort | None = None) -> dict[str, Any]:
        envelope = {"session_id": self.session_id, "catalog_id": self.catalog.catalog_id}
        if cohort is not None:
            envelope["cohort_id"] = cohort.cohort_id
        return envelope

    def _require_open(self) -> None:
        if self.closed:
            raise NavigationError("session_closed", "This task session is closed.")

    def _cohort(self, cohort_id: str | None) -> Cohort:
        if not cohort_id:
            raise NavigationError(
                "cohort_required",
                "Node operations need an explicit cohort_id. Call overview first.",
            )
        try:
            return self.cohorts[cohort_id]
        except KeyError:
            raise NavigationError(
                "unknown_cohort", f"No cohort {cohort_id!r} in this session."
            ) from None

    def record(self, command: str, arguments: Mapping[str, Any], response: Mapping[str, Any], elapsed_ms: float) -> None:
        """Append one trace event. Called by the dispatcher for every command."""
        self._sequence += 1
        size = len(json.dumps(response, ensure_ascii=False))
        self.counters["calls"] += 1
        self.counters["response_chars"] += size
        event = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "seq": self._sequence,
            "ts": time.time(),
            "session_id": self.session_id,
            "task_id": self.task_id,
            "command": command,
            "arguments": dict(arguments),
            "ok": "error" not in response,
            "error_code": response.get("error", {}).get("code") if "error" in response else None,
            "cohort_id": response.get("cohort_id"),
            "tool_count": response.get("tool_count"),
            "prospective_count": response.get("prospective_count"),
            "elapsed_ms": round(elapsed_ms, 3),
            "response_chars": size,
        }
        self.events.append(event)
        if self._event_sink is not None:
            self._event_sink(event)

    # ------------------------------------------------------------------ rendering

    def _expose_tool(self, tool_id: str, cohort: Cohort, via: str) -> dict[str, Any]:
        self.counters["tool_description_displays"] += 1
        self.exposed_tools.setdefault(
            tool_id,
            {"seq": self._sequence + 1, "cohort_id": cohort.cohort_id, "via": via},
        )
        return {"tool_id": tool_id, "description": self.catalog.items[tool_id].description}

    def _node_entry(self, cohort: Cohort, view: CohortView, node_id: str) -> dict[str, Any]:
        cultivar = self.catalog.cultivars[view.cultivar_id]
        provenance = view.view.provenance[node_id]
        key = (cohort.cohort_id, view.cultivar_id, node_id)
        self.cluster_exposures[key] = self.cluster_exposures.get(key, 0) + 1
        self.counters["cluster_displays"] += 1
        source_id = provenance.source_node_id
        entry: dict[str, Any] = {
            "ref": view.alias_by_node[node_id],
            "label": cultivar.labels.get(source_id) or f"Unlabeled group {source_id[:14]}",
            "summary": cultivar.summaries.get(source_id, ""),
            "tools": provenance.surviving_count,
            "retained_pct": round(100.0 * provenance.retained_fraction, 1),
            "restricted": provenance.restricted,
        }
        if provenance.restricted:
            entry["scope_note"] = (
                f"Summary describes the original group of {provenance.source_count} "
                f"tools; {provenance.surviving_count} remain in this cohort."
            )
        staged = [s["action"] for s in cohort.steps if s["key"] == ("node", view.cultivar_id, node_id)]
        if staged:
            entry["staged"] = staged[-1]
        return entry

    def _match_groups(self, cohort: Cohort, tool_ids: set[str]) -> list[dict[str, Any]]:
        """Top-level groups of each cultivar holding any of tool_ids, largest count
        first (ties in alias order). Labels only; not counted as cluster displays."""
        groups: list[dict[str, Any]] = []
        if not tool_ids:
            return groups
        for view in cohort.views.values():
            if view.view.cut_status != CUT_OPTIMAL:
                continue
            cultivar = self.catalog.cultivars[view.cultivar_id]
            found = []
            for child in view.view.cut["root"]["children"]:
                node_id = child["canonical_node_id"]
                count = len(tool_ids.intersection(view.members_of(node_id)))
                if count:
                    source_id = view.view.provenance[node_id].source_node_id
                    label = cultivar.labels.get(source_id) or f"Unlabeled group {source_id[:14]}"
                    found.append({
                        "cultivar": view.cultivar_id,
                        "ref": view.alias_by_node[node_id],
                        "label": label,
                        "count": count,
                    })
            found.sort(key=lambda g: -g["count"])  # stable: alias order within a count
            groups.extend(found)
        return groups

    def _ordered(self, view: CohortView, tool_ids: Iterable[str]) -> list[str]:
        wanted = set(tool_ids)
        return [item_id for item_id in view.view.tree.item_ids if item_id in wanted]

    def _ledger_summary(self, cohort: Cohort) -> dict[str, Any]:
        return {
            "revision": cohort.revision,
            "steps": [f"{s['action']} {s['ref']}" for s in cohort.steps],
            "prospective_count": len(self._selection(cohort)),
        }

    # ------------------------------------------------------------------ selection

    def _search_index(self) -> SearchIndex:
        if self._index is None:
            self._index = build_index(
                (tool_id, item.description) for tool_id, item in self.catalog.items.items()
            )
        return self._index

    def _step_members(self, cohort: Cohort, key: tuple) -> set[str]:
        kind = key[0]
        if kind == "node":
            return set(cohort.views[key[1]].members_of(key[2]))
        if kind == "search":
            return set(self._search_index().match(key[1]))
        if kind == "link":
            return self._linked(key[1], key[2])
        return {key[1]}  # one tool

    # ------------------------------------------------------------------ relations

    def _relations(self) -> Mapping[str, Any]:
        if not self.catalog.relations:
            raise NavigationError(
                "relations_unavailable", "This catalog has no tool relations (needs/feeds)."
            )
        return self.catalog.relations

    def _linked(self, kind: str, tool_id: str) -> set[str]:
        """needs: producers of the tool's required inputs; feeds: tools its outputs fill."""
        entry = self._relations().get(tool_id) or {}
        if kind == "needs":
            return {p["tool_id"] for n in entry.get("needs", []) for p in n.get("producers") or []}
        return {f["tool_id"] for f in entry.get("feeds", [])}

    def _expose_listed(self, tool_id: str, via: str) -> dict[str, Any]:
        """A tool listed by a relation reply counts as shown (its description is in the reply)."""
        self.counters["tool_description_displays"] += 1
        self.exposed_tools.setdefault(tool_id, {"seq": self._sequence + 1, "cohort_id": None, "via": via})
        return {"tool_id": tool_id, "description": self.catalog.items[tool_id].description}

    def _need_entry(self, need: Mapping[str, Any], via: str, cap: int,
                    present: set[str] | None = None) -> dict[str, Any]:
        entry: dict[str, Any] = {"param": need["param"], "source": need.get("source", "task")}
        producers = [p for p in need.get("producers") or [] if p["tool_id"] in self.catalog.items]
        if entry["source"] == "tools":
            if present is not None:
                entry["have"] = [p["tool_id"] for p in producers if p["tool_id"] in present]
            shown = [p for p in producers if present is None or p["tool_id"] not in present][:cap]
            entry["producers"] = [
                {**self._expose_listed(p["tool_id"], via), "via": p.get("via", "")} for p in shown
            ]
            entry["more"] = max(0, len(producers) - len(shown) - len(entry.get("have", [])))
        else:
            entry["producer_count"] = need.get("producer_count", 0)
            if producers:  # ranked top producers of an input usually supplied by the task
                entry["producers"] = [
                    {**self._expose_listed(p["tool_id"], via), "via": p.get("via", "")} for p in producers[:cap]
                ]
                entry["more"] = max(0, entry["producer_count"] - len(entry["producers"]))
        return entry

    def neighborhood(self, tool_id: str | None, *, max_producers: int = 3, max_feeds: int = 5) -> dict[str, Any]:
        """What a tool needs (required inputs and the tools that produce them) and which
        tools its outputs can fill. Listed tools count as shown, so they can be collected."""
        self._require_open()
        relations = self._relations()
        if not tool_id or tool_id not in self.catalog.items:
            raise NavigationError("unknown_tool", f"{tool_id!r} is not a tool of this catalog.")
        entry = relations.get(tool_id) or {"needs": [], "feeds": []}
        via = f"neighborhood:{tool_id}"
        response = self._envelope()
        response["tool_id"] = tool_id
        # the inspected tool is shown too (its description is in the reply), so it can be loaded
        response["description"] = self._expose_listed(tool_id, via)["description"]
        response["needs"] = _group_needs(
            [self._need_entry(n, via, max_producers) for n in entry.get("needs", [])]
        )
        # Feeds sharing one link (e.g. a token that fills every tool of an app) are
        # summarized as a count with two examples instead of listed one by one.
        by_link: dict[tuple, list[dict[str, Any]]] = {}
        for feed in entry.get("feeds", []):
            if feed["tool_id"] in self.catalog.items:
                by_link.setdefault(tuple(feed.get("via") or ()), []).append(feed)
        feeds, summaries = [], []
        for link, group in by_link.items():
            if len(group) > max_feeds:
                summaries.append({
                    "via": list(link), "count": len(group),
                    "examples": [self._expose_listed(f["tool_id"], via) for f in group[:2]],
                })
            else:
                feeds.extend(group)
        response["feeds"] = [
            {**self._expose_listed(f["tool_id"], via), "via": list(f.get("via") or [])}
            for f in feeds[:max_feeds]
        ]
        response["feeds_more"] = max(0, len(feeds) - max_feeds)
        if summaries:
            response["feeds_shared"] = summaries
        return response

    def hopper_check(self, have: Sequence[str] | None = None, *, max_producers: int = 3) -> dict[str, Any]:
        """Graph check over the hopper: required inputs of collected tools that no
        collected (or already available) tool produces, with candidate producers."""
        relations = self._relations()
        present = set(self.hopper) | set(self.available_tools) | set(have or [])
        gaps: dict[tuple, dict[str, Any]] = {}
        from_task: dict[str, list[str]] = {}
        satisfied = 0
        for tool_id in self.hopper:
            for need in (relations.get(tool_id) or {}).get("needs", []):
                if need.get("source") != "tools":
                    from_task.setdefault(tool_id, []).append(need["param"])
                    continue
                producers = tuple(p["tool_id"] for p in need.get("producers") or [])
                if set(producers) & present:
                    satisfied += 1
                    continue
                # one gap per (input, producer set): a token missing for three tools is one gap
                key = (need["param"], producers)
                if key not in gaps:
                    gaps[key] = {"tool_ids": [], **self._need_entry(need, "hopper_check", max_producers, present)}
                gaps[key]["tool_ids"].append(tool_id)
        return {"gaps": list(gaps.values()), "satisfied_inputs": satisfied, "from_task": from_task,
                "checked_tools": len(self.hopper)}

    def _selection(self, cohort: Cohort) -> set[str]:
        """Apply the staged steps in order, starting from all cohort members:
        add unions, keep_only intersects, drop subtracts."""
        result = set(cohort.members)
        for step in cohort.steps:
            members = self._step_members(cohort, step["key"])
            if step["action"] == "add":
                result |= members
            elif step["action"] == "keep_only":
                result &= members
            else:
                result -= members
        return result

    def _sources(self) -> str:
        extra = ["needs", "feeds"] if self.catalog.relations else []
        return ", ".join([*self.catalog.cultivars, "search", *extra])

    def _resolve_ref(self, cohort: Cohort, ref: Any) -> tuple[CohortView, str]:
        """A group ref: 'D3' or 'domain.D3' (source.value)."""
        if not isinstance(ref, str) or not ref:
            raise NavigationError("invalid_ref", f"Reference {ref!r} is not an alias.")
        source, dot, value = ref.partition(".")
        if dot:
            if source == "search":
                raise NavigationError(
                    "invalid_ref", f"{ref!r} is a search ref; search refs work in filter only."
                )
            if source not in cohort.views:
                raise NavigationError(
                    "unknown_source",
                    f"Unknown source {source!r} in {ref!r}. Sources: {self._sources()}.",
                )
            views = [cohort.views[source]]
        else:
            value, views = ref, list(cohort.views.values())
        for view in views:
            node_id = view.node_by_alias.get(value)
            if node_id is not None:
                return view, node_id
        raise NavigationError(
            "unknown_ref",
            f"{ref!r} is not an alias of cohort {cohort.cohort_id}. Aliases are "
            "cohort-scoped; pair each alias with the cohort that returned it.",
        )

    def _resolve_step(self, cohort: Cohort, ref: Any) -> tuple[tuple, str]:
        """(key, canonical ref) for a filter ref: a group or a search term."""
        if isinstance(ref, str) and ref.startswith("search."):
            try:
                term = parse_term(ref[len("search."):])
            except SearchSyntaxError as exc:
                raise NavigationError("invalid_search", str(exc)) from None
            return ("search", term), term.canonical
        if isinstance(ref, str) and ref.startswith(("needs.", "feeds.")):
            kind, _, tool_id = ref.partition(".")
            self._relations()
            if tool_id not in self.catalog.items:
                raise NavigationError("unknown_tool", f"{tool_id!r} in {ref!r} is not a tool of this catalog.")
            return ("link", kind, tool_id), ref
        view, node_id = self._resolve_ref(cohort, ref)
        return ("node", view.cultivar_id, node_id), f"{view.cultivar_id}.{view.alias_by_node[node_id]}"

    # ------------------------------------------------------------------ cohorts

    def _build_views(self, members: frozenset[str]) -> dict[str, CohortView]:
        views = {}
        for cultivar_id, cultivar in self.catalog.cultivars.items():
            views[cultivar_id] = CohortView.build(cultivar, members)
            self.counters["recuts"] += 1
        return views

    def _new_cohort(self, members: frozenset[str], parent_id: str | None, views: dict[str, CohortView]) -> Cohort:
        cohort = Cohort(
            cohort_id=f"C{len(self.cohorts) + 1}",
            parent_id=parent_id,
            members=members,
            views=views,
        )
        self.cohorts[cohort.cohort_id] = cohort
        self.continuation_cohort_id = cohort.cohort_id
        return cohort

    def _render_cohort(self, cohort: Cohort) -> dict[str, Any]:
        response = self._envelope(cohort)
        response["tool_count"] = len(cohort.members)
        if cohort.parent_id:
            response["parent_cohort_id"] = cohort.parent_id
        cultivars = []
        show_individuals = False
        for view in cohort.views.values():
            block: dict[str, Any] = {
                "cultivar": view.cultivar_id,
                "cut_status": view.view.cut_status,
            }
            if view.view.cut_status == CUT_OPTIMAL:
                root = view.view.cut["root"]
                block["groups"] = [
                    self._node_entry(cohort, view, child["canonical_node_id"])
                    for child in root["children"]
                ]
                leaves = [
                    view.view.tree.nodes[node_id]["item_id"]
                    for node_id in root["direct_leaf_node_ids"]
                ]
                if leaves:
                    block["tools"] = [
                        self._expose_tool(tool_id, cohort, "overview")
                        for tool_id in self._ordered(view, leaves)
                    ]
            else:
                block["note"] = view.view.cut_reason
                show_individuals = True
            cultivars.append(block)
        response["cultivars"] = cultivars
        if show_individuals:
            first = next(iter(cohort.views.values()))
            response["tools"] = [
                self._expose_tool(tool_id, cohort, "overview")
                for tool_id in self._ordered(first, cohort.members)
            ]
        response["ledger"] = self._ledger_summary(cohort)
        return response

    # ------------------------------------------------------------------ commands

    def status(self) -> dict[str, Any]:
        response = self._envelope()
        response.update(
            {
                "task_id": self.task_id,
                "open": not self.closed,
                "continuation_cohort_id": self.continuation_cohort_id,
                "counts": {
                    "cohorts": len(self.cohorts),
                    "hopper": len(self.hopper),
                    "tools_exposed": len(self.exposed_tools),
                    "groups_exposed": len(self.cluster_exposures),
                    "schemas_exposed": len(self.schema_exposures),
                    "calls": self.counters["calls"],
                },
            }
        )
        return response

    def close(self) -> dict[str, Any]:
        response = self.status()
        self.closed = True
        self.hopper.clear()
        self.hopper_revision += 1
        response["open"] = False
        response["note"] = "Session closed: cohort access ended and hopper cleared."
        return response

    def overview(self, cohort_id: str | None = None) -> dict[str, Any]:
        self._require_open()
        if cohort_id in (None, "", "new"):
            members = frozenset(self.catalog.items)
            if self._full_views is None:
                self._full_views = self._build_views(members)
            # Fresh cohorts share the immutable full-catalog recut but own their
            # alias maps and ledger.
            views = {
                cultivar_id: CohortView(
                    cultivar_id=cultivar_id,
                    view=view.view,
                    alias_by_node=dict(view.alias_by_node),
                    node_by_alias=dict(view.node_by_alias),
                    cut_nodes=view.cut_nodes,
                )
                for cultivar_id, view in self._full_views.items()
            }
            cohort = self._new_cohort(members, None, views)
        else:
            cohort = self._cohort(cohort_id)
        response = self._render_cohort(cohort)
        response["cut_guarantee"] = CUT_GUARANTEE
        return response

    def explore(self, cohort_id: str | None, refs: Sequence[str]) -> dict[str, Any]:
        self._require_open()
        cohort = self._cohort(cohort_id)
        if not refs:
            raise NavigationError("refs_required", "explore needs at least one ref.")
        resolved = [(ref, *self._resolve_ref(cohort, ref)) for ref in refs]
        response = self._envelope(cohort)
        results = []
        for ref, view, node_id in resolved:
            cultivar = self.catalog.cultivars[view.cultivar_id]
            entry = self._node_entry(cohort, view, node_id)
            entry["cultivar"] = view.cultivar_id
            cut_node = view.cut_nodes[node_id]
            members = view.members_of(node_id)
            children = cut_node["children"]
            if len(members) <= cultivar.settings.individual_threshold or not children:
                leaves = members
            else:
                entry["groups"] = [
                    self._node_entry(cohort, view, child["canonical_node_id"])
                    for child in children
                ]
                leaves = [
                    view.view.tree.nodes[leaf_id]["item_id"]
                    for leaf_id in cut_node["direct_leaf_node_ids"]
                ]
            if leaves:
                entry["tool_cards"] = [
                    self._expose_tool(tool_id, cohort, ref)
                    for tool_id in self._ordered(view, leaves)
                ]
            results.append(entry)
        response["results"] = results
        return response

    def filter(
        self,
        cohort_id: str | None,
        decisions: Sequence[Mapping[str, Any]] | None = None,
        preview: bool = False,
        *,
        request_id: str | None = None,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        self._require_open()
        if request_id and request_id in self._responses:
            replay = dict(self._responses[request_id])
            replay["replayed"] = True
            return replay
        cohort = self._cohort(cohort_id)
        if expected_revision is not None and expected_revision != cohort.revision:
            raise NavigationError(
                "stale_revision",
                f"Ledger revision is {cohort.revision}, not {expected_revision}.",
            )

        # Validate the whole batch before touching the staged steps.
        staged: list[dict[str, Any]] = []
        searches: list[dict[str, Any]] = []
        for decision in decisions or []:
            if not isinstance(decision, Mapping):
                raise NavigationError("invalid_decision", "Each decision must be an object.")
            action = decision.get("action")
            if action not in FILTER_ACTIONS:
                raise NavigationError(
                    "invalid_decision",
                    f"action must be add, keep_only, drop, or clear (got {action!r}).",
                )
            if decision.get("ref"):
                key, ref = self._resolve_step(cohort, decision["ref"])
                if key[0] == "search":
                    matched = self._search_index().match(key[1])
                    # Counts only: search results reach the agent as cohort
                    # membership (after commit), never as a list of tools. The
                    # cohort's top-level groups holding the in-cohort matches point
                    # back into the tree.
                    in_cohort = cohort.members.intersection(matched)
                    searches.append({
                        "ref": ref,
                        "count": len(matched),
                        "in_cohort": len(in_cohort),
                        "groups": self._match_groups(cohort, in_cohort),
                    })
                staged.append({"action": action, "ref": ref, "key": key})
            elif decision.get("tool_id"):
                tool_id = decision["tool_id"]
                if action not in ("drop", "clear"):
                    raise NavigationError(
                        "invalid_decision",
                        "Individual tools can be dropped or cleared only. Collect a "
                        "wanted tool with hopper(add).",
                    )
                if tool_id not in cohort.members:
                    raise NavigationError(
                        "invalid_decision",
                        f"{tool_id!r} is not a member of cohort {cohort.cohort_id}.",
                    )
                if self.strict_exposure and tool_id not in self.exposed_tools:
                    self._reject([tool_id], "not_exposed", "filter")
                    raise NavigationError(
                        "not_exposed",
                        f"{tool_id!r} has not been shown individually in this session.",
                    )
                staged.append({"action": action, "ref": tool_id, "key": ("tool", tool_id)})
            else:
                raise NavigationError("invalid_decision", "A decision needs ref or tool_id.")

        for step in staged:
            # clear removes every staged step on that target; a repeated step moves
            # to the end, since steps apply in order.
            cohort.steps = [
                s for s in cohort.steps
                if not (s["key"] == step["key"] and (step["action"] == "clear" or s["action"] == step["action"]))
            ]
            if step["action"] != "clear":
                cohort.steps.append(step)
        if staged:
            cohort.revision += 1

        selection = self._selection(cohort)
        response = self._envelope(cohort)
        response["prospective_count"] = len(selection)
        response["committed"] = False
        if searches:
            response["search_matches"] = searches
        if preview:
            response["ledger"] = self._ledger_summary(cohort)
        elif not selection:
            # Clear the staged steps so the next filter starts empty instead of
            # stacking on the combination that produced zero.
            response["cleared_ledger"] = self._ledger_summary(cohort)
            cohort.steps.clear()
            cohort.revision += 1
            response["warning"] = (
                "Filtering would leave zero tools. No new cohort was created and the "
                "staged steps were cleared."
            )
        elif selection == set(cohort.members):
            response["warning"] = (
                f"The staged steps keep exactly the {len(cohort.members)} tools of this "
                "cohort (steps start from all of them; add only widens), so no new "
                "cohort was created."
            )
            response["ledger"] = self._ledger_summary(cohort)
        else:
            members = frozenset(selection)
            try:
                views = self._build_views(members)
            except Exception as exc:  # recut failure: source and ledger stay intact
                response["warning"] = (
                    f"Recut failed ({exc}). No new cohort was created and the staged decisions are kept."
                )
                response["ledger"] = self._ledger_summary(cohort)
            else:
                child = self._new_cohort(members, cohort.cohort_id, views)
                applied = [f"{s['action']} {s['ref']}" for s in cohort.steps]
                cohort.steps.clear()
                cohort.revision += 1
                self.counters["commits"] += 1
                response = self._render_cohort(child)
                response["applied_steps"] = applied
                if searches:
                    response["search_matches"] = searches
                response["committed"] = True
                response["source_cohort_id"] = cohort.cohort_id
                response["prospective_count"] = len(members)
                response["warning"] = COMMIT_WARNING
        if request_id:
            self._responses[request_id] = response
        return response

    # ------------------------------------------------------------------ hopper

    def _reject(self, tool_ids: Iterable[str], reason: str, command: str) -> list[dict[str, str]]:
        rejected = []
        for tool_id in tool_ids:
            record = {"tool_id": str(tool_id), "reason": reason}
            rejected.append(record)
            self.rejected_ids.append({**record, "command": command, "seq": self._sequence + 1})
        return rejected

    def hopper_command(
        self,
        action: str,
        tool_ids: Sequence[str] | None = None,
        *,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        self._require_open()
        if request_id and request_id in self._responses:
            replay = dict(self._responses[request_id])
            replay["replayed"] = True
            return replay
        tool_ids = list(dict.fromkeys(tool_ids or []))
        response = self._envelope()
        if action == "list":
            pass
        elif action == "clear":
            if self.hopper:
                self.hopper.clear()
                self.hopper_revision += 1
        elif action == "add":
            if not tool_ids:
                raise NavigationError("tool_ids_required", "hopper add needs tool_ids.")
            added, rejected = [], []
            for tool_id in tool_ids:
                if tool_id not in self.catalog.items:
                    rejected += self._reject([tool_id], "unknown_tool", "hopper")
                elif self.strict_exposure and tool_id not in self.exposed_tools:
                    rejected += self._reject([tool_id], "not_exposed", "hopper")
                elif tool_id not in self.hopper:
                    self.hopper[tool_id] = {"seq": self._sequence + 1}
                    added.append(tool_id)
            if added:
                self.hopper_revision += 1
            response["added"] = added
            if rejected:
                response["rejected"] = rejected
                response["note"] = (
                    "Only tools shown individually in this session can be collected."
                )
        elif action == "remove":
            if not tool_ids:
                raise NavigationError("tool_ids_required", "hopper remove needs tool_ids.")
            removed = [tool_id for tool_id in tool_ids if self.hopper.pop(tool_id, None)]
            if removed:
                self.hopper_revision += 1
            response["removed"] = removed
        elif action == "check":
            # tool_ids here are tools the caller already has outside the hopper
            response["check"] = self.hopper_check(
                [t for t in tool_ids if t in self.catalog.items]
            )
        else:
            raise NavigationError(
                "invalid_action", "hopper action must be add, remove, list, clear, or check."
            )
        response["hopper"] = list(self.hopper)
        response["hopper_revision"] = self.hopper_revision
        if request_id:
            self._responses[request_id] = response
        return response

    def schemas(self, tool_ids: Sequence[str]) -> dict[str, Any]:
        self._require_open()
        tool_ids = list(dict.fromkeys(tool_ids or []))
        if not tool_ids:
            raise NavigationError("tool_ids_required", "schemas needs explicit tool_ids.")
        response = self._envelope()
        found, rejected = [], []
        for tool_id in tool_ids:
            if tool_id not in self.hopper:
                reason = "unknown_tool" if tool_id not in self.catalog.items else "not_collected"
                rejected += self._reject([tool_id], reason, "schemas")
                continue
            self.schema_exposures[tool_id] = self.schema_exposures.get(tool_id, 0) + 1
            self.counters["schema_displays"] += 1
            item = self.catalog.items[tool_id]
            found.append(
                {"tool_id": tool_id, "description": item.description, **dict(item.schema)}
            )
        response["schemas"] = found
        if rejected:
            response["rejected"] = rejected
        response["note"] = "Schema access does not authorize execution."
        return response

    # ------------------------------------------------------------------ snapshot

    def snapshot(self) -> dict[str, Any]:
        """Complete session state for a private evaluator. Never shown to the agent."""
        return {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "session_id": self.session_id,
            "catalog_id": self.catalog.catalog_id,
            "task_id": self.task_id,
            "open": not self.closed,
            "continuation_cohort_id": self.continuation_cohort_id,
            "cohorts": [
                {
                    "cohort_id": cohort.cohort_id,
                    "parent_id": cohort.parent_id,
                    "tool_count": len(cohort.members),
                    "members": sorted(cohort.members),
                    "ledger_revision": cohort.revision,
                }
                for cohort in self.cohorts.values()
            ],
            "hopper": list(self.hopper),
            "hopper_revision": self.hopper_revision,
            "exposed_tools": dict(self.exposed_tools),
            "group_exposures": [
                {"cohort_id": c, "cultivar": t, "node_id": n, "displays": count}
                for (c, t, n), count in self.cluster_exposures.items()
            ],
            "schema_exposures": dict(self.schema_exposures),
            "rejected_ids": list(self.rejected_ids),
            "counters": dict(self.counters),
            "event_count": len(self.events),
        }


class SessionManager:
    """Owns isolated task sessions over one pinned catalog and dispatches commands."""

    def __init__(
        self,
        catalog: Catalog,
        *,
        strict_exposure: bool = True,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
        available_tools: Iterable[str] = (),
    ) -> None:
        self.catalog = catalog
        self.strict_exposure = strict_exposure
        self.event_sink = event_sink
        self.available_tools = frozenset(available_tools)
        self.sessions: dict[str, Session] = {}

    def open(self, task_id: str | None = None) -> Session:
        session = Session(
            self.catalog,
            task_id=task_id,
            strict_exposure=self.strict_exposure,
            event_sink=self.event_sink,
            available_tools=self.available_tools,
        )
        self.sessions[session.session_id] = session
        return session

    def get(self, session_id: str | None) -> Session:
        if session_id:
            try:
                return self.sessions[session_id]
            except KeyError:
                raise NavigationError(
                    "unknown_session", f"No session {session_id!r}."
                ) from None
        live = [session for session in self.sessions.values() if not session.closed]
        if len(live) == 1:
            return live[0]
        if not live:
            raise NavigationError(
                "no_session", "No open session. Call session with action='open' first."
            )
        raise NavigationError(
            "session_required", "Several sessions are open; pass session_id."
        )

    def dispatch(self, command: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Run one command, returning a JSON-ready response; errors become payloads."""
        arguments = dict(arguments or {})
        started = time.perf_counter()
        session: Session | None = None
        try:
            if command == "session":
                action = arguments.get("action", "status")
                if action == "open":
                    session = self.open(arguments.get("task_id"))
                    response = session.status()
                elif action == "status":
                    session = self.get(arguments.get("session_id"))
                    response = session.status()
                elif action == "close":
                    session = self.get(arguments.get("session_id"))
                    response = session.close()
                else:
                    raise NavigationError(
                        "invalid_action", "session action must be open, status, or close."
                    )
            else:
                session = self.get(arguments.get("session_id"))
                if command == "overview":
                    response = session.overview(arguments.get("cohort_id"))
                elif command == "explore":
                    response = session.explore(
                        arguments.get("cohort_id"), arguments.get("refs") or []
                    )
                elif command == "filter":
                    response = session.filter(
                        arguments.get("cohort_id"),
                        arguments.get("decisions") or [],
                        bool(arguments.get("preview", False)),
                        request_id=arguments.get("request_id"),
                        expected_revision=arguments.get("expected_revision"),
                    )
                elif command == "hopper":
                    response = session.hopper_command(
                        arguments.get("action", "list"),
                        arguments.get("tool_ids"),
                        request_id=arguments.get("request_id"),
                    )
                elif command == "neighborhood":
                    response = session.neighborhood(arguments.get("tool_id"))
                elif command == "schemas":
                    response = session.schemas(arguments.get("tool_ids") or [])
                elif command == "snapshot":
                    response = session.snapshot()
                else:
                    raise NavigationError("unknown_command", f"Unknown command {command!r}.")
        except NavigationError as exc:
            response = exc.to_dict()
            if session is not None:
                response = {**session._envelope(), **response}
        if session is not None and command != "snapshot":
            session.record(command, arguments, response, (time.perf_counter() - started) * 1000)
        return response
