"""Load a navigation :class:`~orchard.navigation.Catalog` from saved artifacts.

A catalog manifest (``orchard_mcp_catalog_v1``) pins an Orchard artifact
directory, one feature matrix per cultivar, the label/card set to display, and
cut settings. Relative paths resolve against the manifest's directory. Nothing
here is specific to any one corpus.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.cluster.hierarchy import linkage as scipy_linkage

from orchardlogic.cultivate.induced import CutSettings
from orchardlogic.cultivate.navigation import Catalog, Cultivar, ToolItem, default_alias_prefixes
from orchardlogic.cultivate.orchard import Orchard
from orchardlogic.cultivate.tree import Tree

CATALOG_MANIFEST_SCHEMA_VERSION = "orchard_mcp_catalog_v1"

_JSON_TYPES = {
    "str": "string",
    "string": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "dict": "object",
    "object": "object",
    "list": "array",
    "array": "array",
}


def tool_name_for(tool_id: str) -> str:
    """A registration-safe tool name (``app.api`` -> ``app__api``)."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", tool_id.replace(".", "__"))[:64]


def parameters_to_json_schema(parameters: Sequence[Mapping[str, Any]] | None) -> dict[str, Any]:
    """Normalize ``[{name, type, required, default, description}]`` to JSON Schema."""
    properties: dict[str, Any] = {}
    required: list[str] = []
    for parameter in parameters or []:
        name = str(parameter["name"])
        raw_type = str(parameter.get("type") or "string").strip().lower()
        base = re.split(r"[\[\s|(<]", raw_type, maxsplit=1)[0]
        spec: dict[str, Any] = {"type": _JSON_TYPES.get(base, "string")}
        if parameter.get("description"):
            spec["description"] = str(parameter["description"])
        if parameter.get("default") is not None:
            spec["default"] = parameter["default"]
        properties[name] = spec
        if parameter.get("required"):
            required.append(name)
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _aligned_matrix(path: Path, item_ids: Sequence[str], matrix_key: str, ids_key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        matrix = np.asarray(payload[matrix_key])
        ids = [str(value) for value in payload[ids_key]]
    if set(ids) != set(item_ids) or len(ids) != len(item_ids):
        raise ValueError(f"{path} item IDs do not match the tree's items")
    position = {item_id: index for index, item_id in enumerate(ids)}
    rows = [position[item_id] for item_id in item_ids]
    if matrix.shape[0] == matrix.shape[1] == len(ids):
        # Square profile: permute columns with rows so it stays self-aligned.
        return matrix[np.ix_(rows, rows)]
    return matrix[rows]


def load_catalog(manifest_path: str | Path) -> Catalog:
    """Build a pinned catalog from an ``orchard_mcp_catalog_v1`` manifest."""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CATALOG_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported catalog manifest schema_version")
    base = manifest_path.parent

    orchard = Orchard.load(base / manifest["orchard_dir"])
    cultivar_specs: Mapping[str, Mapping[str, Any]] = manifest["cultivars"]
    prefixes = default_alias_prefixes(list(cultivar_specs))
    default_cut = manifest.get("cut") or {}

    cultivars: dict[str, Cultivar] = {}
    for cultivar_id, spec in cultivar_specs.items():
        tree = orchard.trees[cultivar_id]
        labels: dict[str, str] = {}
        summaries: dict[str, str] = {}
        label_set = spec.get("label_set")
        if label_set:
            labels.update(tree.labels[label_set])
        card_set = spec.get("card_set")
        if card_set:
            for node_id, card in tree.cards[card_set].items():
                labels[node_id] = card.title
                summaries[node_id] = card.summary
        cultivars[cultivar_id] = Cultivar(
            cultivar_id=cultivar_id,
            tree=tree,
            feature_matrix=_aligned_matrix(
                base / spec["feature_matrix"],
                tree.item_ids,
                spec.get("matrix_key", "matrix"),
                spec.get("ids_key", "item_ids"),
            ),
            alias_prefix=spec.get("alias_prefix") or prefixes[cultivar_id],
            settings=CutSettings.from_mapping({**default_cut, **(spec.get("cut") or {})}),
            labels=labels,
            summaries=summaries,
        )

    items: dict[str, ToolItem] = {}
    for document in orchard.documents:
        metadata = document.metadata or {}
        items[document.item_id] = ToolItem(
            tool_id=document.item_id,
            description=document.text,
            schema={
                "name": tool_name_for(document.item_id),
                "input_schema": parameters_to_json_schema(metadata.get("parameters")),
            },
        )
    relations = load_relations(base / manifest["relations"], items) if manifest.get("relations") else {}
    return Catalog(items=items, cultivars=cultivars, name=manifest.get("name", "catalog"), relations=relations)


RELATIONS_SCHEMA_VERSION = "orchard_relations_v1"


def load_relations(path: str | Path, items: Mapping[str, Any]) -> dict[str, Any]:
    """Read an ``orchard_relations_v1`` file; keep entries and references to catalog tools only.

    The file is produced outside Orchard (e.g. by tool-tree-demo's tmatrix). Orchard
    only reads it: per tool, ``needs`` (required inputs, ``source`` "tools" with
    ``producers`` or "task") and ``feeds`` (tools its outputs can fill).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != RELATIONS_SCHEMA_VERSION:
        raise ValueError(f"unsupported relations schema_version in {path}")
    relations: dict[str, Any] = {}
    for tool_id, entry in (data.get("tools") or {}).items():
        if tool_id not in items:
            continue
        needs = []
        for need in entry.get("needs") or []:
            need = dict(need)
            if need.get("producers") is not None:
                need["producers"] = [p for p in need["producers"] if p.get("tool_id") in items]
            needs.append(need)
        feeds = [f for f in entry.get("feeds") or [] if f.get("tool_id") in items]
        relations[tool_id] = {"needs": needs, "feeds": feeds, "feeds_count": entry.get("feeds_count", len(feeds))}
    return relations


def build_fixture_catalog(item_count: int = 60, seed: int = 7) -> Catalog:
    """A small synthetic two-cultivar catalog for tests and protocol demos."""
    rng = np.random.default_rng(seed)
    topics = ["mail", "music", "pay", "files", "notes", "shop"]
    verbs = ["search", "create", "update", "delete", "show"]
    item_ids = [
        f"{topics[index % len(topics)]}.{verbs[index % len(verbs)]}_{index:02d}"
        for index in range(item_count)
    ]
    topic_centers = rng.normal(size=(len(topics), 6)) * 4
    verb_centers = rng.normal(size=(len(verbs), 6)) * 4

    def cultivar(cultivar_id: str, centers: np.ndarray, modulus: int, prefix: str) -> Cultivar:
        features = np.vstack(
            [centers[index % modulus] + rng.normal(size=6) for index in range(item_count)]
        )
        tree = Tree.from_linkage(
            scipy_linkage(features, method="average"),
            item_ids=item_ids,
            tree_id=cultivar_id,
        )
        labels = {
            node_id: f"{cultivar_id} group of {node['descendant_count']}"
            for node_id, node in tree.nodes.items()
            if node["kind"] == "internal"
        }
        summaries = {
            node_id: "Covers " + ", ".join(node["descendant_item_ids"][:3]) + "..."
            for node_id, node in tree.nodes.items()
            if node["kind"] == "internal"
        }
        return Cultivar(
            cultivar_id=cultivar_id,
            tree=tree,
            feature_matrix=features,
            alias_prefix=prefix,
            labels=labels,
            summaries=summaries,
        )

    items = {
        item_id: ToolItem(
            tool_id=item_id,
            description=f"Fixture tool that can {item_id.split('.')[1].rsplit('_', 1)[0]} "
            f"{item_id.split('.')[0]} records.",
            schema={
                "name": tool_name_for(item_id),
                "input_schema": parameters_to_json_schema(
                    [{"name": "query", "type": "str", "required": True, "description": "Input."}]
                ),
            },
        )
        for item_id in item_ids
    }
    return Catalog(
        items=items,
        cultivars={
            "domain": cultivar("domain", topic_centers, len(topics), "D"),
            "function": cultivar("function", verb_centers, len(verbs), "F"),
        },
        name="fixture",
    )
