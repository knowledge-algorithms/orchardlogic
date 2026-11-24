"""Plain-text (markdown) rendering of Orchard responses for the model.

The MCP server returns this text as the tool result content and the full response
dict as structuredContent, so programs never parse the text. Rendering drops
fields that rarely inform navigation (session/catalog IDs, retained_pct,
restricted) and writes each shared tool description once, under the full tool IDs
that share it.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

NAVIGATION_COMMANDS = {"overview", "explore", "filter"}


def render_text(command: str, response: Mapping[str, Any]) -> str:
    """Render one command response as compact markdown text."""
    if "error" in response:
        error = response["error"]
        return f"Error ({error.get('code')}): {error.get('message')}"
    if command in ("overview", "filter") and "cultivars" in response:
        return _cohort(response)
    if command == "filter":
        return _filter_preview(response)
    if command == "explore":
        return _explore(response)
    if command == "hopper":
        return _hopper(response)
    if command == "neighborhood":
        return _neighborhood(response)
    if command == "session":
        return _session(response)
    # schemas, snapshot and anything else: exact JSON (definitions must stay exact).
    return json.dumps(response, ensure_ascii=False, separators=(",", ":"))


def tool_lines(tools: Iterable[Mapping[str, Any]]) -> list[str]:
    """One line per full tool ID; a description shared by several IDs is written once.

    Groups keep first-appearance order. A lone tool is `- id: description`; a
    shared description is a bullet followed by its tool IDs, one per line.
    """
    groups: dict[str, list[str]] = {}
    for tool in tools:
        groups.setdefault(tool.get("description", ""), []).append(tool["tool_id"])
    lines: list[str] = []
    for description, ids in groups.items():
        if len(ids) == 1:
            lines.append(f"- {ids[0]}: {description}")
        else:
            lines.append(f"- {description}")
            lines.extend(f"    {tool_id}" for tool_id in ids)
    return lines


def _group_line(group: Mapping[str, Any]) -> str:
    line = f"- {group['ref']} {group['label']} ({group['tools']})"
    if group.get("staged"):
        line += f" [staged: {group['staged']}]"
    if group.get("summary"):
        line += f": {group['summary']}"
    if group.get("scope_note"):
        line += f" ({group['scope_note']})"
    return line


def _ledger(ledger: Mapping[str, Any] | None, label: str = "Staged") -> list[str]:
    if not ledger or not ledger.get("steps"):
        return []
    return [f"{label}: {'; '.join(ledger['steps'])} -> {ledger.get('prospective_count')} tools"]


def _search_matches(response: Mapping[str, Any]) -> list[str]:
    lines = []
    for match in response.get("search_matches") or []:
        cohort_id = response.get("source_cohort_id") or response["cohort_id"]
        line = f"{match['ref']}: {match['count']} matches ({match['in_cohort']} in cohort {cohort_id})"
        # In-cohort matches by top-level group: ", " within a cultivar, " · " between.
        by_cultivar: dict[str, list[str]] = {}
        for group in match.get("groups") or []:
            by_cultivar.setdefault(group["cultivar"], []).append(
                f"{group['ref']} {group['label']} ({group['count']})")
        for parts in by_cultivar.values():
            line += " · " + ", ".join(parts)
        lines.append(line)
    return lines


def _cohort(response: Mapping[str, Any]) -> str:
    head = f"Cohort {response['cohort_id']}: {response['tool_count']} tools"
    if response.get("parent_cohort_id"):
        head += f" (filtered from {response['parent_cohort_id']}, which stays available)"
    lines = [head]
    if response.get("applied_steps"):
        lines.append("Applied: " + "; ".join(response["applied_steps"]))
    lines.extend(_search_matches(response))
    for block in response["cultivars"]:
        lines.append("")
        lines.append(f"## {block['cultivar']}")
        if block.get("note"):
            lines.append(block["note"])
        lines.extend(_group_line(group) for group in block.get("groups", []))
        if block.get("tools"):
            lines.append("Tools:")
            lines.extend(tool_lines(block["tools"]))
    if response.get("tools"):
        lines.append("")
        lines.append("Tools:")
        lines.extend(tool_lines(response["tools"]))
    lines.extend(_ledger(response.get("ledger")))
    for key in ("warning", "cut_guarantee"):
        if response.get(key):
            lines.append(str(response[key]))
    return "\n".join(lines)


def _filter_preview(response: Mapping[str, Any]) -> str:
    lines = [
        f"Cohort {response['cohort_id']}: {response.get('prospective_count')} tools "
        "would remain (no new cohort created)."
    ]
    lines.extend(_search_matches(response))
    lines.extend(_ledger(response.get("ledger")))
    lines.extend(_ledger(response.get("cleared_ledger"), "Cleared"))
    if response.get("warning"):
        lines.append(str(response["warning"]))
    return "\n".join(lines)


def _explore(response: Mapping[str, Any]) -> str:
    lines = [f"Cohort {response['cohort_id']}"]
    for entry in response["results"]:
        lines.append("")
        head = f"## {entry['ref']} {entry['label']} ({entry['tools']} tools, {entry.get('cultivar')})"
        lines.append(head)
        if entry.get("summary"):
            lines.append(entry["summary"])
        if entry.get("scope_note"):
            lines.append(entry["scope_note"])
        if entry.get("groups"):
            lines.append("Subgroups:")
            lines.extend(_group_line(group) for group in entry["groups"])
        if entry.get("tool_cards"):
            lines.append("Tools:")
            lines.extend(tool_lines(entry["tool_cards"]))
    return "\n".join(lines)


def _hopper(response: Mapping[str, Any]) -> str:
    lines = []
    if "added" in response:
        lines.append("Added: " + (", ".join(response["added"]) or "none"))
    if "removed" in response:
        lines.append("Removed: " + (", ".join(response["removed"]) or "none"))
    for item in response.get("rejected", []):
        lines.append(f"Rejected {item['tool_id']}: {item['reason']}")
    if response.get("note"):
        lines.append(response["note"])
    hopper = response.get("hopper", [])
    lines.append(f"Hopper ({len(hopper)}): " + (", ".join(hopper) or "empty"))
    if "check" in response:
        lines.extend(_check(response["check"]))
    if response.get("replayed"):
        lines.append("(replayed)")
    return "\n".join(lines)


def _producer_lines(need: Mapping[str, Any], indent: str = "    ") -> list[str]:
    lines = [f"{indent}{p['tool_id']}: {p['description']}" for p in need.get("producers", [])]
    if need.get("more"):
        lines.append(f"{indent}(+{need['more']} more)")
    return lines


def _check(check: Mapping[str, Any]) -> list[str]:
    gaps = check.get("gaps") or []
    lines = [
        f"Check: {check.get('checked_tools', 0)} collected tools, "
        f"{check.get('satisfied_inputs', 0)} required inputs covered by your tools, "
        f"{len(gaps)} missing."
    ]
    if gaps:
        lines.append("Missing (no tool you have produces it):")
        for gap in gaps:
            lines.append(f"- {', '.join(gap['tool_ids'])} need{'s' if len(gap['tool_ids']) == 1 else ''} "
                         f"{gap['param']}, from any of:")
            lines.extend(_producer_lines(gap))
    task = check.get("from_task") or {}
    if task:
        lines.append("From the task or you: " + "; ".join(
            f"{tool_id}: {', '.join(params)}" for tool_id, params in task.items()))
    return lines


def _neighborhood(response: Mapping[str, Any]) -> str:
    lines = [f"{response['tool_id']}: {response.get('description', '')}"]
    needs = response.get("needs") or []
    lines.append("")
    lines.append("## Needs (required inputs)" if needs else "## Needs: no required inputs")
    for need in needs:
        params = ", ".join(need["params"])
        if need.get("source") == "tools":
            lines.append(f"- {params} from any of:")
            lines.extend(_producer_lines(need))
        elif need.get("producers"):
            lines.append(f"- {params}: from the task or you; otherwise any of:")
            lines.extend(_producer_lines(need))
        else:
            lines.append(f"- {params}: from the task or you")
    feeds = response.get("feeds") or []
    shared = response.get("feeds_shared") or []
    if feeds or shared:
        lines.append("")
        lines.append("## Feeds (its results fill these tools' inputs)")
        for feed in feeds:
            lines.append(f"- {feed['tool_id']} ({', '.join(feed['via'])}): {feed['description']}")
        if response.get("feeds_more"):
            lines.append(f"(+{response['feeds_more']} more)")
        for group in shared:
            examples = ", ".join(e["tool_id"] for e in group["examples"])
            lines.append(f"- {', '.join(group['via'])} fills {group['count']} tools, e.g. {examples}")
    return "\n".join(lines)


def _session(response: Mapping[str, Any]) -> str:
    state = "open" if response.get("open") else "closed"
    line = f"Session {response.get('session_id')} {state}"
    if response.get("continuation_cohort_id"):
        line += f"; continue with cohort {response['continuation_cohort_id']}"
    counts = response.get("counts")
    if counts:
        line += f"; hopper {counts.get('hopper', 0)}"
    lines = [line]
    if response.get("unloaded"):
        lines.append("Unloaded: " + ", ".join(response["unloaded"]))
    if response.get("note"):
        lines.append(response["note"])
    return "\n".join(lines)
