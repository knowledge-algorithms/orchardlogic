# Query test matrix

Status: **accepted at G3 (D-025), rewritten by P5B on 2026-11-09.** P5A wrote the cases against
candidate options. G3 chose the OD-6 semantics (`docs/DECISIONS.md` D-025), so each step now has
one `expected` outcome and the `option_axis` / `expected_by_option` fields are gone. Rules and
topology are unchanged. The old and new expectations are tabled in `docs/reports/P5B.md`.
The runner is `tests/query/test_query_cases.py`; it runs on fixture F7
(`fixtures/synthetic/F7_twocult8.json`, D-023), not on the inline topology below, which stays
only for comparison (the P4A cohort tests still use it). Node names `A1`.. are labels in the
`p5a_nodes` label set of F7. An `expected` may also list `ignored_rules`.

## Fixture F5 `twocult8` (inline definition, flagged for reconciliation)

P1 runs in parallel, so its F5 file was not available. The topology is defined in
`twocult8_topology.json` in this folder. **P1 reconciliation needed:** item IDs
(`i1`..`i8`), cultivar names (`A`, `B`) and node labels (`A0`..`A6`, `B0`..`B6`) are
P5A-local. When P1's F5 exists, either rename here or map names; the set structure below
is what the expectations rely on. The 8 item IDs are identical in both cultivars and
both trees are binary (7 internal nodes each). The topologies differ.

```text
Cultivar A                              Cultivar B
A0                                      B0
├─ A1 (d1)                              ├─ B1 (d1)
│   ├─ A3 (d2): i1 i2                   │   ├─ B3 (d2): i1 i5
│   └─ A4 (d2): i3 i4                   │   └─ B4 (d2): i2 i6
└─ A2 (d1)                              └─ B2 (d1)
    ├─ i5                                   ├─ i3
    └─ A5 (d2)                              └─ B5 (d2)
        ├─ i6                                   ├─ i7
        └─ A6 (d3): i7 i8                       └─ B6 (d3): i4 i8
```

Node populations:

| A | items | B | items |
|---|---|---|---|
| A1 | i1 i2 i3 i4 | B1 | i1 i2 i5 i6 |
| A2 | i5 i6 i7 i8 | B2 | i3 i4 i7 i8 |
| A3 | i1 i2 | B3 | i1 i5 |
| A4 | i3 i4 | B4 | i2 i6 |
| A5 | i6 i7 i8 | B5 | i4 i7 i8 |
| A6 | i7 i8 | B6 | i4 i8 |

Depth is the distance from the cultivar root (root = 0), as in the UI. In the candidate
model, depth is only used to order rules along one item's ancestor chain.

## Case file format

```json
{
  "id": "Q07", "title": "...", "fixture": "F5/twocult8",
  "clause": "ARCH 5 step 3 ...",
  "status": "candidate",
  "cohorts": { "c0": "all", "c1": ["i1", "i2"] },
  "steps": [
    { "query": { "version": "0.1-candidate", "base": {"cohort": "c0"}, "rules": [ ... ] },
      "result_as": "c1",
      "expected": { "items": ["i7", "i8"] } }
  ],
  "permute": true
}
```

- `expected` is `{"items": [...]}` (optionally with `ignored_rules`) or `{"error": "CODE"}`. Items are in fixture order.
- `permute: true` means the result must be identical for every ordering of `rules`.
- `result_as` names the result cohort so a later step can use it as its base.
- `{"op": "cohort_union", ...}` is P4A cohort algebra, not Execute.
- `observed_applicable: false` means the observed UI has no equivalent.

## Coverage map

| Requirement | Cases |
|---|---|
| ancestor exclude / descendant include | Q01 |
| ancestor include / descendant exclude | Q02, Q03 |
| sibling union (+ deeper exclusion) | Q04, Q05 |
| cross-cultivar intersection | Q06 |
| cross-cultivar exclusion, no depth contest | Q07, Q07b, Q07c |
| leaf include overrides cluster exclude (inside base) | Q08, Q08b, Q08c |
| leaf include outside base (ignore or error) | Q09a, Q09b |
| exclusions only | Q10a, Q10b |
| unconstrained cultivar | Q11a, Q11b, Q11c |
| order permutations | Q12a, Q12b (and every `permute` case) |
| duplicate node rules (identical and contradictory) | Q13a, Q13b1-3 |
| contradictory item rules | Q13c1-2, Q13d |
| `neutral` | Q14a, Q14b |
| successive Executes narrow | Q15 |
| no resurrection / explicit union | Q16a, Q16b, Q16c |
| explicit Boolean grouping (extension) | Q17 |

Run `node tools/ledger_probe/probe.js` to check the expectations against a reference
evaluator and to see where the observed UI logic differs.
