# Contributing

Bug reports and pull requests are welcome at
https://github.com/knowledge-algorithms/orchardlogic/issues.

## Setup

```bash
uv sync
uv run pytest
```

The default test run is offline. Tests marked `live_labeling` call the OpenAI API and
are excluded by default. Tests marked `optional_model` need the neural extras.

## Pull requests

- Keep changes focused, and add or update a test for any behaviour change.
- Do not change artifact or schema version strings (`orchard_cultivar_v1`,
  `treedb.query/1`, the DuckDB `treedb.*` schema) without a version bump and a note
  in the changelog section of the release.
- Contributions are accepted under the Apache License 2.0 (see LICENSE, section 5).
  Only submit code you have the right to license that way, and say where any
  third-party code came from.
