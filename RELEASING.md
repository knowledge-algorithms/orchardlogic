# Releasing

1. Update `version` in `pyproject.toml` and `__version__` in `src/orchardlogic/__init__.py`.
2. `uv run pytest`
3. `uv build`, then `uvx twine check dist/*`
4. Commit, tag `vX.Y.Z`, push the tag.
5. Create a GitHub release for the tag. The `publish` workflow builds and uploads to
   PyPI through Trusted Publishing (no API token is stored).
6. Check: `uv run --isolated --no-project --with orchardlogic==X.Y.Z python -c "import orchardlogic"`.

One-time PyPI setup: on pypi.org, add a pending Trusted Publisher for project
`orchardlogic` with the GitHub owner and repository, workflow `publish.yml` and
environment `pypi`.
