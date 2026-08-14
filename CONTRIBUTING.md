# Contributing

First of all, thank you for Contributing.

## Code formatting

We have set rules for formatting this project. These rules are enforced through
pre-commit/prek. To run these rules locally, install `prek` (ideally inside a
virtual environment) and run the rules defined in our
[.pre-commit-config.yaml](./.pre-commit-config.yaml).

```sh
uv venv
uv pip install prek
uv run prek run --all-files --color always --verbose
```
