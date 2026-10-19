# Contributing

First of all, thank you for Contributing.

## Tests

Unit tests are written using [Catch2](https://catch2.org/). To build and run
the tests, use the following command:

```sh
cmake -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

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
