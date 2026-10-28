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

### Test data

Some unit tests rely on the presence of test data which has been committed to
the repository under [test/data](./test/data). These data file represent
physics based inputs and expected outputs that have been translated over from
the original MUSIC code.

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

## Benchmarks

We have written benchmarks for performance relevant functions. These can be
found alongside the tests in files prefixed with `benchmark`. To compile and
run the benchmarks, use the following command:

```sh
cmake -B build
cmake --build build
./build/test/mini_pbp_benchmarks
```

You can also set a different number of samples via `--benchmark-samples` which
should, in theory, lead to a more accurate performance measurement but will
increase the execution time of the benchmarks. To understand more about
benchmarking with Catch2, read
[their documentation on authoring benchmarks](https://catch2-temp.readthedocs.io/en/latest/benchmarks.html).

### Test fixtures

Test fixtures allow logic for setting up state within tests to be defined once
and reused by multiple tests. For more information read the
[Catch2 documentation on test fixtures](https://catch2-temp.readthedocs.io/en/latest/test-fixtures.html)

Therefore, to reduce duplication and separate setup from the logic to be
benchmarked, we have defined several fixture classes. These classes are stored
in the `fixtures` dir.
