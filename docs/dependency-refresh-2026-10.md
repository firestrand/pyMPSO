# Dependency refresh — 2026-10-04

Scope: refresh the existing Python package and its tooling, preserve public behavior,
and add focused tests until repository line and branch coverage each exceed 80%.
Reviewed against `Python Code Standards.md`, revision 2026-09-29, from the shared
Software-Standards library. No application source or existing assertions changed.

## Dependency audit

The baseline lock had 31 distinct known advisories across nine packages: Click,
idna, msgpack, Pillow, Pygments, pytest, Ray, Requests, and urllib3. The scanner
returned 57 rows because some advisories have multiple identifiers.
The refreshed lock's 61 third-party package pins, including optional Ray, developer
tools, and platform-specific dependencies, pass pip-audit with zero known advisories.
Setuptools 84.0.0, wheel 0.48.0, and packaging 26.3 also pass a separate build-tool audit.

All resolved packages are at the latest stable release available on the audit date,
except docutils 0.22.4: [Sphinx 9.1.0's metadata](https://pypi.org/pypi/Sphinx/9.1.0/json)
requires docutils <0.23. Updating it independently would violate the documentation
builder's supported dependency range.

## Lockfile upgrades

Classification compares the previous committed lock with the refreshed lock,
rather than the old permissive manifest floors. Exact artifacts and hashes remain
in `uv.lock`.

| Package | Before | After | Release change |
| --- | --- | --- | --- |
| annotated-doc | 0.0.4 | 0.0.5 | patch |
| attrs | 25.4.0 | 26.1.0 | calendar version |
| certifi | 2026.1.4 | 2026.7.22 | calendar version |
| charset-normalizer | 3.4.4 | 3.5.2 | minor |
| click | 8.3.1 | 8.5.0 | minor |
| contourpy | 1.3.3 | 1.4.0 | minor |
| coverage | 7.13.4 | 7.16.2 | minor |
| filelock | 3.24.3 | 4.0.10 | major |
| fonttools | 4.61.1 | 4.66.1 | minor |
| idna | 3.11 | 3.20 | minor |
| imagesize | 1.4.1 | 2.0.1 | major |
| kiwisolver | 1.4.9 | 1.5.1 | minor |
| markdown-it-py | 4.0.0 | 4.2.0 | minor |
| markupsafe | 3.0.3 | 3.0.4 | patch |
| matplotlib | 3.10.8 | 3.11.2 | minor |
| msgpack | 1.1.2 | 1.2.3 | minor |
| numpy | 2.4.2 | 2.5.3 | minor |
| packaging | 26.0 | 26.3 | minor |
| pillow | 12.1.1 | 12.3.0 | minor |
| protobuf | 6.33.5 | 7.36.2 | major |
| pygments | 2.19.2 | 2.21.0 | minor |
| pyparsing | 3.3.2 | 3.3.3 | patch |
| pytest | 9.0.2 | 9.1.1 | minor |
| pytest-cov | 7.0.0 | 7.1.0 | minor |
| pytest-mock | 3.15.1 | 3.16.0 | minor |
| ray | 2.54.0 | 2.59.0 | minor |
| requests | 2.32.5 | 2.34.2 | minor |
| rich | 14.3.2 | 15.0.0 | major |
| rpds-py | 0.30.0 | 2026.9.1 | calendar version |
| ruff | 0.15.1 | 0.16.10 | minor |
| snowballstemmer | 3.0.1 | 3.1.1 | minor |
| ty | 0.0.17 | 0.0.84 | patch |
| typer | 0.24.0 | 0.27.2 | minor |
| typing-extensions | 4.15.0 | 4.16.0 | minor |
| urllib3 | 2.6.3 | 2.8.0 | minor |

## Major upgrades and compatibility

- [Rich 15](https://github.com/Textualize/rich/releases/tag/v15.0.0) removes Python 3.8
  support, below this project's Python 3.12 floor. CLI rendering is covered by tests.
- [Protobuf 7](https://protobuf.dev/support/migration/#changes-in-python) removes
  deprecated descriptor APIs and tightens value validation. The application has no
  direct Protobuf API calls; actual Ray workers were exercised with the updated
  transitive dependency.
- [filelock 4](https://github.com/tox-dev/filelock/releases/tag/4.0.0),
  [imagesize 2](https://github.com/shibukawa/imagesize_py/releases/tag/2.0.1), and
  [rpds-py's calendar version](https://github.com/crate-py/rpds/releases/tag/v2026.5.1)
  are transitive dependencies with no direct application API usage.
- [attrs 26.1](https://www.attrs.org/en/stable/changelog.html) uses calendar
  versioning and changes field-transformer alias initialization. The application
  does not use attrs decorators or field transformers directly.
- Setuptools 84 removes legacy `pkg_resources`, which this package does not use.
  The package now declares its existing MIT license as an SPDX string, removing the
  [deprecated TOML license table](https://setuptools.pypa.io/en/stable/userguide/license_migration.html).
  Wheel and source distribution builds complete without deprecation warnings.

The development interpreter is pinned to [Python 3.14.8](https://www.python.org/downloads/release/python-3148/),
and uv was updated from 0.12.22 to 0.12.23. `requires-python >=3.12` and Ruff/ty's
Python 3.12 compatibility settings remain aligned. Python 3.12.15 and 3.13.16 were
installed for compatibility verification; Python 3.15 prereleases were excluded.

## Verification

The unit-only run passes 476 tests with warnings treated as errors. Coverage includes
all production modules under `src` (3,463 statements and 940 branches); the existing
19 coverage exclusions were not changed. Unit-only coverage is 92.84% of lines
(3,215/3,463) and 86.49% of branches (813/940).

Commands used for the local gate:

```bash
uv lock --check
uv sync --locked --all-extras
uv run --locked --all-extras ruff check src tests
uv run --locked --all-extras ruff format --check src tests
uv run --locked --all-extras ty check src tests
uv run --locked --all-extras pytest -W error
uv run --locked --all-extras pytest tests/unit -W error --cov=src --cov-branch --cov-report=json
uv build --out-dir /tmp/pympso-final-dist
```

Ruff 0.16 also formats Markdown code fences; its check required one focused
formatting change in `src/clamping/README.md`. No lint/type rules were disabled,
no suppressions were added, and all original tests and assertions remain intact.

The built wheel was installed into a clean environment with only locked runtime
dependencies. Its CLI dry-run and optimization execution passed outside the
checkout, including persisted JSON results and expected iteration/evaluation counts.
Real Ray 2.59.0 workers passed deterministic benchmark runs, automatically owned
cluster cleanup, and reuse of an existing cluster without shutting it down.

The repository contains no checked-in GitHub Actions workflow. Verification was
performed locally on Linux ARM64; conditional Windows dependency pins were audited,
but Windows execution was not tested.

| Applicable standards group | Result | Evidence |
| --- | --- | --- |
| Runtime and dependency declarations | PASS | Python floor, development pin, refreshed manifest and lock |
| Dependency advisories | PASS | Every current locked third-party pin and build tools audited |
| Scope and public behavior | PASS | Application source and original assertions unchanged |
| Quality tools and package delivery | PASS | Lint, formatting, types, wheel/sdist and installed CLI pass |
| Focused unit coverage | PASS | Independent numerical, validation, state, plugin and failure tests; both coverage targets exceeded |

| Coverage run | Lines | Branches |
| --- | --- | --- |
| Baseline full suite | 68.78% | 46.81% |
| Final unit suite | 92.84% | 86.49% |
| Final full suite | 94.95% | 89.26% |

Python 3.14.8: 600 tests passed in 178.65 seconds.
Python 3.13.16: 600 tests passed in 178.69 seconds.
Python 3.12.15: 600 tests passed in 181.99 seconds.
All runs use `-W error`; no test flakes or deprecation warnings were observed.
