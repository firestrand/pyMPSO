"""Run a config with the example paired-ring topology plugin.

Usage:
    uv run python examples/plugins/run_custom_topology_plugin_demo.py
"""

from pprint import pprint

from examples.plugins import pair_topology_plugin  # noqa: F401
from src import run_pso


def main() -> None:
    """Execute a single run using the custom topology registration."""

    results = run_pso.run_pso_from_config("examples/plugins/pair_topology_config.yaml", verbose=True)
    pprint(results)


if __name__ == "__main__":
    main()
