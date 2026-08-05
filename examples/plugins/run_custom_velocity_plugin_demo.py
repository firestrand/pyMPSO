"""Run a config with the example chaos velocity plugin.

Usage:
    uv run python examples/plugins/run_custom_velocity_plugin_demo.py
"""

from pprint import pprint

from examples.plugins import chaos_velocity_plugin  # noqa: F401
from src import run_pso


def main() -> None:
    """Execute a single run using the custom plugin registration."""
    results = run_pso.run_pso_from_config("examples/plugins/chaos_velocity_config.yaml", verbose=True)
    pprint(results)


if __name__ == "__main__":
    main()
