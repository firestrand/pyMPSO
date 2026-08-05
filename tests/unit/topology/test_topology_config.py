import numpy as np

from src.topology.config import GlobalTopologyConfig, RandomTopologyConfig


def test_global_topology_config():
    config = GlobalTopologyConfig()
    assert isinstance(config, GlobalTopologyConfig)


def test_random_topology_config_defaults():
    config = RandomTopologyConfig()
    assert config.k == 3
    assert config.rebuild_probability == 1.0
    assert isinstance(config.rng, np.random.RandomState)


def test_random_topology_config_custom():
    rng = np.random.RandomState(42)
    config = RandomTopologyConfig(k=5, rebuild_probability=0.5, rng=rng)
    assert config.k == 5
    assert config.rebuild_probability == 0.5
    assert config.rng is rng
