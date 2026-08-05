import numpy as np

from src.components.factory import get_registered_component_names
from src.influence.orthogonal_learning import OrthogonalLearningInfluence
from src.particles.standard import StandardParticle


def test_orthogonal_learning_influence_uses_expected_sources():
    rng = np.random.RandomState(1)
    particle_a = StandardParticle(initial_position=np.array([0.0, 0.0]), initial_velocity=np.zeros(2))
    particle_a.pbest = np.array([0.0, 0.0])
    particle_a.pbest_fitness = 1.0

    particle_b = StandardParticle(initial_position=np.array([1.0, 1.0]), initial_velocity=np.zeros(2))
    particle_b.pbest = np.array([1.0, 1.0])
    particle_b.pbest_fitness = 0.5

    particle_c = StandardParticle(initial_position=np.array([2.0, 2.0]), initial_velocity=np.zeros(2))
    particle_c.pbest = np.array([2.0, 2.0])
    particle_c.pbest_fitness = 2.0

    strat = OrthogonalLearningInfluence(self_probability=0.2, neighbor_probability=0.7)
    output = strat.get_informant_position(
        particle_a,
        [particle_b, particle_c],
        {"rng": rng},
    )

    source_vectors = (
        particle_a.pbest,
        particle_b.pbest,
        particle_c.pbest,
    )

    # Dimension-wise sampling means output can be a mixed coordinate-wise combination.
    assert output.shape == particle_a.pbest.shape
    for idx, value in enumerate(output):
        assert value in {source[idx] for source in source_vectors}


def test_orthogonal_learning_registered_in_component_catalog():
    names = get_registered_component_names("influence_strategy")
    assert "orthogonal_learning" in names
