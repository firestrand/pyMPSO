"""Behavioral contracts for the scalar velocity strategies retained by the library."""

import numpy as np
import pytest

from src.clamping.velocity import MaxNormVelocityClampingStrategy
from src.particles.standard import StandardParticle
from src.velocity.bare_bones import BareBonesPSOVelocityUpdate
from src.velocity.clpso import CLPSOVelocityUpdate
from src.velocity.constriction import ConstrictionCoefficientVelocityUpdate
from src.velocity.de_hybrid import DEHybridVelocityUpdate
from src.velocity.fips import FIPSVelocityUpdate
from src.velocity.hypersphere import HypersphereVelocity
from src.velocity.inertial_weight import InertiaWeightVelocityUpdate
from src.velocity.orthogonal_mutation import OrthogonalMutationVelocityUpdate


@pytest.fixture
def particle() -> StandardParticle:
    """A particle with distinct position, velocity, and personal best."""
    result = StandardParticle(np.array([1.0, -2.0]), np.array([4.0, -6.0]))
    result.pbest = np.array([3.0, 2.0])
    result.pbest_fitness = 10.0
    return result


@pytest.mark.parametrize("use_abs_std", [True, False])
def test_bare_bones_gaussian_displacement(particle, use_abs_std):
    strategy = BareBonesPSOVelocityUpdate(use_abs_std=use_abs_std)
    result = strategy.update(particle, np.array([5.0, -2.0]), {"rng": np.random.RandomState(7)})
    if use_abs_std:
        # Seed 7 standard normal draws are [1.6905257, -0.46593737].
        np.testing.assert_allclose(result, [6.3810514076, 0.1362505178])
    else:
        # Negative signed deviations use the exploration floor; positive ones retain their span.
        np.testing.assert_allclose(result, [3.0, 0.1362505178], atol=1e-9)
    np.testing.assert_array_equal(particle.position, [1.0, -2.0])
    np.testing.assert_array_equal(particle.velocity, [4.0, -6.0])


def test_bare_bones_minimum_exploration_and_clamping(particle):
    strategy = BareBonesPSOVelocityUpdate(MaxNormVelocityClampingStrategy(0.25), min_std=-1)
    result = strategy.update(
        particle, particle.pbest, {"rng": np.random.RandomState(4), "bounds": np.array([[-5, 5], [-5, 5]])}
    )
    np.testing.assert_array_equal(result, [0.25, 0.25])
    assert strategy.min_std == 1e-10
    assert strategy.get_required_hyperparams() == []
    assert strategy.get_optional_hyperparams() == {"bounds": None}
    assert "use_abs_std=True" in str(strategy)
    assert "MaxNormVelocityClampingStrategy" in repr(strategy)


@pytest.mark.parametrize("clamp", [False, True])
def test_inertia_weight_isolates_previous_velocity(particle, clamp):
    strategy = InertiaWeightVelocityUpdate(MaxNormVelocityClampingStrategy(1.0))
    result = strategy.update(
        particle,
        np.array([100.0, 100.0]),
        {"w": 0.5, "c1": 0.0, "c2": 0.0, "bounds": np.ones((2, 2)) if clamp else None},
    )
    np.testing.assert_array_equal(result, [1.0, -1.0] if clamp else [2.0, -3.0])


@pytest.mark.parametrize(
    "phi1, phi2, expected", [(2.05, 2.05, 0.7298437881), (1.0, 1.0, 0.7298437881), (3.0, 3.0, 0.2679491924)]
)
def test_constriction_coefficient_convergence_fallback(phi1, phi2, expected):
    assert ConstrictionCoefficientVelocityUpdate._calculate_constriction_coefficient(phi1, phi2) == pytest.approx(
        expected
    )


@pytest.mark.parametrize("clamp", [False, True])
def test_constriction_explicit_coefficient(particle, clamp):
    strategy = ConstrictionCoefficientVelocityUpdate(MaxNormVelocityClampingStrategy(0.5))
    result = strategy.update(
        particle, particle.position, {"chi": 0.25, "phi1": 0, "phi2": 0, "bounds": np.ones((2, 2)) if clamp else None}
    )
    np.testing.assert_array_equal(result, [0.5, -0.5] if clamp else [1.0, -1.5])


@pytest.mark.parametrize(
    "neighbors, message", [(None, "requires 'neighbors'"), ([], "non-empty list"), ((), "non-empty list")]
)
def test_fips_requires_neighbors(particle, neighbors, message):
    with pytest.raises(ValueError, match=message):
        FIPSVelocityUpdate().update(particle, particle.position, {"neighbors": neighbors})


@pytest.mark.parametrize("use_personal_best", [True, False])
def test_fips_uses_selected_neighbor_positions(particle, use_personal_best):
    neighbor = StandardParticle(np.array([5.0, 2.0]), np.zeros(2))
    neighbor.pbest = np.array([9.0, 6.0])
    strategy = FIPSVelocityUpdate(use_personal_best=use_personal_best)
    result = strategy.update(
        particle, np.array([999.0, 999.0]), {"neighbors": [neighbor], "phi": 2.0, "rng": np.random.RandomState(0)}
    )
    # Seed 0 uniform draws are [0.5488135039, 0.7151893664].
    expected = [9.3281451794, 3.9725615333] if use_personal_best else [6.1237601660, -0.2032505977]
    np.testing.assert_allclose(result, expected, atol=2e-6)


def test_fips_fallback_phi_and_real_clamping(particle):
    strategy = FIPSVelocityUpdate(MaxNormVelocityClampingStrategy(0.1), phi=2.0)
    result = strategy.update(
        particle, particle.position, {"neighbors": [particle], "phi": 0, "bounds": np.ones((2, 2))}
    )
    np.testing.assert_array_equal(result, [0.1, -0.1])
    assert strategy.phi == 4.1
    assert strategy.get_required_hyperparams() == ["neighbors"]
    assert strategy.get_optional_hyperparams() == {"bounds": None, "phi": 4.1}
    assert "phi=4.10" in str(strategy)
    assert "MaxNormVelocityClampingStrategy" in repr(strategy)


@pytest.mark.parametrize(
    "swarm_size, index, expected", [(1, 0, 0.5), (5, 0, 0.5), (5, 2, 0.275), (5, 4, 0.05), (5, 99, 0.05), (5, -1, 0.5)]
)
def test_clpso_learning_probability_rank_limits(swarm_size, index, expected):
    assert CLPSOVelocityUpdate()._calculate_learning_probability(index, swarm_size) == pytest.approx(expected)


def test_clpso_stagnation_refresh_and_improvement(particle):
    strategy = CLPSOVelocityUpdate(refresh_gap=2)
    assert strategy._should_refresh_exemplars(particle)
    assert not strategy._should_refresh_exemplars(particle)
    particle.pbest_fitness = 9.0
    assert not strategy._should_refresh_exemplars(particle)
    assert not strategy._should_refresh_exemplars(particle)
    assert strategy._should_refresh_exemplars(particle)
    strategy.reset()
    assert strategy._should_refresh_exemplars(particle)


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_clpso_tournament_selects_fitter_neighbor(particle, seed):
    fitter = StandardParticle(np.array([8.0, 8.0]), np.zeros(2))
    fitter.pbest_fitness = 0.0
    strategy = CLPSOVelocityUpdate()
    assignments = strategy._generate_exemplar_assignments(
        particle, [particle, fitter], 1.0, np.random.RandomState(seed)
    )
    np.testing.assert_array_equal(assignments, [1, 1])
    own = strategy._generate_exemplar_assignments(particle, [particle, fitter], 0.0, np.random.RandomState(seed))
    np.testing.assert_array_equal(own, [0, 0])


@pytest.mark.parametrize("iteration, expected", [(0, [3.6, -5.4]), (10, [1.6, -2.4])])
def test_clpso_single_particle_adaptive_inertia(particle, iteration, expected):
    strategy = CLPSOVelocityUpdate(c=0.0)
    result = strategy.update(
        particle,
        particle.position,
        {
            "all_particles": [particle],
            "current_iteration": iteration,
            "max_iterations": 10,
            "rng": np.random.RandomState(0),
        },
    )
    np.testing.assert_allclose(result, expected)
    assert strategy.get_required_hyperparams() == ["all_particles"]
    assert strategy.get_optional_hyperparams()["max_iterations"] == 1000
    assert "gap=7" in str(strategy)
    assert "NoClampingStrategy" in repr(strategy)


def test_clpso_requires_swarm_and_clamps(particle):
    strategy = CLPSOVelocityUpdate(MaxNormVelocityClampingStrategy(0.2), c=0)
    with pytest.raises(ValueError, match="requires 'all_particles'"):
        strategy.update(particle, particle.position, {})
    result = strategy.update(particle, particle.position, {"all_particles": [particle], "bounds": np.ones((2, 2))})
    np.testing.assert_array_equal(result, [0.2, -0.2])


@pytest.mark.parametrize("swarm_size, index", [(0, 0), (3, 0), (4, -1)])
def test_de_hybrid_skips_injection_without_sufficient_swarm(particle, swarm_size, index):
    strategy = DEHybridVelocityUpdate()
    result = strategy.update(
        particle,
        particle.position,
        {"c1": 0, "c2": 0, "all_particles": [particle] * swarm_size, "particle_index": index},
    )
    np.testing.assert_array_equal(result, particle.velocity)


@pytest.mark.parametrize("probability, changed_dimensions", [(0.0, 1), (1.0, 2)])
def test_de_hybrid_forces_at_least_one_crossover_dimension(particle, probability, changed_dimensions):
    neighbors = [StandardParticle(np.array([9.0, 6.0]), np.zeros(2)) for _ in range(3)]
    particle.pbest = particle.position.copy()
    strategy = DEHybridVelocityUpdate(crossover_probability=probability)
    result = strategy.update(
        particle,
        particle.position,
        {
            "c1": 0,
            "c2": 0,
            "all_particles": [particle, *neighbors],
            "particle_index": 0,
            "rng": np.random.RandomState(3),
        },
    )
    difference = result - particle.velocity
    assert np.count_nonzero(difference) == changed_dimensions
    assert set(difference) <= {0.0, 4.0}


def test_de_hybrid_indices_are_distinct_and_clamping_applies(particle):
    strategy = DEHybridVelocityUpdate(MaxNormVelocityClampingStrategy(0.3))
    selected = strategy._random_distinct_indices(np.random.RandomState(5), 4, 2)
    assert set(selected) == {0, 1, 3}
    result = strategy.update(particle, particle.position, {"c1": 0, "c2": 0, "bounds": np.ones((2, 2))})
    np.testing.assert_array_equal(result, [0.3, -0.3])


@pytest.mark.parametrize("axes", [0, 1])
def test_orthogonal_mutation_degenerate_dimensions(axes):
    particle = StandardParticle(np.array([1.0]), np.array([2.0]))
    result = OrthogonalMutationVelocityUpdate()._sample_orthogonal_mutation(
        particle, None, axes, 1.0, np.random.RandomState(0)
    )
    np.testing.assert_array_equal(result, [0.0])


def test_orthogonal_mutation_scale_and_zero_span(particle):
    strategy = OrthogonalMutationVelocityUpdate()
    baseline = strategy._sample_orthogonal_mutation(particle, None, 2, 1.0, np.random.RandomState(8))
    scaled = strategy._sample_orthogonal_mutation(
        particle, np.array([[-2.0, 2.0], [3.0, 3.0]]), 2, 0.5, np.random.RandomState(8)
    )
    np.testing.assert_allclose(scaled, baseline * [2.0, 0.5])
    empty = strategy._sample_orthogonal_mutation(particle, np.empty((0, 2)), 2, 1.0, np.random.RandomState(8))
    np.testing.assert_array_equal(empty, baseline)


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_orthogonal_mutation_gate_preserves_base_velocity(particle, probability):
    strategy = OrthogonalMutationVelocityUpdate(mutation_probability=probability, mutation_strength=0.0)
    result = strategy.update(particle, particle.position, {"c1": 0, "c2": 0, "rng": np.random.RandomState(2)})
    np.testing.assert_array_equal(result, particle.velocity)


def test_orthogonal_mutation_real_clamping(particle):
    strategy = OrthogonalMutationVelocityUpdate(
        MaxNormVelocityClampingStrategy(0.4), mutation_probability=1.0, mutation_axes=99
    )
    result = strategy.update(
        particle,
        particle.position,
        {"c1": 0, "c2": 0, "mutation_strength": 0, "bounds": np.ones((2, 2)), "rng": np.random.RandomState(1)},
    )
    np.testing.assert_array_equal(result, [0.4, -0.4])


@pytest.mark.parametrize("shared_best, expected", [(True, [3.0, 2.0]), (False, [5.0, 2.0])])
def test_hypersphere_gravity_center(particle, shared_best, expected):
    informant = particle.pbest if shared_best else np.array([5.0, 0.0])
    result = HypersphereVelocity()._calculate_gravity_center(particle.position, particle.pbest, informant, 2.0, 2.0)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("distrib", [0, -1])
def test_hypersphere_sample_stays_inside_radius(distrib):
    center = np.array([3.0, -4.0])
    strategy = HypersphereVelocity()
    point = strategy._sample_from_hypersphere(center, 2.0, np.random.RandomState(0), distrib)
    distance = np.linalg.norm(point - center)
    assert 0.0 < distance < 2.0
    assert distance == pytest.approx(1.2055267521 if distrib == 0 else 1.5527567428)


@pytest.mark.parametrize("clamp", [False, True])
def test_hypersphere_stationary_gravity_center_uses_inertia(particle, clamp):
    particle.pbest = particle.position.copy()
    strategy = HypersphereVelocity(MaxNormVelocityClampingStrategy(0.5))
    result = strategy.update(particle, particle.position, {"w": 0.25, "bounds": np.ones((2, 2)) if clamp else None})
    np.testing.assert_array_equal(result, [0.5, -0.5] if clamp else [1.0, -1.5])
