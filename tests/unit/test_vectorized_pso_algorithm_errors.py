import numpy as np
import pytest

from src.boundary.vectorized_position_clamping import VectorizedPositionClampingBoundaryHandler
from src.influence.vectorized_single_best import VectorizedSingleBestInfluence
from src.initialization.random_uniform import RandomUniformInitialization
from src.position.standard import StandardPositionUpdate
from src.problem.sphere import SphereFunction
from src.stopping.max_iterations import MaxIterationsStopping
from src.topology.config import GlobalTopologyConfig
from src.topology.vectorized_global import VectorizedGlobalTopology
from src.vectorized_pso_algorithm import VectorizedPSOAlgorithm
from src.velocity.config import StandardVelocityConfig
from src.velocity.vectorized_standard import VectorizedStandardVelocityUpdate


def test_vectorized_pso_algorithm_invalid_particles():
    """Verify ValueError is raised if num_particles is non-positive."""
    dimension = 2
    bounds = np.array([[-5.0, 5.0]] * dimension)
    problem = SphereFunction(dimension=dimension, bounds=bounds)
    stopping_criteria = MaxIterationsStopping(max_iterations=10)

    with pytest.raises(ValueError, match="Number of particles must be positive"):
        VectorizedPSOAlgorithm(
            problem=problem,
            stopping_criteria=stopping_criteria,
            initialization_strategy=RandomUniformInitialization(bounds=bounds, seed=42),
            topology=VectorizedGlobalTopology(),
            topology_config=GlobalTopologyConfig(),
            influence_strategy=VectorizedSingleBestInfluence(),
            velocity_strategy=VectorizedStandardVelocityUpdate(),
            velocity_config=StandardVelocityConfig(c1=1.5, c2=1.5),
            position_strategy=StandardPositionUpdate(),
            boundary_handler=VectorizedPositionClampingBoundaryHandler(),
            num_particles=0,  # Invalid
            verbose=False,
        )
