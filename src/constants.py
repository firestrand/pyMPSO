"""Common constants and default parameters for PSO algorithms.

This module centralizes commonly used magic numbers and default values
to improve maintainability and consistency across the codebase.
"""

import numpy as np

# ============================================================================
# Standard PSO Parameters
# ============================================================================
# Cognitive and Social Coefficients
DEFAULT_C1 = 2.05  # Cognitive coefficient (personal best attraction)
DEFAULT_C2 = 2.05  # Social coefficient (global best attraction)
STANDARD_C1 = 2.0  # Original PSO cognitive coefficient
STANDARD_C2 = 2.0  # Original PSO social coefficient
# Inertia Weight Parameters
DEFAULT_W = 0.729  # Default inertia weight for standard PSO
INERTIA_W_START = 0.9  # Starting inertia weight for linear decay
INERTIA_W_END = 0.4  # Ending inertia weight for linear decay
# ============================================================================
# SPSO 2007 Parameters
# ============================================================================
SPSO2007_C1 = 2.05
SPSO2007_C2 = 2.05
SPSO2007_CHI = 0.7298437881283576  # Constriction coefficient: 2/(2 - phi - sqrt(phi^2 - 4 * phi))
SPSO2007_PHI = SPSO2007_C1 + SPSO2007_C2  # Total acceleration = 4.1
# ============================================================================
# SPSO 2011 Parameters
# ============================================================================
SPSO2011_C1 = 0.5 + np.log(2)  # ~1.193147
SPSO2011_C2 = 0.5 + np.log(2)  # ~1.193147
SPSO2011_W = 1.0 / (2.0 * np.log(2))  # ~0.7213475
# ============================================================================
# FIPS Parameters
# ============================================================================
FIPS_PHI = 4.1  # Total acceleration coefficient for FIPS
FIPS_CHI_DEFAULT = 0.7298437881283576  # Default constriction for phi=4.1
# ============================================================================
# CLPSO Parameters
# ============================================================================
CLPSO_C = 1.49445  # Acceleration coefficient for CLPSO
CLPSO_W_START = 0.9  # Starting inertia weight
CLPSO_W_END = 0.4  # Ending inertia weight
CLPSO_REFRESH_GAP = 7  # Iterations without improvement before refreshing exemplars
CLPSO_PC_MIN = 0.05  # Minimum learning probability
CLPSO_PC_MAX = 0.5  # Maximum learning probability
# ============================================================================
# Bare Bones PSO Parameters
# ============================================================================
BBPSO_MIN_STD = 1e-10  # Minimum standard deviation for Gaussian sampling
# ============================================================================
# Velocity Clamping Parameters
# ============================================================================
DEFAULT_MAX_VELOCITY_RATIO = 0.1  # Default max velocity as proportion of bounds
DEFAULT_VELOCITY_CLAMP_EPS = 1e-10  # Small value to avoid division by zero
# ============================================================================
# Boundary Handling Parameters
# ============================================================================
DAMPING_FACTOR = 0.5  # Damping factor for reflection boundary handling
# ============================================================================
# Algorithm Defaults
# ============================================================================
DEFAULT_NUM_PARTICLES = 30  # Default swarm size
DEFAULT_MAX_ITERATIONS = 1000  # Default maximum iterations
DEFAULT_TARGET_FITNESS = 1e-8  # Default target fitness for convergence
# ============================================================================
# Topology Parameters
# ============================================================================
RING_TOPOLOGY_K = 1  # Number of neighbors on each side in ring topology
RANDOM_TOPOLOGY_K = 3  # Number of neighbors in random topology
