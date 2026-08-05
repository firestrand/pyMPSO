"""Orthogonal-learning-inspired influence strategy."""

from __future__ import annotations

import numpy as np

from ..particles.base import ParticleBase
from .base import NeighborhoodInfluenceStrategy


class OrthogonalLearningInfluence(NeighborhoodInfluenceStrategy):
    """Dimension-wise influence selection inspired by orthogonal learning variants.

    For each dimension, choose the source of the social component from:
    - the current particle's personal best,
    - the best neighbor in the provided neighborhood,
    - a sampled candidate from neighbors.
    """

    def __init__(self, self_probability: float = 0.35, neighbor_probability: float = 0.45) -> None:
        self.self_probability = float(self_probability)
        self.neighbor_probability = float(neighbor_probability)

    def _neighbor_particles(self, particle: ParticleBase, neighbors: list[ParticleBase]) -> list[ParticleBase]:
        neighborhood = [particle]
        if neighbors:
            neighborhood.extend(neighbors)
        return neighborhood

    def _pick_partner(self, rng: np.random.RandomState, neighborhood: list[ParticleBase]) -> ParticleBase:
        if len(neighborhood) <= 1:
            return neighborhood[0]
        return neighborhood[int(rng.randint(0, len(neighborhood)))]

    def get_informant_position(
        self, particle: ParticleBase, neighbors: list[ParticleBase], hyperparams: dict | None = None
    ) -> np.ndarray:
        if hyperparams is None:
            hyperparams = {}

        neighborhood = self._neighbor_particles(particle, neighbors)
        if not neighborhood:
            return np.asarray(particle.pbest)

        best_neighbor = min(neighborhood, key=lambda candidate: candidate.pbest_fitness)
        rng = hyperparams.get("rng", np.random)
        self_prob = min(max(self.self_probability, 0.0), 1.0)
        neighbor_prob = min(max(self.neighbor_probability, 0.0), max(0.0, 1.0 - self_prob))

        partner = self._pick_partner(rng, neighborhood)
        best_position = np.asarray(best_neighbor.pbest, dtype=np.float64)
        self_position = np.asarray(particle.pbest, dtype=np.float64)

        informant = np.empty_like(self_position)
        rand_vals = rng.random(self_position.shape[0])
        for idx in range(self_position.size):
            if rand_vals[idx] < self_prob:
                informant[idx] = self_position[idx]
            elif rand_vals[idx] < self_prob + neighbor_prob:
                informant[idx] = best_position[idx]
            else:
                informant[idx] = partner.pbest[idx]

        return informant
