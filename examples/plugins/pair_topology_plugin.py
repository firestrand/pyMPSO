"""Example topology plugin for research prototypes.

This plugin creates a deterministic local neighborhood where each particle is connected
to its immediate predecessor and successor (cyclic), with configurable optional halo.
"""

from __future__ import annotations

from src.components.factory import register_component
from src.particles.base import ParticleBase
from src.topology.base import NeighborhoodTopology


@register_component("topology", "paired_ring")
class PairedRingTopology(NeighborhoodTopology):
    """Neighborhood topology that links each particle to nearby indices."""

    def __init__(self, halo: int = 1):
        self.halo = max(1, int(halo))

    def get_neighbors(
        self, particles: list[ParticleBase], rebuild: bool | None = None
    ) -> dict[ParticleBase, list[ParticleBase]]:
        """Return cyclic local neighbors for each particle.

        Each particle is connected to ``halo`` predecessors and ``halo`` successors
        (excluding itself), with wrap-around.
        """

        _ = rebuild
        swarm_size = len(particles)
        if swarm_size <= 1:
            return {particle: [] for particle in particles}

        neighbors: dict[ParticleBase, list[ParticleBase]] = {}
        for index, particle in enumerate(particles):
            neighbor_indices: list[int] = []
            for offset in range(1, self.halo + 1):
                neighbor_indices.append((index - offset) % swarm_size)
                neighbor_indices.append((index + offset) % swarm_size)
            # Remove duplicates and maintain deterministic order.
            seen: set[int] = set()
            ordered: list[int] = []
            for idx in neighbor_indices:
                if idx in seen:
                    continue
                seen.add(idx)
                ordered.append(idx)
            neighbors[particle] = [particles[nidx] for nidx in ordered]

        return neighbors
