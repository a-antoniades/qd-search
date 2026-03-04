"""MAP-Elites archives: GridArchive and CVTArchive.

Ported from science-codeevolve/src/codeevolve/database.py (Apache 2.0),
decoupled from the Program dataclass.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from qd.cvt import closest_centroid_idx, cvt


@dataclass(frozen=True)
class Feature:
    """Defines one dimension of the MAP-Elites feature space.

    Attributes:
        name: Feature name (must match keys in the ``features`` dict passed to
              :meth:`Archive.add`).
        min_val: Lower bound of the feature range.
        max_val: Upper bound of the feature range.
        num_bins: Number of bins for grid discretisation.  Required for
                  :class:`GridArchive`; ignored by :class:`CVTArchive`.
    """

    name: str
    min_val: float
    max_val: float
    num_bins: int | None = None


@dataclass(frozen=True)
class EliteEntry:
    """A single elite stored in an archive cell.

    Attributes:
        id: Caller-provided identifier for the solution.
        fitness: Fitness value of the solution.
    """

    id: str
    fitness: float


class Archive(ABC):
    """Abstract base class for a MAP-Elites archive."""

    def __init__(self, features: list[Feature]) -> None:
        self.features: list[Feature] = features

    @abstractmethod
    def add(self, id: str, fitness: float, features: dict[str, float]) -> bool:
        """Try to insert a solution into the archive.

        Args:
            id: Unique identifier for the solution.
            fitness: Fitness value.
            features: Dict mapping feature name → value.

        Returns:
            ``True`` if the solution was inserted (new cell or better fitness),
            ``False`` otherwise.
        """

    @abstractmethod
    def elites(self) -> list[EliteEntry]:
        """Return all elite entries currently in the archive."""

    @property
    def elite_ids(self) -> list[str]:
        """Return the IDs of all elites."""
        return [e.id for e in self.elites()]

    @property
    def size(self) -> int:
        """Number of occupied cells."""
        return len(self.elites())

    @abstractmethod
    def cell_count(self) -> int:
        """Total number of cells (occupied + empty)."""


class GridArchive(Archive):
    """Fixed-grid MAP-Elites archive.

    Each feature dimension is discretised into ``num_bins`` bins, producing a
    regular grid.  Each cell stores at most one elite (the highest-fitness
    solution mapped to that cell).
    """

    def __init__(self, features: list[Feature]) -> None:
        super().__init__(features)
        for f in self.features:
            if f.num_bins is None:
                raise ValueError(
                    f"Feature '{f.name}' must have num_bins set for GridArchive."
                )
        self._num_cells: int = math.prod(f.num_bins for f in self.features)
        self._map: dict[tuple[int, ...], EliteEntry] = {}

    def cell_count(self) -> int:
        return self._num_cells

    # ------------------------------------------------------------------

    def _cell_index(self, features: dict[str, float]) -> tuple[int, ...] | None:
        indices: list[int] = []
        for feat in self.features:
            value = features.get(feat.name)
            if value is None:
                return None
            value = max(feat.min_val, min(value, feat.max_val))
            span = feat.max_val - feat.min_val
            proportion = (value - feat.min_val) / span if span > 0 else 0.0
            idx = int(proportion * (feat.num_bins - 1))
            indices.append(idx)
        return tuple(indices)

    def add(self, id: str, fitness: float, features: dict[str, float]) -> bool:
        cell = self._cell_index(features)
        if cell is None:
            return False
        existing = self._map.get(cell)
        if existing is None or fitness > existing.fitness:
            self._map[cell] = EliteEntry(id=id, fitness=fitness)
            return True
        return False

    def elites(self) -> list[EliteEntry]:
        return list(self._map.values())

    def occupied_cells(self) -> list[tuple[tuple[int, ...], EliteEntry]]:
        """Return all occupied cells as (cell_index, elite) pairs."""
        return list(self._map.items())

    def __repr__(self) -> str:
        return f"GridArchive(cells={self._num_cells}, occupied={self.size})"


class CVTArchive(Archive):
    """Centroidal Voronoi Tessellation MAP-Elites archive.

    The feature space is partitioned into Voronoi regions around centroids
    computed via Lloyd's algorithm.
    """

    def __init__(
        self,
        features: list[Feature],
        num_centroids: int,
        num_init_samples: int = 10_000,
        max_iter: int = 300,
        tolerance: float = 1e-6,
    ) -> None:
        super().__init__(features)
        feature_bounds = [(f.min_val, f.max_val) for f in features]
        self.centroids: np.ndarray = cvt(
            num_centroids, num_init_samples, feature_bounds, max_iter, tolerance
        )
        self._map: dict[int, EliteEntry] = {}

    def cell_count(self) -> int:
        return len(self.centroids)

    def add(self, id: str, fitness: float, features: dict[str, float]) -> bool:
        point = np.zeros(len(self.features))
        for i, feat in enumerate(self.features):
            value = features.get(feat.name)
            if value is None:
                return False
            point[i] = value

        idx = closest_centroid_idx(point, self.centroids)
        existing = self._map.get(idx)
        if existing is None or fitness > existing.fitness:
            self._map[idx] = EliteEntry(id=id, fitness=fitness)
            return True
        return False

    def elites(self) -> list[EliteEntry]:
        return list(self._map.values())

    def __repr__(self) -> str:
        return f"CVTArchive(centroids={len(self.centroids)}, occupied={self.size})"
