"""Selection operators for MAP-Elites archives.

Ported from ProgramDatabase selection methods in
science-codeevolve/src/codeevolve/database.py (Apache 2.0),
decoupled from the Program dataclass.  All methods return lists of IDs.
"""

from __future__ import annotations

import random as _random

from qd.map_elites import Archive, EliteEntry


class Selector:
    """Stateful selector over a pool of :class:`EliteEntry` items.

    Call :meth:`update` to set the pool (from an :class:`Archive` or an
    explicit list), then use the selection methods to draw IDs.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = _random.Random(seed)
        self._pool: list[EliteEntry] = []
        self._rank: dict[str, int] = {}

    # ------------------------------------------------------------------

    def update(
        self,
        archive: Archive | None = None,
        pool: list[EliteEntry] | None = None,
    ) -> None:
        """Refresh the internal pool and rank cache.

        Provide *either* an archive (elites are extracted automatically) or an
        explicit pool of :class:`EliteEntry` objects.
        """
        if archive is not None:
            self._pool = archive.elites()
        elif pool is not None:
            self._pool = list(pool)
        else:
            raise ValueError("Provide either archive or pool.")
        # Rank 0 = best (highest fitness).
        sorted_entries = sorted(self._pool, key=lambda e: e.fitness, reverse=True)
        self._rank = {e.id: i for i, e in enumerate(sorted_entries)}

    # ------------------------------------------------------------------

    def _ids(self) -> list[str]:
        return [e.id for e in self._pool]

    def _fitness_map(self) -> dict[str, float]:
        return {e.id: e.fitness for e in self._pool}

    # ------------------------------------------------------------------

    def random(self, k: int = 1) -> list[str]:
        """Uniform random selection (with replacement)."""
        ids = self._ids()
        if not ids or k <= 0:
            return []
        return self._rng.choices(ids, k=min(len(ids), k))

    def roulette(self, k: int = 1, by_rank: bool = True) -> list[str]:
        """Roulette-wheel (fitness-proportionate or rank-proportionate) selection."""
        ids = self._ids()
        if not ids or k <= 0:
            return []

        if by_rank:
            weights = [1.0 / (1 + self._rank[pid]) for pid in ids]
        else:
            fm = self._fitness_map()
            weights = [fm[pid] for pid in ids]
            total = sum(weights)
            if total <= 0:
                weights = [1.0] * len(ids)

        wsum = sum(weights)
        weights = [w / wsum for w in weights]
        return self._rng.choices(ids, weights=weights, k=min(len(ids), k))

    def tournament(self, k: int = 1, tournament_size: int = 3) -> list[str]:
        """Tournament selection: draw ``tournament_size`` candidates, keep the
        ``k`` best by rank."""
        ids = self._ids()
        if not ids or k <= 0 or tournament_size <= 0:
            return []
        candidates = self._rng.choices(ids, k=min(tournament_size, len(ids)))
        # Lower rank = better.
        best = sorted(candidates, key=lambda pid: self._rank[pid])[:k]
        return best

    def best(self, k: int = 1) -> list[str]:
        """Deterministic: return the ``k`` highest-fitness IDs."""
        ids = self._ids()
        if not ids or k <= 0:
            return []
        return sorted(ids, key=lambda pid: self._rank[pid])[:k]

    # ------------------------------------------------------------------

    def sample(self, policy: str, k: int = 1, **kwargs) -> list[str]:
        """Dispatch to a named selection policy.

        Args:
            policy: One of ``"random"``, ``"roulette"``, ``"tournament"``,
                    ``"best"``.
            k: Number of IDs to return.
            **kwargs: Forwarded to the underlying method (e.g.
                      ``by_rank``, ``tournament_size``).
        """
        methods = {
            "random": self.random,
            "roulette": self.roulette,
            "tournament": self.tournament,
            "best": self.best,
        }
        fn = methods.get(policy)
        if fn is None:
            raise ValueError(f"Unknown policy '{policy}'. Choose from {list(methods)}")
        return fn(k=k, **kwargs)
