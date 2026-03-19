"""qd — A standalone MAP-Elites library.

Public API re-exports:

- Archives: :class:`GridArchive`, :class:`CVTArchive`, :class:`Archive` (ABC)
- Data: :class:`Feature`, :class:`EliteEntry`
- Selection: :class:`Selector`
- Metrics: :func:`coverage`, :func:`qd_score`, :func:`best_fitness`
- Features: :func:`extract_features`, :func:`feature_names`,
  :data:`DEFAULT_FEATURES`, :data:`MODEL_FAMILY_NAMES`, :data:`DATA_STRATEGY_NAMES`
- CVT utilities: :func:`cvt`, :func:`closest_centroid_idx`
- Integration: :func:`check_qd_ready`, :func:`rebuild_archive`,
  :func:`format_archive_context`, :func:`select_parents`
"""

from qd.cvt import closest_centroid_idx, cvt
from qd.features import (
    DEFAULT_FEATURES,
    DATA_STRATEGY_NAMES,
    MODEL_FAMILY_NAMES,
    extract_features,
    feature_names,
)
from qd.integration import (
    check_qd_ready,
    format_archive_context,
    rebuild_archive,
    select_parents,
)
from qd.map_elites import Archive, CVTArchive, EliteEntry, Feature, GridArchive
from qd.metrics import best_fitness, coverage, qd_score
from qd.selection import Selector

__all__ = [
    "Archive",
    "CVTArchive",
    "DATA_STRATEGY_NAMES",
    "DEFAULT_FEATURES",
    "EliteEntry",
    "Feature",
    "GridArchive",
    "MODEL_FAMILY_NAMES",
    "Selector",
    "best_fitness",
    "check_qd_ready",
    "closest_centroid_idx",
    "coverage",
    "cvt",
    "extract_features",
    "feature_names",
    "format_archive_context",
    "qd_score",
    "rebuild_archive",
    "select_parents",
]
