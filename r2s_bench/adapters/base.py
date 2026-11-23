"""Dataset adapter interfaces for Real2Sim-Bench."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

from r2s_bench.schema import CaseBundleSpec


class BaseDatasetAdapter(ABC):
    """Abstract base class for converting real-robot datasets into CaseBundles."""

    @abstractmethod
    def discover_clusters(self, **kwargs: Any) -> List[Dict[str, Any]]:
        """Discover candidate single-scene multi-trajectory clusters."""

    @abstractmethod
    def build_case_bundle(
        self,
        cluster_id: str,
        out_dir: Path,
        n_public: int = 1,
        n_heldout: int = 3,
        tier: str = "tier1_rigid",
        provide_extrinsics: bool = True,
    ) -> CaseBundleSpec:
        """Download minimal assets and build a standardized CaseBundle directory."""
