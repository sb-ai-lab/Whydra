"""Generate synthetic data with structural causal models for uplift and CATE benchmarks."""

from . import scenarios
from .config import (
    PATHOLOGY_NUMBERS,
    GraphConfig,
    MechanismConfig,
    MediatorConfig,
    OutcomeConfig,
    PathologyConfig,
    ScenarioConfig,
    SegmentConfig,
    TauConfig,
    TreatmentConfig,
    pathology_number,
    pathology_phase,
)
from .diagnostics import data_properties, outcome_profile
from .generate import Dataset, generate, generate_pair, generate_split
from .graph import SampledGraph, graph_properties, sample_graph
from .io import load_dataset, load_scm, prune_versions, replay, save_dataset
from .pathologies import apply_pathologies
from .reference import pehe, segment_dim, t_learner_ridge
from .scm import SCM, sample_scm
from .segments import build_segments
from .truth import refresh_truth, truth_targets

__version__ = "0.11.0"

__all__ = [
    "PATHOLOGY_NUMBERS",
    "SCM",
    "Dataset",
    "GraphConfig",
    "MechanismConfig",
    "MediatorConfig",
    "OutcomeConfig",
    "PathologyConfig",
    "SampledGraph",
    "ScenarioConfig",
    "SegmentConfig",
    "TauConfig",
    "TreatmentConfig",
    "__version__",
    "apply_pathologies",
    "build_segments",
    "data_properties",
    "generate",
    "generate_pair",
    "generate_split",
    "graph_properties",
    "load_dataset",
    "load_scm",
    "outcome_profile",
    "pathology_number",
    "pathology_phase",
    "pehe",
    "prune_versions",
    "refresh_truth",
    "replay",
    "sample_graph",
    "sample_scm",
    "save_dataset",
    "scenarios",
    "segment_dim",
    "t_learner_ridge",
    "truth_targets",
]
