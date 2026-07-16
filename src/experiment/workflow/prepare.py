from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Literal, Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from common.validate import NonEmptyStr
from experiment.workflow.artifacts import (
    EnvironmentManifest,
    canonical_json,
    environment_manifest,
    file_sha256,
    read_json,
)
from experiment.workflow.artifacts import (
    prepare_experiment_root as prepare_config_root,
)
from experiment.workflow.cache import (
    CACHE_GENERATOR_VERSION,
    SYNTHETIC_PROFILES,
    build_synthetic_prediction_cache,
)
from experiment.workflow.calibration import (
    CapacitySummary,
    capacity_path,
)
from experiment.workflow.config import (
    ExperimentConfig,
    LoadPercent,
    Scenario,
    TrialSpec,
)
from experiment.workflow.plan import (
    build_trial_matrix,
    poisson_arrival_offsets,
    session_permutation,
)
from experiment.workflow.sample import (
    SampleManifestEntry,
    SampleStratum,
    Sha256Digest,
    load_sample_manifest,
)
from workflow.artifacts import load_prediction_cache

Repetition = Literal[1, 2, 3, 4, 5]
GpuCount = Literal[1, 2, 3]
REPETITIONS: tuple[Repetition, ...] = (1, 2, 3, 4, 5)

PREPARED_DIR = Path("prepared")
PREPARATION_MANIFEST_PATH = PREPARED_DIR / "preparation_manifest.json"
TRIAL_MATRIX_PATH = PREPARED_DIR / "trial_matrix.jsonl"
SAMPLE_DIR = PREPARED_DIR / "samples"
CACHE_PATH = PREPARED_DIR / "synthetic_prediction_cache.json"
CACHE_GENERATION_PATH = PREPARED_DIR / "synthetic_prediction_cache_generation.json"
PERMUTATION_DIR = PREPARED_DIR / "burst_permutations"
ARRIVAL_DIR = PREPARED_DIR / "arrivals"
REUSE_DIR = PREPARED_DIR / "reuse"
SELECTION_DIR = PREPARED_DIR / "sample_selection"
PARENT_PREFLIGHT_REFERENCE_PATH = REUSE_DIR / "parent60_preflight.json"
CAPACITY_REUSE_PATH = Path("calibration/qmsum__wf-cache__g2/reuse_manifest.json")
ASSET_DIR = Path(__file__).parent / "assets"
PARENT_EXPERIMENT_ID = "system_20260713"
PARENT_EXPERIMENT_ROOT = Path("output/workflow/experiments") / PARENT_EXPERIMENT_ID
TARGET_COST_RANKS = (0, 3, 5, 8, 11, 14, 16, 19)

FORMAL_SAMPLE_IDS: dict[Scenario, tuple[str, ...]] = {
    "qmsum": (
        "test_IS1003a_specific_003",
        "test_ES2004a_specific_002",
        "test_ES2011a_specific_003",
        "test_TS3004a_general_000",
        "test_IS1003b_general_000",
        "test_IS1003c_specific_006",
        "test_TS3011d_specific_000",
        "test_TS3011b_specific_004",
        "test_ES2004b_specific_003",
        "test_TS3011c_specific_003",
        "test_ES2004c_general_001",
        "test_ES2004d_specific_000",
        "test_education_17_specific_005",
        "test_TS3004d_specific_000",
        "test_IS1003d_specific_003",
        "test_Bed008_specific_000",
        "test_education_9_specific_002",
        "test_Bmr014_specific_003",
        "test_Bmr023_specific_001",
        "test_Bro004_specific_003",
        "test_covid_4_specific_005",
        "test_Bro027_specific_000",
        "test_covid_9_specific_002",
        "test_Bmr006_general_000",
    ),
    "mbpp": (
        "437",
        "478",
        "296",
        "394",
        "459",
        "71",
        "141",
        "111",
        "434",
        "411",
        "69",
        "410",
        "421",
        "400",
        "425",
        "161",
        "292",
        "271",
        "166",
        "251",
        "98",
        "170",
        "262",
        "255",
    ),
}


class PreparationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArtifactReference(PreparationModel):
    relative_path: NonEmptyStr
    sha256: Sha256Digest


class ExternalArtifactReference(PreparationModel):
    repository_path: NonEmptyStr
    sha256: Sha256Digest


class ParentPreflightReference(PreparationModel):
    version: PositiveInt = 1
    source_experiment_id: NonEmptyStr
    source_directory: NonEmptyStr
    construction_sha256: Sha256Digest
    artifacts: dict[NonEmptyStr, Sha256Digest]
    summaries: dict[Scenario, dict[str, JsonValue]]


class SampleSelectionManifest(PreparationModel):
    version: PositiveInt = 1
    scenario: Scenario
    parent_sample_count: PositiveInt = 60
    formal_sample_count: PositiveInt = 24
    parent_manifest: ExternalArtifactReference
    cost_source: ExternalArtifactReference
    cost_metric: Literal[
        "sum_chunk_input_tokens",
        "repair_input_tokens",
    ]
    target_cost_ranks: tuple[NonNegativeInt, ...]
    selected_sample_ids: tuple[NonEmptyStr, ...] = Field(
        min_length=24,
        max_length=24,
    )
    stratum_counts: dict[SampleStratum, PositiveInt]
    formal_manifest: ArtifactReference


class CapacityReuseManifest(PreparationModel):
    version: PositiveInt = 1
    source_experiment_id: NonEmptyStr
    parent_session_count: PositiveInt = 60
    source_capacity: ExternalArtifactReference
    reused_capacity: ArtifactReference


class SyntheticProfileManifest(PreparationModel):
    model_name: NonEmptyStr
    sequence_lengths: tuple[PositiveInt, ...]
    output_lengths: tuple[PositiveInt, ...]
    load_sec: PositiveFloat
    base_vram_mb: PositiveFloat
    power_watts: PositiveFloat
    seconds_per_input_token: PositiveFloat
    seconds_per_output_token: PositiveFloat


class SyntheticCacheGenerationManifest(PreparationModel):
    version: PositiveInt = 1
    generator: NonEmptyStr
    generator_version: PositiveInt
    profiles: tuple[SyntheticProfileManifest, ...]
    cache: ArtifactReference


class BurstPermutation(PreparationModel):
    version: PositiveInt = 1
    scenario: Scenario
    repetition: Repetition
    seed: int
    sample_manifest_sha256: Sha256Digest
    sample_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)


class PreparationManifest(PreparationModel):
    version: PositiveInt = 1
    experiment_id: NonEmptyStr
    trial_count: PositiveInt
    trial_matrix: ArtifactReference
    sample_manifests: dict[Scenario, ArtifactReference]
    sample_selection_manifests: dict[Scenario, ArtifactReference]
    synthetic_cache: ArtifactReference
    synthetic_cache_generation: ArtifactReference
    burst_permutations: dict[Scenario, tuple[ArtifactReference, ...]]
    parent_preflight: ArtifactReference
    capacity_reuse: ArtifactReference
    environment: EnvironmentManifest


class ArrivalPoint(PreparationModel):
    position: NonNegativeInt
    sample_id: NonEmptyStr
    offset_sec: NonNegativeFloat


class ArrivalTrace(PreparationModel):
    version: PositiveInt = 1
    trace_key: NonEmptyStr
    scenario: Scenario
    gpu_count: GpuCount
    load_percent: LoadPercent
    repetition: Repetition
    absolute_sessions_per_sec: PositiveFloat
    arrivals: tuple[ArrivalPoint, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_arrivals(self) -> Self:
        positions = [arrival.position for arrival in self.arrivals]
        if positions != list(range(len(self.arrivals))):
            raise ValueError("arrival positions must be contiguous and ordered")
        offsets = [arrival.offset_sec for arrival in self.arrivals]
        if offsets[0] != 0:
            raise ValueError("the first arrival offset must be zero")
        if any(left >= right for left, right in pairwise(offsets)):
            raise ValueError("arrival offsets must be strictly increasing")
        return self


class ArrivalTraceManifest(PreparationModel):
    version: PositiveInt = 1
    trace_key: NonEmptyStr
    scenario: Scenario
    gpu_count: GpuCount
    load_percent: LoadPercent
    repetition: Repetition
    capacity_sessions_per_sec: PositiveFloat
    absolute_sessions_per_sec: PositiveFloat
    permutation_sha256: Sha256Digest
    trace: ArtifactReference


@dataclass(frozen=True, slots=True)
class PreparedExperiment:
    root: Path
    manifest: PreparationManifest
    trial_matrix: tuple[TrialSpec, ...]

    @property
    def trial_matrix_path(self) -> Path:
        return self._path(self.manifest.trial_matrix)

    @property
    def trial_matrix_sha256(self) -> str:
        return self.manifest.trial_matrix.sha256

    @property
    def sample_manifest_paths(self) -> dict[Scenario, Path]:
        return {
            scenario: self._path(reference)
            for scenario, reference in self.manifest.sample_manifests.items()
        }

    @property
    def sample_manifest_sha256(self) -> dict[Scenario, str]:
        return {
            scenario: reference.sha256
            for scenario, reference in self.manifest.sample_manifests.items()
        }

    @property
    def environment(self) -> EnvironmentManifest:
        return self.manifest.environment

    @property
    def parent_preflight_path(self) -> Path:
        return self._path(self.manifest.parent_preflight)

    @property
    def capacity_reuse_path(self) -> Path:
        return self._path(self.manifest.capacity_reuse)

    @property
    def synthetic_cache_path(self) -> Path:
        return self._path(self.manifest.synthetic_cache)

    @property
    def synthetic_cache_sha256(self) -> str:
        return self.manifest.synthetic_cache.sha256

    @property
    def cache_generation_manifest_path(self) -> Path:
        return self._path(self.manifest.synthetic_cache_generation)

    @property
    def cache_generation_manifest_sha256(self) -> str:
        return self.manifest.synthetic_cache_generation.sha256

    def sample_manifest_path(self, scenario: Scenario) -> Path:
        return self._path(self.manifest.sample_manifests[scenario])

    def burst_permutation_path(
        self,
        scenario: Scenario,
        repetition: int,
    ) -> Path:
        return self._path(self._permutation_reference(scenario, repetition))

    def burst_permutation(
        self,
        scenario: Scenario,
        repetition: int,
    ) -> BurstPermutation:
        return BurstPermutation.model_validate(
            read_json(self.burst_permutation_path(scenario, repetition))
        )

    def burst_permutation_sha256(
        self,
        scenario: Scenario,
        repetition: int,
    ) -> str:
        return self._permutation_reference(scenario, repetition).sha256

    def _permutation_reference(
        self,
        scenario: Scenario,
        repetition: int,
    ) -> ArtifactReference:
        if repetition not in range(1, 6):
            raise ValueError("repetition must be between one and five")
        return self.manifest.burst_permutations[scenario][repetition - 1]

    def _path(self, reference: ArtifactReference) -> Path:
        return _resolve_relative(self.root, reference.relative_path)


@dataclass(frozen=True, slots=True)
class ArrivalTraceArtifact:
    path: Path
    manifest_path: Path
    sha256: str
    manifest_sha256: str
    trace: ArrivalTrace


def prepare_experiment(config: ExperimentConfig) -> PreparedExperiment:
    prepare_config_root(config)
    root = _output_root(config)
    (root / "trials").mkdir(exist_ok=True)
    trials = build_trial_matrix()
    trial_matrix_sha = _write_jsonl_or_match(
        root,
        root / TRIAL_MATRIX_PATH,
        trials,
    )

    sample_entries: dict[Scenario, tuple[SampleManifestEntry, ...]] = {}
    sample_references: dict[Scenario, ArtifactReference] = {}
    selection_references: dict[Scenario, ArtifactReference] = {}
    for scenario in ("qmsum", "mbpp"):
        source = _sample_asset_path(scenario)
        parent_entries = load_sample_manifest(source)
        entries = _select_formal_samples(scenario, parent_entries)
        if len(entries) != config.session_count:
            raise ValueError(f"{scenario} sample manifest must contain 24 entries")
        target = root / _sample_relative_path(scenario)
        digest = _write_sample_manifest_or_match(root, target, entries)
        sample_entries[scenario] = entries
        sample_references[scenario] = _reference(root, target, digest)
        selection = _sample_selection_manifest(
            scenario,
            source,
            sample_references[scenario],
            entries,
        )
        selection_path = root / _selection_relative_path(scenario)
        selection_sha = _write_json_or_match(root, selection_path, selection)
        selection_references[scenario] = _reference(
            root,
            selection_path,
            selection_sha,
        )

    cache = build_synthetic_prediction_cache()
    cache_sha = _write_json_or_match(root, root / CACHE_PATH, cache)
    cache_reference = _reference(root, root / CACHE_PATH, cache_sha)
    generation_manifest = _cache_generation_manifest(cache_reference)
    generation_sha = _write_json_or_match(
        root,
        root / CACHE_GENERATION_PATH,
        generation_manifest,
    )

    permutation_references: dict[Scenario, tuple[ArtifactReference, ...]] = {}
    for scenario in ("qmsum", "mbpp"):
        references: list[ArtifactReference] = []
        sample_ids = tuple(entry.sample_id for entry in sample_entries[scenario])
        for repetition in REPETITIONS:
            permutation = BurstPermutation(
                scenario=scenario,
                repetition=repetition,
                seed=config.seed,
                sample_manifest_sha256=sample_references[scenario].sha256,
                sample_ids=session_permutation(
                    sample_ids,
                    seed=config.seed,
                    repetition=repetition,
                ),
            )
            path = root / _permutation_relative_path(scenario, repetition)
            digest = _write_json_or_match(root, path, permutation)
            references.append(_reference(root, path, digest))
        permutation_references[scenario] = tuple(references)

    preflight_path = root / PARENT_PREFLIGHT_REFERENCE_PATH
    preflight_sha = _write_json_or_match(
        root,
        preflight_path,
        _parent_preflight_reference(),
    )
    capacity_reuse_path = root / CAPACITY_REUSE_PATH
    capacity_reference = _prepare_reused_capacity(config, root)
    capacity_reuse_sha = _write_json_or_match(
        root,
        capacity_reuse_path,
        _capacity_reuse_manifest(capacity_reference),
    )

    manifest = PreparationManifest(
        experiment_id=config.experiment_id,
        trial_count=len(trials),
        trial_matrix=_reference(root, root / TRIAL_MATRIX_PATH, trial_matrix_sha),
        sample_manifests=sample_references,
        sample_selection_manifests=selection_references,
        synthetic_cache=cache_reference,
        synthetic_cache_generation=_reference(
            root,
            root / CACHE_GENERATION_PATH,
            generation_sha,
        ),
        burst_permutations=permutation_references,
        parent_preflight=_reference(root, preflight_path, preflight_sha),
        capacity_reuse=_reference(
            root,
            capacity_reuse_path,
            capacity_reuse_sha,
        ),
        environment=_frozen_environment(root),
    )
    _write_json_or_match(root, root / PREPARATION_MANIFEST_PATH, manifest)
    prepared = validate_prepared_experiment(config)
    _prepare_formal_arrivals(config, capacity_reference)
    return prepared


def prepare_experiment_root(config: ExperimentConfig) -> PreparedExperiment:
    return prepare_experiment(config)


def load_prepared_experiment(config: ExperimentConfig) -> PreparedExperiment:
    return validate_prepared_experiment(config)


def validate_prepared_experiment(config: ExperimentConfig) -> PreparedExperiment:
    root = _output_root(config)
    _validate_config_manifest(config, root)
    saved_manifest = PreparationManifest.model_validate(
        read_json(root / PREPARATION_MANIFEST_PATH)
    )
    trials = _load_trial_matrix(root / TRIAL_MATRIX_PATH)
    expected_trials = build_trial_matrix()
    if trials != expected_trials:
        raise ValueError("prepared trial matrix does not match the formal matrix")
    trial_reference = _reference_for_existing(root, root / TRIAL_MATRIX_PATH)

    sample_references: dict[Scenario, ArtifactReference] = {}
    selection_references: dict[Scenario, ArtifactReference] = {}
    sample_entries: dict[Scenario, tuple[SampleManifestEntry, ...]] = {}
    for scenario in ("qmsum", "mbpp"):
        source = _sample_asset_path(scenario)
        target = root / _sample_relative_path(scenario)
        entries = load_sample_manifest(target)
        expected_entries = _select_formal_samples(
            scenario,
            load_sample_manifest(source),
        )
        if entries != expected_entries:
            raise ValueError(f"prepared {scenario} sample manifest does not match")
        if len(entries) != config.session_count:
            raise ValueError(f"{scenario} sample manifest must contain 24 entries")
        sample_entries[scenario] = entries
        sample_references[scenario] = _reference_for_existing(root, target)
        selection_path = root / _selection_relative_path(scenario)
        selection = SampleSelectionManifest.model_validate(read_json(selection_path))
        expected_selection = _sample_selection_manifest(
            scenario,
            source,
            sample_references[scenario],
            entries,
        )
        if selection != expected_selection:
            raise ValueError(f"prepared {scenario} sample selection does not match")
        selection_references[scenario] = _reference_for_existing(root, selection_path)

    cache_path = root / CACHE_PATH
    cache = load_prediction_cache(cache_path)
    if cache != build_synthetic_prediction_cache():
        raise ValueError("prepared synthetic prediction cache does not match")
    cache_reference = _reference_for_existing(root, cache_path)
    generation_path = root / CACHE_GENERATION_PATH
    generation = SyntheticCacheGenerationManifest.model_validate(
        read_json(generation_path)
    )
    if generation != _cache_generation_manifest(cache_reference):
        raise ValueError("synthetic cache generation manifest does not match")

    permutation_references: dict[Scenario, tuple[ArtifactReference, ...]] = {}
    for scenario in ("qmsum", "mbpp"):
        references: list[ArtifactReference] = []
        sample_ids = tuple(entry.sample_id for entry in sample_entries[scenario])
        for repetition in REPETITIONS:
            path = root / _permutation_relative_path(scenario, repetition)
            permutation = BurstPermutation.model_validate(read_json(path))
            expected = BurstPermutation(
                scenario=scenario,
                repetition=repetition,
                seed=config.seed,
                sample_manifest_sha256=sample_references[scenario].sha256,
                sample_ids=session_permutation(
                    sample_ids,
                    seed=config.seed,
                    repetition=repetition,
                ),
            )
            if permutation != expected:
                raise ValueError(f"prepared burst permutation does not match: {path}")
            references.append(_reference_for_existing(root, path))
        permutation_references[scenario] = tuple(references)

    preflight_path = root / PARENT_PREFLIGHT_REFERENCE_PATH
    preflight = ParentPreflightReference.model_validate(read_json(preflight_path))
    if preflight != _parent_preflight_reference():
        raise ValueError("parent preflight reference does not match")
    capacity_file = capacity_path(config, "qmsum", 2)
    capacity = CapacitySummary.model_validate(read_json(capacity_file))
    expected_capacity = CapacitySummary.model_validate_json(
        _capacity_asset_path().read_text(encoding="utf-8")
    )
    if capacity != expected_capacity:
        raise ValueError("reused QMSum capacity does not match its frozen asset")
    capacity_reference = _reference_for_existing(root, capacity_file)
    capacity_reuse_path = root / CAPACITY_REUSE_PATH
    capacity_reuse = CapacityReuseManifest.model_validate(
        read_json(capacity_reuse_path)
    )
    if capacity_reuse != _capacity_reuse_manifest(capacity_reference):
        raise ValueError("QMSum capacity reuse manifest does not match")

    expected_manifest = PreparationManifest(
        experiment_id=config.experiment_id,
        trial_count=len(trials),
        trial_matrix=trial_reference,
        sample_manifests=sample_references,
        sample_selection_manifests=selection_references,
        synthetic_cache=cache_reference,
        synthetic_cache_generation=_reference_for_existing(root, generation_path),
        burst_permutations=permutation_references,
        parent_preflight=_reference_for_existing(root, preflight_path),
        capacity_reuse=_reference_for_existing(root, capacity_reuse_path),
        environment=saved_manifest.environment,
    )
    if saved_manifest != expected_manifest:
        raise ValueError("preparation manifest does not match prepared artifacts")
    return PreparedExperiment(
        root=root,
        manifest=saved_manifest,
        trial_matrix=trials,
    )


def load_or_create_arrival_trace(
    config: ExperimentConfig,
    spec: TrialSpec,
    capacity_sessions_per_sec: float,
) -> ArrivalTraceArtifact:
    if capacity_sessions_per_sec <= 0:
        raise ValueError("capacity_sessions_per_sec must be positive")
    prepared = validate_prepared_experiment(config)
    _validate_arrival_spec(spec)
    assert spec.load_percent is not None
    trace_key = arrival_trace_key(spec)
    permutation = prepared.burst_permutation(spec.scenario, spec.repetition)
    absolute_rate = capacity_sessions_per_sec * spec.load_percent / 100
    offsets = poisson_arrival_offsets(
        len(permutation.sample_ids),
        sessions_per_sec=absolute_rate,
        seed=config.seed,
        trace_key=trace_key,
    )
    trace = ArrivalTrace(
        trace_key=trace_key,
        scenario=spec.scenario,
        gpu_count=spec.gpu_count,
        load_percent=spec.load_percent,
        repetition=spec.repetition,
        absolute_sessions_per_sec=absolute_rate,
        arrivals=tuple(
            ArrivalPoint(position=index, sample_id=sample_id, offset_sec=offset)
            for index, (sample_id, offset) in enumerate(
                zip(permutation.sample_ids, offsets, strict=True)
            )
        ),
    )
    trace_path, manifest_path = _arrival_paths(prepared.root, trace_key)
    trace_sha = _write_json_or_match(prepared.root, trace_path, trace)
    trace_reference = _reference(prepared.root, trace_path, trace_sha)
    manifest = ArrivalTraceManifest(
        trace_key=trace_key,
        scenario=spec.scenario,
        gpu_count=spec.gpu_count,
        load_percent=spec.load_percent,
        repetition=spec.repetition,
        capacity_sessions_per_sec=capacity_sessions_per_sec,
        absolute_sessions_per_sec=absolute_rate,
        permutation_sha256=prepared.burst_permutation_sha256(
            spec.scenario,
            spec.repetition,
        ),
        trace=trace_reference,
    )
    _write_json_or_match(prepared.root, manifest_path, manifest)
    return load_arrival_trace(config, spec)


def load_arrival_trace(
    config: ExperimentConfig,
    spec: TrialSpec,
) -> ArrivalTraceArtifact:
    prepared = validate_prepared_experiment(config)
    _validate_arrival_spec(spec)
    assert spec.load_percent is not None
    trace_key = arrival_trace_key(spec)
    trace_path, manifest_path = _arrival_paths(prepared.root, trace_key)
    manifest = ArrivalTraceManifest.model_validate(read_json(manifest_path))
    trace = ArrivalTrace.model_validate(read_json(trace_path))
    expected_identity = (
        spec.scenario,
        spec.gpu_count,
        spec.load_percent,
        spec.repetition,
    )
    if (
        manifest.scenario,
        manifest.gpu_count,
        manifest.load_percent,
        manifest.repetition,
    ) != expected_identity:
        raise ValueError("arrival manifest identity does not match the trial")
    if (
        trace.scenario,
        trace.gpu_count,
        trace.load_percent,
        trace.repetition,
    ) != expected_identity:
        raise ValueError("arrival trace identity does not match the trial")
    if manifest.trace_key != trace_key or trace.trace_key != trace_key:
        raise ValueError("arrival trace key does not match the trial")
    trace_sha = file_sha256(trace_path)
    expected_reference = _reference(prepared.root, trace_path, trace_sha)
    if manifest.trace != expected_reference:
        raise ValueError("arrival trace hash does not match its manifest")
    if manifest.absolute_sessions_per_sec != trace.absolute_sessions_per_sec:
        raise ValueError("arrival absolute rate does not match its manifest")
    expected_rate = manifest.capacity_sessions_per_sec * spec.load_percent / 100
    if expected_rate != trace.absolute_sessions_per_sec:
        raise ValueError("arrival absolute rate does not match capacity")
    permutation = prepared.burst_permutation(spec.scenario, spec.repetition)
    if manifest.permutation_sha256 != prepared.burst_permutation_sha256(
        spec.scenario,
        spec.repetition,
    ):
        raise ValueError("arrival permutation hash does not match")
    if tuple(point.sample_id for point in trace.arrivals) != permutation.sample_ids:
        raise ValueError("arrival samples do not match the burst permutation")
    return ArrivalTraceArtifact(
        path=trace_path,
        manifest_path=manifest_path,
        sha256=trace_sha,
        manifest_sha256=file_sha256(manifest_path),
        trace=trace,
    )


def arrival_trace_key(spec: TrialSpec) -> str:
    _validate_arrival_spec(spec)
    assert spec.load_percent is not None
    return (
        f"{spec.scenario}__g{spec.gpu_count}__load{spec.load_percent:03d}__"
        f"rep{spec.repetition}"
    )


def _validate_arrival_spec(spec: TrialSpec) -> None:
    if spec.workload != "open-loop" or spec.load_percent is None:
        raise ValueError("arrival traces require an open-loop trial")
    formal_ids = {trial.trial_id for trial in build_trial_matrix()}
    if spec.trial_id not in formal_ids:
        raise ValueError("arrival trace trial is not in the formal matrix")


def _load_trial_matrix(path: Path) -> tuple[TrialSpec, ...]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError("prepared trial matrix must be non-empty JSONL")
    return tuple(TrialSpec.model_validate_json(line) for line in lines)


def _cache_generation_manifest(
    cache_reference: ArtifactReference,
) -> SyntheticCacheGenerationManifest:
    return SyntheticCacheGenerationManifest(
        generator="experiment.workflow.cache.build_synthetic_prediction_cache",
        generator_version=CACHE_GENERATOR_VERSION,
        profiles=tuple(
            SyntheticProfileManifest.model_validate(asdict(profile))
            for profile in SYNTHETIC_PROFILES
        ),
        cache=cache_reference,
    )


def _validate_config_manifest(config: ExperimentConfig, root: Path) -> None:
    expected = {
        "version": 1,
        "experiment_id": config.experiment_id,
        "config": config.model_dump(mode="json"),
    }
    if read_json(root / "experiment_manifest.json") != expected:
        raise ValueError("experiment manifest does not match the requested config")


def _select_formal_samples(
    scenario: Scenario,
    parent_entries: tuple[SampleManifestEntry, ...],
) -> tuple[SampleManifestEntry, ...]:
    if len(parent_entries) != 60:
        raise ValueError(f"{scenario} parent manifest must contain 60 entries")
    by_id = {entry.sample_id: entry for entry in parent_entries}
    selected = tuple(
        by_id[sample_id].model_copy(update={"position": position})
        for position, sample_id in enumerate(FORMAL_SAMPLE_IDS[scenario])
    )
    stratum_counts = {
        stratum: sum(entry.stratum == stratum for entry in selected)
        for stratum in ("short", "medium", "long")
    }
    if stratum_counts != {"short": 8, "medium": 8, "long": 8}:
        raise ValueError(f"{scenario} formal sample strata are invalid")
    if scenario == "qmsum":
        general_count = sum(
            entry.metadata.get("query_type") == "general" for entry in selected
        )
        meetings = {
            stratum: {
                entry.sample_id.rsplit("_", maxsplit=2)[0]
                for entry in selected
                if entry.stratum == stratum
            }
            for stratum in ("short", "medium", "long")
        }
        if general_count != 4 or any(len(values) != 8 for values in meetings.values()):
            raise ValueError("QMSum formal sample diversity is invalid")
    elif "255" not in {entry.sample_id for entry in selected}:
        raise ValueError("MBPP formal samples must retain the 1600-token case")
    return selected


def _sample_selection_manifest(
    scenario: Scenario,
    parent_path: Path,
    formal_reference: ArtifactReference,
    entries: tuple[SampleManifestEntry, ...],
) -> SampleSelectionManifest:
    cost_names = {
        "qmsum": ("qmsum_rows.jsonl", "sum_chunk_input_tokens"),
        "mbpp": ("mbpp_rows.jsonl", "repair_input_tokens"),
    }
    cost_name, raw_cost_metric = cost_names[scenario]
    cost_metric = cast(
        Literal["sum_chunk_input_tokens", "repair_input_tokens"],
        raw_cost_metric,
    )
    cost_hashes = {
        "qmsum_rows.jsonl": (
            "2d5e9bb42a44f5e020c071c05ea081f9f4bfaa453ebf9741a10c6795dd4cfcce"
        ),
        "mbpp_rows.jsonl": (
            "e5d5e24b5a74642c03ab0df4475b4332895d97a6067a8f7599f3fc779fdcee4a"
        ),
    }
    return SampleSelectionManifest(
        scenario=scenario,
        parent_manifest=ExternalArtifactReference(
            repository_path=(
                Path("src/experiment/workflow/assets") / parent_path.name
            ).as_posix(),
            sha256=file_sha256(parent_path),
        ),
        cost_source=ExternalArtifactReference(
            repository_path=(
                PARENT_EXPERIMENT_ROOT / "prepared/preflight" / cost_name
            ).as_posix(),
            sha256=cost_hashes[cost_name],
        ),
        cost_metric=cost_metric,
        target_cost_ranks=TARGET_COST_RANKS,
        selected_sample_ids=tuple(entry.sample_id for entry in entries),
        stratum_counts={"short": 8, "medium": 8, "long": 8},
        formal_manifest=formal_reference,
    )


def _parent_preflight_reference() -> ParentPreflightReference:
    return ParentPreflightReference(
        source_experiment_id=PARENT_EXPERIMENT_ID,
        source_directory=(PARENT_EXPERIMENT_ROOT / "prepared/preflight").as_posix(),
        construction_sha256=(
            "1b37fb440d4b2050205533590367e3a821324814b9c28529cfa86abf1305e189"
        ),
        artifacts={
            "construction_manifest.json": (
                "bbd964efa8bb2ef15c7a5ead2d061c264117d3bf758aebfca20e844997419fd8"
            ),
            "qmsum_rows.jsonl": (
                "2d5e9bb42a44f5e020c071c05ea081f9f4bfaa453ebf9741a10c6795dd4cfcce"
            ),
            "qmsum_summary.json": (
                "92ae0134e1c820016b6ac9b692891f0647fa1bee606b8de22ee9fad9eaa20cb2"
            ),
            "mbpp_rows.jsonl": (
                "e5d5e24b5a74642c03ab0df4475b4332895d97a6067a8f7599f3fc779fdcee4a"
            ),
            "mbpp_summary.json": (
                "baf088d53f40e369436f358b0621bbb986fd154f6869a72456f2c2bace13bc41"
            ),
        },
        summaries={
            "qmsum": {
                "sample_count": 60,
                "prompt_count": 420,
                "max_input_tokens": 6578,
                "max_total_tokens": 6962,
                "overflow_count": 0,
            },
            "mbpp": {
                "sample_count": 60,
                "prompt_count": 180,
                "max_input_tokens": 1600,
                "max_total_tokens": 2112,
                "overflow_count": 0,
            },
        },
    )


def _prepare_reused_capacity(
    config: ExperimentConfig,
    root: Path,
) -> ArtifactReference:
    asset_path = _capacity_asset_path()
    capacity = CapacitySummary.model_validate_json(
        asset_path.read_text(encoding="utf-8")
    )
    if (
        capacity.calibration_key != "qmsum__wf-cache__g2"
        or capacity.capacity_sessions_per_sec != 0.07748262032531168
    ):
        raise ValueError("frozen QMSum capacity asset is invalid")
    target = capacity_path(config, "qmsum", 2)
    digest = _write_bytes_or_match(root, target, asset_path.read_bytes())
    return _reference(root, target, digest)


def _capacity_reuse_manifest(
    capacity_reference: ArtifactReference,
) -> CapacityReuseManifest:
    return CapacityReuseManifest(
        source_experiment_id=PARENT_EXPERIMENT_ID,
        source_capacity=ExternalArtifactReference(
            repository_path=(
                PARENT_EXPERIMENT_ROOT
                / "calibration/qmsum__wf-cache__g2/capacity.json"
            ).as_posix(),
            sha256=(
                "90873c291b6f8c4f4f5b68c77b1fd7e6751d0c923b421d707f5c10edddd21069"
            ),
        ),
        reused_capacity=capacity_reference,
    )


def _frozen_environment(root: Path) -> EnvironmentManifest:
    path = root / PREPARATION_MANIFEST_PATH
    if path.is_file():
        return PreparationManifest.model_validate(read_json(path)).environment
    return environment_manifest()


def _prepare_formal_arrivals(
    config: ExperimentConfig,
    capacity_reference: ArtifactReference,
) -> None:
    capacity_path = _resolve_relative(
        _output_root(config),
        capacity_reference.relative_path,
    )
    capacity = CapacitySummary.model_validate(
        read_json(capacity_path)
    )
    seen: set[str] = set()
    for trial in build_trial_matrix():
        if trial.workload != "open-loop":
            continue
        key = arrival_trace_key(trial)
        if key in seen:
            continue
        seen.add(key)
        load_or_create_arrival_trace(
            config,
            trial,
            capacity.capacity_sessions_per_sec,
        )
    if len(seen) != 10:
        raise RuntimeError(
            f"formal preparation must contain 10 arrivals, got {len(seen)}"
        )


def _capacity_asset_path() -> Path:
    return ASSET_DIR / "qmsum_parent60_capacity.json"


def _sample_asset_path(scenario: Scenario) -> Path:
    names = {
        "qmsum": "qmsum_test_60_seed42.jsonl",
        "mbpp": "mbpp_sanitized_test_60_seed42.jsonl",
    }
    return ASSET_DIR / names[scenario]


def _sample_relative_path(scenario: Scenario) -> Path:
    return SAMPLE_DIR / f"{scenario}.jsonl"


def _selection_relative_path(scenario: Scenario) -> Path:
    return SELECTION_DIR / f"{scenario}.json"


def _permutation_relative_path(scenario: Scenario, repetition: int) -> Path:
    return PERMUTATION_DIR / f"{scenario}__rep{repetition}.json"


def _arrival_paths(root: Path, trace_key: str) -> tuple[Path, Path]:
    trace_path = root / ARRIVAL_DIR / f"{trace_key}.json"
    manifest_path = root / ARRIVAL_DIR / f"{trace_key}.manifest.json"
    _ensure_within_root(root, trace_path)
    _ensure_within_root(root, manifest_path)
    return trace_path, manifest_path


def _output_root(config: ExperimentConfig) -> Path:
    return config.output_root.expanduser().resolve()


def _reference_for_existing(root: Path, path: Path) -> ArtifactReference:
    if not path.is_file():
        raise FileNotFoundError(path)
    return _reference(root, path, file_sha256(path))


def _reference(root: Path, path: Path, digest: str) -> ArtifactReference:
    resolved = _ensure_within_root(root, path)
    return ArtifactReference(
        relative_path=resolved.relative_to(root.resolve()).as_posix(),
        sha256=digest,
    )


def _resolve_relative(root: Path, relative_path: str) -> Path:
    if Path(relative_path).is_absolute():
        raise ValueError("prepared artifact path must be relative")
    return _ensure_within_root(root, root / relative_path)


def _ensure_within_root(root: Path, path: Path) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"artifact path escapes experiment output root: {path}")
    return resolved


def _write_json_or_match(root: Path, path: Path, value: object) -> str:
    return _write_bytes_or_match(
        root,
        path,
        f"{canonical_json(value)}\n".encode(),
    )


def _write_jsonl_or_match(
    root: Path,
    path: Path,
    rows: tuple[TrialSpec, ...],
) -> str:
    payload = "".join(f"{canonical_json(row)}\n" for row in rows).encode()
    return _write_bytes_or_match(root, path, payload)


def _write_sample_manifest_or_match(
    root: Path,
    path: Path,
    rows: tuple[SampleManifestEntry, ...],
) -> str:
    payload = "".join(f"{canonical_json(row)}\n" for row in rows).encode()
    return _write_bytes_or_match(root, path, payload)


def _write_bytes_or_match(root: Path, path: Path, payload: bytes) -> str:
    path = _ensure_within_root(root, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"prepared artifact content mismatch: {path}")
    else:
        with path.open("xb") as file:
            file.write(payload)
            file.flush()
    return file_sha256(path)
