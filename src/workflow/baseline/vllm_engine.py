from __future__ import annotations

import ray

from workflow.replica import GenerationEngine


class StaticVLLMEngine(GenerationEngine):
    engine_label = "static vLLM engine"
    # No scheduler admission control sits in front of the baseline path, so
    # it relies on vLLM's own batching rather than a max_num_seqs cutoff.
    enforce_capacity = False


StaticVLLMEngineActor = ray.remote(num_gpus=1)(StaticVLLMEngine)
