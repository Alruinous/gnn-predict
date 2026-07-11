# Workflow Runtime Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved workflow runtime specification in `docs/workflow/codex_plan_20260708.md` as a generic, testable Ray dataflow and prediction-aware scheduling system.

**Architecture:** Keep scheduling decisions in a deterministic Python core and expose them through one async Ray scheduler actor. Node workers own bounded data-plane queues and static fan-in; dedicated actors own model replicas, terminal results, and trace files. Runtime scheduling consumes only offline prediction and deployment-profile artifacts.

**Tech Stack:** Python 3.12, Pydantic 2, Ray 2.55, Ray Queue, LangChain Core, Transformers, PyTorch, pytest, Ruff, ty.

## Global Constraints

- The runtime supports only static, non-empty DAGs with exactly one entry and one terminal node.
- Runtime node kinds are `agent` and registered `function`; `input`, `output`, `tool`, and `evaluator` are rejected.
- One session executes each static node at most once, and one source contributes at most one item to a target per session.
- One bounded input queue belongs to each node and is shared by all incoming edges.
- Each worker executes at most one in-flight node task; no runtime batching or continuous batching is added.
- Each phase 1 agent task performs one granted model inference and returns a LangChain-compatible `AgentState`.
- One live or loading replica is allowed per `(model_key, gpu_kind)` and each current replica owns exactly one accelerator.
- Runtime scheduling never exports ONNX, builds PyG data, runs a GNN checkpoint, polls NVML, or mutates offline prediction artifacts.
- Missing prediction buckets, invalid DAG state, resource-ledger violations, and required trace failures fail explicitly.
- Node retry exhaustion fails one session; unrelated sessions continue.
- Existing `WorkflowModelFeatureKey` field names and digest semantics remain compatible with cache manifests and pickle filenames.
- Baseline code remains isolated and is not changed in this implementation.
- New Python modules use future annotations, explicit signatures, narrow exceptions, and one-line rationale comments only.

---

**Execution order note:** Tasks run as 1, 2, 3, 5, 4, 11, 10, 9, 6, 7, 8 so each scheduling layer has a focused review gate.

### Task 1: Workflow schema and immutable DAG contract

**Files:**
- Create: `src/workflow/schema.py`
- Modify: `src/workflow/types.py`
- Test: `tests/test_workflow_schema.py`

**Interfaces:**
- Produces: `Workflow`, `AgentNodeConfig`, `FunctionNodeConfig`, `WorkflowGraph`, `WorkflowDataItem`, runtime state enums, and the compatible `WorkflowModelFeatureKey`.
- Consumes: `NonEmptyStr`, `NonNegativeInt`, and `PositiveInt` from `src/common/validate.py`.

- [x] **Step 1: Write schema tests before production changes**

```python
def test_workflow_computes_one_static_graph() -> None:
    workflow = Workflow.model_validate(WORKFLOW_PAYLOAD)
    assert workflow.graph.entry_node == "split"
    assert workflow.graph.terminal_node == "merge"
    assert workflow.graph.topological_order == ("split", "left", "right", "merge")


def test_workflow_rejects_cycle() -> None:
    with pytest.raises(ValidationError, match="acyclic"):
        Workflow.model_validate(CYCLIC_PAYLOAD)
```

- [x] **Step 2: Run the focused tests and confirm the missing-schema failure**

Run: `PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_workflow_schema.py -q`

Expected: collection fails because `workflow.schema` does not exist.

- [x] **Step 3: Implement discriminated node configs and DAG validation**

```python
class AgentNodeConfig(NodeConfigBase):
    type: Literal["agent"]
    model: WorkflowModelConfig
    execution: ExecutionConfig
    token_budget: TokenBudgetConfig
    prompt_template: NonEmptyStr


class FunctionNodeConfig(NodeConfigBase):
    type: Literal["function"]
    function: NonEmptyStr
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    routing: Literal["broadcast", "targeted"] = "broadcast"
```

The `Workflow` validator rejects empty graphs, unknown endpoints, self-edges, duplicate edges, cycles, and any graph without exactly one entry and one terminal. It computes adjacency, dependencies, topological order, entry, and terminal once in `WorkflowGraph`.

- [x] **Step 4: Add data-plane and lifecycle types without changing the cache key**

```python
class WorkflowDataItem(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: NonEmptyStr
    item_id: NonEmptyStr = Field(default_factory=lambda: str(uuid4()))
    source_node: NonEmptyStr | None
    target_node: NonEmptyStr
    message: AgentState
    session_input_ref: Any
```

- [x] **Step 5: Run schema tests and the existing cache conversion tests**

Run: `PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_workflow_schema.py tests/test_arch_configs.py -q`

Expected: all selected tests pass.

### Task 2: Offline prediction, deployment profile, and deterministic token policy

**Files:**
- Create: `src/workflow/artifacts.py`
- Create: `src/workflow/policy.py`
- Test: `tests/test_workflow_artifacts.py`
- Test: `tests/test_workflow_policy.py`

**Interfaces:**
- Consumes: `WorkflowModelFeatureKey`, `AgentNodeConfig`, and `AcceleratorConfig`.
- Produces: `PredictionCache`, `DeploymentProfile`, `SchedulerConfig`, `select_token_budget()`, and `select_eviction_victim()`.

- [x] **Step 1: Write artifact and policy tests**

```python
def test_token_policy_chooses_largest_feasible_cached_budget() -> None:
    decision = select_token_budget(
        node=agent_node(min_tokens=128, default_tokens=512, max_tokens=1024),
        input_tokens=1100,
        accelerators=[accelerator("v100", 16_000), accelerator("a100", 40_000)],
        predictions=prediction_cache(),
        history={},
        oom_penalties={},
        eps_mem_mb=512,
    )
    assert decision.gpu_kind == "v100"
    assert decision.granted_max_new_tokens == 512
    assert decision.action == TokenBudgetAction.FIXED
```

- [x] **Step 2: Run tests and confirm missing-module failures**

Run: `PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_workflow_artifacts.py tests/test_workflow_policy.py -q`

Expected: collection fails because the artifact and policy modules do not exist.

- [x] **Step 3: Implement strict YAML loaders and indexed prediction lookup**

```python
class PredictionEntry(BaseModel):
    key: WorkflowModelFeatureKey
    predicted_run_sec: PositiveFloat
    predicted_peak_vram_mb: PositiveFloat
    predicted_power_watts: NonNegativeFloat | None = None
    predictor_metadata: dict[str, JsonValue] = Field(default_factory=dict)


def load_prediction_cache(path: str | Path) -> PredictionCache:
    with Path(path).open() as stream:
        return PredictionCache.model_validate(yaml.safe_load(stream))
```

Duplicate keys and duplicate deployment profile keys are validation errors. Runtime lookups use `phase="decode"`, `batch_size=1`, and the existing cache key.

- [x] **Step 4: Implement token feasibility and eviction as pure functions**

The token policy selects the smallest cached input bucket covering exact prompt tokens, searches cached output buckets inside the configured range, applies memory margin and in-run OOM penalty, and returns a deterministic placement. The eviction policy ranks only `IDLE` and `SUSPECT` replicas by future reuse distance minus reload cost, then falls back to longest idle.

- [x] **Step 5: Run artifact and policy tests**

Run: `PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_workflow_artifacts.py tests/test_workflow_policy.py -q`

Expected: all selected tests pass.

### Task 3: Result, trace, and summary ownership

**Files:**
- Modify: `src/workflow/types.py`
- Create: `src/workflow/storage.py`
- Test: `tests/test_workflow_storage.py`

**Interfaces:**
- Consumes: `AgentState`.
- Produces: `TraceEvent`, `ResultStore`, `ResultStoreActor`, `TraceWriter`, `TraceWriterActor`, and `write_run_summary()`.

- [x] **Step 1: Write persistence tests**

```python
def test_result_store_persists_each_session_once(tmp_path: Path) -> None:
    store = ResultStore(tmp_path, "run-1")
    state = AgentState(messages=[AIMessage(content="answer")])
    store.put("session-1", "merge", state)
    with pytest.raises(ValueError, match="already has a result"):
        store.put("session-1", "merge", state)
```

- [x] **Step 2: Run the focused test and confirm the missing-module failure**

Run: `PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_workflow_storage.py -q`

Expected: collection fails because `workflow.storage` does not exist.

- [x] **Step 3: Implement narrow LangChain serialization and ordered JSONL writers**

```python
serialized_state = dumpd(state)
assert not contains_not_implemented(serialized_state), serialized_state
row = {
    "run_id": self.run_id,
    "session_id": session_id,
    "node_id": node_id,
    "serialization_format": "langchain_dump",
    "langchain_core_version": langchain_core.__version__,
    "state": serialized_state,
}
```

Use ordinary `json.dumps` and file `flush`; do not add string coercion or pickle fallback.

- [x] **Step 4: Derive `run_summary.json` from trace and result JSONL**

The summary includes session counts, per-node task counts, token actions, OOM count, load/reuse/eviction counts, and active/resident GPU seconds.

- [x] **Step 5: Run storage tests**

Run: `PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_workflow_storage.py -q`

Expected: all selected tests pass.

### Task 4: Session and node-task lifecycle core

**Files:**
- Create: `src/workflow/scheduler.py`
- Test: `tests/test_workflow_scheduler.py`

**Interfaces:**
- Consumes: `Workflow.graph` and runtime report contracts.
- Produces: `SchedulerCore`, session/task records, begin/acquire/cancel/function-complete/finish/fail transitions, cancellation actions, and drain state.

- [x] **Step 1: Write state-transition and invariant tests**

```python
def test_function_task_completes_only_after_output_emission() -> None:
    core = scheduler_core()
    task_id = core.begin_node("s1", "function", ["item-1"])
    decision = core.complete(task_id, successful_function_report())
    assert decision.emit_output is True
    assert core.tasks[task_id].state == NodeTaskState.EMITTING
    core.finish_node(task_id, output_report())
    assert core.tasks[task_id].state == NodeTaskState.COMPLETED
```

- [x] **Step 2: Run the focused tests and confirm the missing-module failure**

Run: `PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_workflow_scheduler.py -q`

Expected: collection fails because `workflow.scheduler` does not exist.

- [x] **Step 3: Implement strict records, reports, and lifecycle transitions**

The core validates session registration, one task per `(session, node)`, static completed dependencies, function `RUNNING -> EMITTING -> COMPLETED`, agent `ACQUIRING`, pending acquire creation/cancellation, and terminal session completion.

- [x] **Step 4: Cover session failure isolation, late task cancellation, and drain state**

Add tests proving failed sessions cancel pending and not-yet-running work without stopping unrelated sessions, late completions cannot revive failed sessions, emission failure is terminal, and drain waits for active task states.

- [x] **Step 5: Run scheduler tests**

Run: `PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_workflow_scheduler.py -q`

Expected: all selected tests pass.

### Task 11: Ready acquire, replica lifecycle, and accelerator ledger

**Files:**
- Modify: `src/workflow/scheduler.py`
- Test: `tests/test_workflow_scheduler_resources.py`

**Interfaces:**
- Consumes: Task 4 lifecycle core, scheduler artifacts, token policy, deployment profile, and replica reports.
- Produces: grants, ready load actions, replica state transitions, one-accelerator ownership, and one OOM reacquire.

- [x] **Step 1: Write resource-state tests**

```python
def test_agent_releases_replica_before_output_emission() -> None:
    core = scheduler_core_with_idle_replica()
    acquire_id = core.request_ready_agent("s1", "agent", input_tokens=512)
    core.tick_once(now=10.0)
    grant = core.poll_grant(acquire_id)
    decision = core.complete_agent(successful_agent_report(grant))
    assert decision.emit_output is True
    assert core.replicas[grant.replica_id].state == ModelReplicaState.IDLE
```

- [x] **Step 2: Run tests and confirm resource scheduling is absent**

Run: `PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_workflow_scheduler_resources.py -q`

Expected: tests fail because ready placement, grants, and replica lifecycle are absent.

- [x] **Step 3: Implement deterministic ready grant and load passes**

`tick_once()` first grants reusable idle pairs, then reserves free same-kind accelerators and emits ready load actions. `LOADING`/`BUSY` pairs wait, and every mutation checks pair uniqueness and non-overlapping accelerator ownership.

- [x] **Step 4: Implement load completion, lease release, and OOM reacquire**

Load completion binds the reported physical GPU id atomically. Normal completion releases `BUSY -> IDLE`; OOM marks `SUSPECT`, updates only the in-memory exact-bucket penalty, allows one reacquire, and fails the session on the second OOM.

- [x] **Step 5: Run lifecycle and resource tests**

Run: `PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_workflow_scheduler.py tests/test_workflow_scheduler_resources.py -q`

Expected: all selected tests pass.

### Task 10: Near-ready prefetch, history, and future-reuse eviction

**Files:**
- Modify: `src/workflow/scheduler.py`
- Test: `tests/test_workflow_scheduler_policy.py`

**Interfaces:**
- Consumes: `SchedulerCore`, workflow DAG state, rolling runtime history, deployment profile, `select_token_budget()`, and `select_eviction_victim()`.
- Produces: near-ready estimates, explicit prefetch deadlines, history aggregates, and eviction actions.

- [x] **Step 1: Write policy-integration tests**

```python
def test_ready_work_outranks_near_ready_prefetch() -> None:
    core = scheduler_core_with_one_free_accelerator()
    core.request_ready_agent("ready-session", "ready-node")
    core.record_running_upstream("near-session", "upstream", finish_at=20.0)
    actions = core.tick_once(now=10.0)
    assert actions[0].reason == "ready_load"
```

- [x] **Step 2: Run tests and confirm missing policy integration**

Run: `PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_workflow_scheduler_policy.py -q`

Expected: tests fail because near-ready, history, and future-reuse behavior is absent.

- [x] **Step 3: Implement bounded rolling history and near-ready deadlines**

History exposes `output_tokens_p90`, `hit_limit_rate`, `duration_sec_ema`, and `oom_count`. Near-ready timing uses running dependency ETA and `prefetch_at = upstream_eta - load_sec - eps_time_sec`; no wall-clock polling or I/O enters `tick_once()`.

- [x] **Step 4: Integrate prefetch and eviction ordering**

Ready grant and ready load always precede near-ready prefetch. Prefetch cannot evict a replica needed by ready work, only legal victims participate, and one accelerator emits at most one eviction action per tick.

- [x] **Step 5: Run core and policy tests**

Run: `PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_workflow_scheduler.py tests/test_workflow_scheduler_policy.py -q`

Expected: all selected tests pass.

### Task 9: Async scheduler actor and lifecycle watchers

**Files:**
- Modify: `src/workflow/scheduler.py`
- Test: `tests/test_workflow_scheduler_actor.py`

**Interfaces:**
- Consumes: `SchedulerCore`, `ModelReplicaActor`, `TraceWriterActor`, resolved Ray node ids, and typed lifecycle actions.
- Produces: `SchedulerActor`, the worker-facing async protocol, non-blocking replica/trace watchers, and run-level failure state.

- [x] **Step 1: Write mailbox and watcher tests**

```python
def test_scheduler_actor_processes_commands_only_through_run_loop(ray_session) -> None:
    actor = SchedulerActor.remote(CORE_INPUTS)
    run_ref = actor.run.remote()
    ray.get(actor.register_session.remote("session-1", 1.0))
    assert ray.get(actor.get_session_state.remote("session-1")) == SessionState.ACTIVE
    ray.get(actor.stop_loop.remote())
    ray.get(run_ref)
```

- [x] **Step 2: Run the actor tests and confirm the missing actor failure**

Run: `PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_workflow_scheduler_actor.py -q`

Expected: tests fail because `SchedulerActor` is not defined.

- [x] **Step 3: Implement the async actor mailbox**

```python
async def run(self) -> None:
    while not self._stop_loop:
        commands = await self._next_command_batch()
        for command in commands:
            self._apply_command(command)
        actions = self.core.tick_once(time.time())
        self._dispatch_actions(actions)
        self._submit_trace_batch()
```

Public mutating APIs enqueue commands and await acknowledgements. Model load, eviction, worker cancellation, and trace append watchers only enqueue typed completion commands back to the mailbox.

- [x] **Step 4: Verify event-driven deadlines and failure propagation**

Tests prove commands wake the loop before `max_tick_interval_sec`, load/eviction completion changes state without an unrelated acquire, trace append failures become run-level failures, and `tick_once()` never performs Ray or file I/O.

- [x] **Step 5: Run scheduler actor tests**

Run: `PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_workflow_scheduler.py tests/test_workflow_scheduler_actor.py -q`

Expected: all selected tests pass.

### Task 5: Model replica and prompt token boundary

**Files:**
- Create: `src/workflow/replica.py`
- Test: `tests/test_workflow_replica.py`

**Interfaces:**
- Consumes: deployment configuration and `granted_max_new_tokens`.
- Produces: `ModelReplica`, `ModelReplicaActor`, exact prompt token counting, load identity reports, and inference reports.

- [x] **Step 1: Write fake-backend tests**

```python
def test_replica_uses_granted_budget() -> None:
    backend = FakeBackend()
    replica = ModelReplica(DEPLOYMENT, backend_factory=lambda _: backend)
    replica.load()
    result = replica.invoke("prompt", input_tokens=3, max_new_tokens=96, generation=GENERATION)
    assert backend.max_new_tokens == 96
    assert result.output_tokens == 4
```

- [x] **Step 2: Run the focused tests and confirm the missing-module failure**

Run: `PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_workflow_replica.py -q`

Expected: collection fails because `workflow.replica` does not exist.

- [x] **Step 3: Implement one-GPU Hugging Face backend**

The actor loads weights once on process-local `cuda:0`, reports `ray.get_gpu_ids()`, measures load/inference duration with local `perf_counter`, accepts request-specific generation settings, and never receives host-level `cuda:N` strings.

- [x] **Step 4: Normalize backend failures at the replica boundary**

CUDA OOM produces a typed OOM report and marks the replica suspect through scheduler completion. Other backend exceptions remain explicit node execution failures.

- [x] **Step 5: Run replica tests**

Run: `PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_workflow_replica.py -q`

Expected: all selected CPU fake-backend tests pass.

### Task 6: Async node worker and execution functions

**Files:**
- Replace: `src/workflow/worker.py`
- Test: `tests/test_workflow_worker.py`

**Interfaces:**
- Consumes: workflow node config, one Ray input queue, successor queues, scheduler/result actor handles, registered callables, and immutable session input references.
- Produces: `NodeWorker`, `NodeWorkerActor`, `resolve_fanin()`, `build_prompt_context()`, `execute_function()`, and `execute_agent()`.

- [x] **Step 1: Write fan-in, prompt, routing, retry, and cancellation tests**

```python
def test_fanin_is_scoped_by_session_and_source() -> None:
    store: FaninStore = {}
    assert resolve_fanin(store, ("left", "right"), item("s1", "right")) is None
    ready = resolve_fanin(store, ("left", "right"), item("s1", "left"))
    assert ready is not None
    assert tuple(ready.states) == ("left", "right")
    assert "s1" not in store
```

- [x] **Step 2: Run the focused tests and confirm behavior failures against the old worker**

Run: `PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_workflow_worker.py -q`

Expected: tests fail because the old worker lacks structured inputs, function routing, async control, and scheduler protocol.

- [x] **Step 3: Implement async queue consumption and source-indexed fan-in**

`run()` awaits `get_async()` with a short control wake-up, discards cancelled session items, and never carries stop messages in data queues. Synchronous functions run through `asyncio.to_thread`; awaitable functions are awaited directly.

- [x] **Step 4: Implement prompt validation before acquire**

The context contains all external input keys plus `content`, `node_outputs_json`, and `previous_output` only for one dependency. Unknown template fields raise before `request_acquire()`.

- [x] **Step 5: Implement common begin/complete/emit/finish lifecycle**

Broadcast results go to every successor. Targeted function results must exactly match successor names. Terminal persistence is acknowledged before `finish_node()`. Exhausted retries call `fail_node()` and purge that session without stopping the actor.

- [x] **Step 6: Run worker tests**

Run: `PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_workflow_worker.py -q`

Expected: all selected tests pass.

### Task 7: Controller lifecycle, Ray wiring, and shutdown

**Files:**
- Replace: `src/workflow/controller.py`
- Test: `tests/test_workflow_controller.py`
- Test: `tests/test_workflow_end_to_end.py`

**Interfaces:**
- Consumes: validated workflow, function registry, scheduler config, output directory, and optional replica/tokenizer test seams.
- Produces: `WorkflowController.from_yaml()`, `start()`, `submit()`, result/status APIs, `stop_now()`, and `drain_and_stop()`.

- [x] **Step 1: Write construction and queue-wiring tests**

```python
def test_each_edge_reuses_target_input_queue(ray_session) -> None:
    queues, outputs = prepare_queues(WORKFLOW)
    assert outputs["split"]["left"] is queues["left"]
    assert outputs["split"]["right"] is queues["right"]
```

- [x] **Step 2: Write a function-only end-to-end test before replacing the controller**

The test submits several sessions to a targeted split, two parallel branches, and a fan-in terminal function. It asserts duplicate admission is rejected, every terminal state is persisted, sessions do not mix, and the three output artifacts exist after drain.

- [x] **Step 3: Run controller tests and confirm failures against the old controller**

Run: `PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_workflow_controller.py tests/test_workflow_end_to_end.py -q`

Expected: tests fail because the old controller creates actors before full validation and lacks structured submission, result/trace actors, scheduler ownership, and drain semantics.

- [x] **Step 4: Implement validation-before-side-effects and actor construction**

Function registrations, prediction/profile paths, model paths, Ray initialization, live host/GPU counts, and graph invariants are checked before data queues and runtime actors are created.

- [x] **Step 5: Implement admission and lifecycle ownership**

`submit()` records scheduler admission before `ray.put()` and initial enqueue, retains one input reference per active session, and rejects duplicate ids. `drain_and_stop()` closes admission, waits for scheduler terminal states and trace acknowledgements, stops workers, evicts replicas, closes actors/queues, releases input references, and writes the summary. Timeout raises without forcing cleanup.

- [x] **Step 6: Implement bounded immediate cleanup**

`stop_now()` closes admission, cancels pending scheduler work, requests worker stop, kills actors that do not stop within the cleanup timeout, force-shuts queues, and does not claim incomplete sessions succeeded.

- [x] **Step 7: Run controller and end-to-end tests**

Run: `PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_workflow_controller.py tests/test_workflow_end_to_end.py -q`

Expected: all selected tests pass.

### Task 8: Example artifacts, compatibility checks, and full verification

**Files:**
- Create: `example/workflow/runtime.yaml`
- Create: `example/workflow/predictions.yaml`
- Modify: `example/workflow/profile.yaml`
- Test: `tests/test_workflow_examples.py`

**Interfaces:**
- Consumes: the public schema and artifact loaders.
- Produces: one parseable agent/function runtime example without dataset-specific runtime logic.

- [x] **Step 1: Add an example-validation test**

```python
def test_runtime_examples_validate() -> None:
    workflow = load_workflow("example/workflow/runtime.yaml")
    predictions = load_prediction_cache("example/workflow/predictions.yaml")
    profile = load_deployment_profile("example/workflow/profile.yaml")
    assert workflow.graph.entry_node == "prepare"
    assert predictions.version == 1
    assert profile.version == 1
```

- [x] **Step 2: Run the example test and confirm it fails before artifacts exist**

Run: `PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_workflow_examples.py -q`

Expected: test fails because the runtime and prediction examples do not exist.

- [x] **Step 3: Add minimal generic examples and validate cache-key compatibility**

The example uses one broadcast function entry and one agent terminal. Prediction rows cover the configured V100/A100 buckets and use `WorkflowModelFeatureKey` fields unchanged.

- [x] **Step 4: Run focused workflow tests**

Run: `PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_workflow_*.py -q`

Expected: all workflow tests pass.

- [x] **Step 5: Run static checks**

Run: `./.venv/bin/ruff check src/workflow tests/test_workflow_*.py`

Expected: no Ruff diagnostics.

Run: `./.venv/bin/ty check src/workflow tests/test_workflow_*.py`

Expected: no type errors in the new workflow surface.

- [ ] **Step 6: Run the full repository suite**

Run: `PYTHONPATH=src ./.venv/bin/python -m pytest -q`

Expected: all repository tests pass.

Observed on 2026-07-10: 538 tests passed and 27 unrelated existing tests failed because two Gemma4 configs and four recommender builder modules are absent, and three monitoring CLI mocks use the old call signature. All 202 workflow tests pass independently.

- [x] **Step 7: Verify repository hygiene**

Run: `git status --short`

Expected: only requested workflow deliverables and the user's pre-existing changes are present; no temporary PDF, Ray, cache, output, or review files remain.
