from __future__ import annotations

import os
import pickle
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import ray
import torch
import yaml
from langchain.agents import AgentState
from langgraph.graph.state import CompiledStateGraph
from ray.util.queue import Empty, Queue
from torch import nn
from torch_geometric.data.data import Data
from transformers import (
    Gemma4ForCausalLM,
    Gemma4TextConfig,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from common.log import get_logger
from gnn_archs.causal_lm_builder import (
    CausalLMDecodeOnnxExport,
    CausalLMPrefillOnnxExport,
)
from gnn_archs.variant_runner import temporary_causal_lm_export_mode
from gnn_model.data.onnx_graph import build_graph_data_from_onnx
from workflow.cache_config import load_graph_feature_cache
from workflow.model import build_model_graph_feature
from workflow.types import (
    FailureRecord,
    WorkerQueueItem,
    WorkerState,
    Workflow,
    WorkflowDataItem,
    WorkflowModelFeatureKey,
    WorkflowStatus,
)
from workflow.worker import NodeWorker

log_name = "workflow_controller"
logger = get_logger(log_name)


class WorkflowController:
    def __init__(
        self,
        workflow: Workflow,
        graph_feature_cache_dir_list: list[Path | str] | None = None,
        agent_factory: Callable[..., CompiledStateGraph] | None = None,
    ):
        self.workflow = workflow
        input_queue_map, output_queues_map, adjacency_list, dependencies = (
            prepare_queues_for_workflow(workflow)
        )
        entry_nodes = [name for name in workflow.node_names() if not dependencies[name]]
        assert len(entry_nodes) == 1, f"多入口节点不在当前阶段支持范围 {entry_nodes}"
        self.entry_node_name = entry_nodes[0]
        self.terminal_node_names = [
            name for name in workflow.node_names() if not adjacency_list[name]
        ]
        self.input_queue_map = input_queue_map
        self.failure_queue: Queue = Queue()
        self.failures: list[FailureRecord] = []

        logger.info(
            f"Preparing node workers for workflow with {len(workflow.node_names())} nodes"
        )
        self.workers: dict[str, ray.actor.ActorHandle] = dict()
        self.workers = prepare_node_workers(
            workflow,
            input_queue_map,
            output_queues_map,
            dependencies,
            self.failure_queue,
            agent_factory,
        )

        logger.info(f"WorkflowController initialized with {len(self.workers)} workers")
        self.torch_models: dict[str, nn.Module] = prepare_torch_model_for_workflow(
            workflow
        )

        self.graph_feature_cache: dict[WorkflowModelFeatureKey, Data] = {}
        if graph_feature_cache_dir_list is not None:
            logger.info(
                f"Loading graph feature cache from {graph_feature_cache_dir_list}"
            )
            for d in graph_feature_cache_dir_list:
                cache_dir = Path(d)
                if not cache_dir.exists():
                    raise ValueError(
                        f"graph feature cache dir {cache_dir} does not exist"
                    )
                self.graph_feature_cache.update(load_graph_feature_cache(cache_dir))
        else:
            logger.info("No graph feature cache dir provided, skipping cache load")

        self.prepare_parallism: int = 16
        if not ray.is_initialized():
            pass  # TODO

    def start_workflow(self) -> None:
        node_map = self.workflow.node_map()
        load_refs = []
        for node_name, worker in self.workers.items():
            if node_map[node_name].execution is None:
                continue
            logger.info(f"Loading model for node: {node_name}")
            load_refs.append(worker.load.remote())
        ray.get(load_refs)
        for node_name, worker in self.workers.items():
            logger.info(f"Starting worker for node: {node_name}")
            worker.loop.remote()  # 限定每个 worker 使用一个设备

    def submit(self, session_id: str, message: AgentState) -> None:
        self.entry_queue.put(
            WorkerQueueItem(
                data=WorkflowDataItem(
                    session_id=session_id,
                    source_node=self.entry_node_name,
                    target_node=self.entry_node_name,
                    message=message,
                )
            )
        )

    def is_session_complete(self, session_id: str) -> bool:
        checks = [
            self.workers[name].has_result.remote(session_id)
            for name in self.terminal_node_names
        ]
        return all(ray.get(checks))

    def get_session_results(self, session_id: str) -> dict[str, AgentState]:
        return {
            name: ray.get(self.workers[name].get_result.remote(session_id))
            for name in self.terminal_node_names
        }

    def stop_workflow(self) -> None:
        logger.info("Stopping workers")
        self.entry_queue.put(WorkerQueueItem(worker_state=WorkerState.STOPPED))

    def get_workflow_status(self) -> WorkflowStatus:
        node_states = {
            name: ray.get(worker.get_state.remote())
            for name, worker in self.workers.items()
        }
        queue_sizes = {
            name: queue.qsize() for name, queue in self.input_queue_map.items()
        }
        while True:
            try:
                self.failures.append(self.failure_queue.get_nowait())
            except Empty:
                break
        return WorkflowStatus(
            node_states=node_states,
            queue_sizes=queue_sizes,
            failures=list(self.failures),
        )

    @property
    def entry_queue(self) -> Queue:
        assert self.entry_node_name in self.input_queue_map, self.entry_node_name
        return self.input_queue_map[self.entry_node_name]

    @classmethod
    def from_yaml(
        cls,
        path: str,
        agent_factory: Callable[..., CompiledStateGraph] | None = None,
    ) -> WorkflowController:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(Workflow.model_validate(raw), agent_factory=agent_factory)


def prepare_node_workers(
    workflow: Workflow,
    input_queue_map: dict[str, Queue],
    output_queues_map: dict[str, dict[str, Queue]],
    dependencies: dict[str, list[str]],
    failure_queue: Queue,
    agent_factory: Callable[..., CompiledStateGraph] | None = None,
) -> dict[str, ray.actor.ActorHandle]:
    node_map = workflow.node_map()
    node_workers = {}
    for node_name in workflow.node_names():
        node_info = node_map[node_name]
        node_workers[node_name] = NodeWorker.remote(
            node_name=node_name,
            execution_config=node_info.execution,
            prompt_template=node_info.prompt_template,
            input_queue=input_queue_map[node_name],
            output_queues=output_queues_map[node_name],
            tools=[],  # 当前场景暂时不需要
            system_prompt=node_info.system_prompt,
            dependencies=dependencies[node_name],
            retry_config=node_info.retry,
            failure_queue=failure_queue,
            agent_factory=agent_factory,
        )
    return node_workers


def prepare_queues_for_workflow(
    workflow: Workflow,
) -> tuple[
    dict[str, Queue],
    dict[str, dict[str, Queue]],
    dict[str, list[str]],
    dict[str, list[str]],
]:
    node_names = workflow.node_names()
    node_map = workflow.node_map()
    adjacency_list: dict[str, list[str]] = {node_name: [] for node_name in node_names}
    dependencies: dict[str, list[str]] = {node_name: [] for node_name in node_names}
    for edge in workflow.edges:
        adjacency_list[edge.source].append(edge.target)
        dependencies[edge.target].append(edge.source)
    # 每个节点一个共享 input queue 上游 output queue 必须复用同一对象 否则边不连通
    input_queue_map = {
        node_name: Queue(maxsize=node_map[node_name].queue_capacity)
        for node_name in node_names
    }
    output_queues_map = {
        source_node: {
            target_node: input_queue_map[target_node]
            for target_node in adjacency_list[source_node]
        }
        for source_node in node_names
    }
    return input_queue_map, output_queues_map, adjacency_list, dependencies


def prepare_torch_model_for_workflow(
    workflow: Workflow,
) -> dict[str, nn.Module]:
    """
    从 workflow 中扫描所有节点，获取 unqie torch model
    由于 batch size 和 sequence length 未知，暂时只缓存模型实例，不负责ONNX导出和特征抽取的逻辑。
    """
    node_map = workflow.node_map()
    # model_name 应该和模型图特征缓存中定义的模型名称对应
    model_path_map = {}
    for node_info in node_map.values():
        model_path_map[node_info.execution.model_name] = node_info.execution.model_path

    models: dict[str, nn.Module] = {}
    for model_name, model_path in model_path_map.items():
        p = Path(model_path)
        if not p.exists():
            raise ValueError(f"model path {model_path} does not exist")

        if "qwen3" in model_name:
            model = Qwen3ForCausalLM.from_pretrained(
                model_path,
                dtype=torch.float16,
                local_files_only=True,
            )
        elif "gemma4" in model_name:
            model = Gemma4ForCausalLM.from_pretrained(
                model_path,
                dtype=torch.float16,
                local_files_only=True,
            )
        models[model_name] = model
    return models


@ray.remote
def _build_graph_feature_remote(
    model_ref: ray.ObjectRef,
    model_name: str,
    phase: Literal["prefill", "decode"],
    gpu_name: Literal["v100", "a100"],
    batch_size: int,
    sequence_length: int,
    decode_output_length: int,
    seed: int,
) -> tuple[WorkflowModelFeatureKey, Data]:
    logger = get_logger("build_graph_feature_remote")
    model = ray.get(model_ref)
    vocab_size = int(model.config.vocab_size)  # type: ignore[union-attr]
    torch.manual_seed(seed)
    if phase == "prefill":
        input_ids = torch.randint(
            0, vocab_size, (batch_size, sequence_length), dtype=torch.long
        )
        attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.long)
    else:
        input_ids = torch.randint(0, vocab_size, (batch_size, 1), dtype=torch.long)
        attention_mask = torch.ones(
            (batch_size, sequence_length + decode_output_length),
            dtype=torch.long,
        )
    input_map = {"input_ids": input_ids, "attention_mask": attention_mask}
    logger.info(
        f"building feature model={model_name} phase={phase} gpu={gpu_name} "
        f"batch={batch_size} seq={sequence_length} decode={decode_output_length}"
    )
    data = build_model_graph_feature(
        model_name=model_name,
        model=model,
        input_map=input_map,
        input_names=["input_ids", "attention_mask"],
        output_names=[],
        phase=phase,
        gpu_name=gpu_name,
        batch_size=batch_size,
        decode_output_length=decode_output_length,
    )
    key = WorkflowModelFeatureKey(
        model_name=model_name,
        phase=phase,
        gpu_name=gpu_name,
        batch_size=batch_size,
        sequence_length=sequence_length,
        decode_output_length=decode_output_length,
    )
    return key, data


def prepare_graph_feature_for_workflow(
    workflow: Workflow,
    models: dict[str, nn.Module],
    all_phases: set[
        Literal["prefill", "decode"]
    ],  # TODO: 由于 src/gnn_archs 实验流程问题，供GNN预测器使用llm模型decode阶段数据集实际是完整一次generate的实验过程，即prefill+decode，暂时不考虑修复，因此传入改参数应该是decode。
    all_gpu_names: set[Literal["v100", "a100"]],
    parallelism: int = 8,
    cache: dict[WorkflowModelFeatureKey, Data] | None = None,
) -> dict[WorkflowModelFeatureKey, Data]:
    logger = get_logger("prepare_graph_feature_for_workflow")
    logger.info("Preparing graph features for workflow with ray tasks")
    if parallelism <= 0:
        raise ValueError("parallelism must be positive")
    if not models:
        return {}

    node_map = workflow.node_map()
    seen: set[tuple[str, str, str, int, int, int]] = set()
    specs: list[
        tuple[str, Literal["prefill", "decode"], Literal["v100", "a100"], int, int, int]
    ] = []
    for node_info in node_map.values():
        model_name = node_info.execution.model_name
        if model_name not in models:
            raise KeyError(
                f"node references model '{model_name}' not in provided models"
            )
        batch_size = node_info.runtime.batch_size
        sequence_length = node_info.runtime.sequence_length
        decode_max_output_length = node_info.runtime.decode_max_output_length
        if batch_size <= 0 or sequence_length is None:
            raise RuntimeError(
                f"node '{node_info.name}': batch_size and sequence_length must be set"
            )
        for phase in all_phases:
            decode_output_length = (
                decode_max_output_length
                if phase == "decode" and decode_max_output_length
                else 0
            )
            for gpu_name in all_gpu_names:
                ident = (
                    model_name,
                    phase,
                    gpu_name,
                    batch_size,
                    sequence_length,
                    decode_output_length,
                )
                if ident in seen:
                    continue
                seen.add(ident)
                specs.append(ident)

    spec_logger = get_logger("prepare_graph_feature_for_workflow")
    spec_logger.info(f"dispatching {len(specs)} unique feature tasks")
    model_refs: dict[str, ray.ObjectRef] = {
        model_name: ray.put(model) for model_name, model in models.items()
    }
    unit_cpus = max(1, (os.cpu_count() or 1) // parallelism)
    remote_fn = _build_graph_feature_remote.options(num_cpus=unit_cpus)

    features: dict[WorkflowModelFeatureKey, Data] = {}
    refs: list[ray.ObjectRef] = []
    for model_name, phase, gpu_name, batch_size, sequence_length, decode_len in specs:
        ident = (
            model_name,
            phase,
            gpu_name,
            batch_size,
            sequence_length,
            decode_len,
        )
        key = WorkflowModelFeatureKey(
            model_name=model_name,
            phase=phase,
            gpu_name=gpu_name,
            batch_size=batch_size,
            sequence_length=sequence_length,
            decode_output_length=decode_len,
        )
        # 先查找本地数据中看看有没有
        if cache is not None and key in cache:
            data = cache[key]
            logger.info(f"loading cached feature for {key}")
            features[key] = data
            continue
        seed = abs(hash(key)) & 0x7FFFFFFF
        ref = remote_fn.remote(
            model_refs[model_name],
            model_name,
            phase,
            gpu_name,
            batch_size,
            sequence_length,
            decode_len,
            seed,
        )
        refs.append(ref)

    pending = list(refs)
    succeeded = 0
    while pending:
        ready, pending = ray.wait(pending, num_returns=1)
        for ref in ready:
            try:
                key, data = ray.get(ref)
                features[key] = data
                succeeded += 1
            except Exception:
                logger.exception("graph feature build task failed")
    if succeeded != len(refs):
        raise RuntimeError(
            f"feature preparation incomplete: {succeeded}/{len(refs)} tasks succeeded"
        )
    logger.info(
        "graph feature preparation finished: "
        f"{succeeded} features across {len(models)} models"
    )
    return features
