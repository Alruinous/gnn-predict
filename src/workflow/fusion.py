"""Compile maximal same-model agent chains into single fused nodes.

Each hop between two agent nodes costs a queue transfer plus a full acquire/grant
handshake, and leaves a window in which the shared model can be evicted and
reloaded mid-session. When neighbouring nodes deploy byte-identical models the hop
buys nothing, so this pass rewrites the graph before the fleet ever sees it.
"""

from __future__ import annotations

from workflow.replica import ModelDeploymentConfig
from workflow.schema import (
    AgentNodeConfig,
    EdgeConfig,
    FusedAgentNodeConfig,
    FusedStage,
    NodeConfig,
    Workflow,
)

FUSED_NAME_SEPARATOR = "+"


def fuse_workflow(workflow: Workflow) -> Workflow:
    """Return an equivalent workflow with fusable agent chains collapsed."""
    chains = find_fusable_chains(workflow)
    if not chains:
        return workflow

    absorbed: dict[str, str] = {}
    fused_nodes: dict[str, FusedAgentNodeConfig] = {}
    for chain in chains:
        fused = _build_fused_node(chain)
        fused_nodes[chain[0].name] = fused
        for node in chain:
            absorbed[node.name] = fused.name

    nodes: list[NodeConfig] = []
    for node in workflow.nodes:
        if node.name in fused_nodes:
            nodes.append(fused_nodes[node.name])
        elif node.name not in absorbed:
            nodes.append(node)

    edges: list[EdgeConfig] = []
    seen: set[tuple[str, str]] = set()
    for edge in workflow.edges:
        source = absorbed.get(edge.source, edge.source)
        target = absorbed.get(edge.target, edge.target)
        if source == target:
            continue
        if (source, target) in seen:
            continue
        seen.add((source, target))
        edges.append(EdgeConfig(source=source, target=target))

    return Workflow(
        workflow_name=workflow.workflow_name,
        nodes=tuple(nodes),
        edges=tuple(edges),
    )


def find_fusable_chains(
    workflow: Workflow,
) -> tuple[tuple[AgentNodeConfig, ...], ...]:
    """Maximal chains of agent nodes that share a model and a private edge.

    Only the links inside a chain must be one-to-one; the head may fan in and the
    tail may fan out, so a merge node with six upstream chunks still fuses with the
    nodes that follow it.
    """
    graph = workflow.graph
    node_map = workflow.node_map()
    successor_of: dict[str, AgentNodeConfig] = {}
    for name in graph.topological_order:
        node = node_map[name]
        adjacency = graph.adjacency[name]
        if not isinstance(node, AgentNodeConfig) or len(adjacency) != 1:
            continue
        successor = node_map[adjacency[0]]
        if not isinstance(successor, AgentNodeConfig):
            continue
        if graph.dependencies[successor.name] != (name,):
            continue
        if not _same_deployment(node, successor):
            continue
        successor_of[name] = successor

    tails = {successor.name for successor in successor_of.values()}
    chains: list[tuple[AgentNodeConfig, ...]] = []
    for name in graph.topological_order:
        if name in tails or name not in successor_of:
            continue
        head = node_map[name]
        if not isinstance(head, AgentNodeConfig):
            continue
        chain = [head]
        while chain[-1].name in successor_of:
            chain.append(successor_of[chain[-1].name])
        chains.append(tuple(chain))
    return tuple(chains)


def _same_deployment(first: AgentNodeConfig, second: AgentNodeConfig) -> bool:
    if isinstance(first, FusedAgentNodeConfig) or isinstance(
        second, FusedAgentNodeConfig
    ):
        return False
    if (
        ModelDeploymentConfig.from_node(first).model_key
        != ModelDeploymentConfig.from_node(second).model_key
    ):
        return False
    # model_key covers the deployment (path, dtype, serving) but not decoding, and
    # stages share one engine handle, so the rest of the execution config must match.
    ignored = {"max_new_tokens"}
    return first.execution.model_dump(exclude=ignored) == second.execution.model_dump(
        exclude=ignored
    )


def _build_fused_node(chain: tuple[AgentNodeConfig, ...]) -> FusedAgentNodeConfig:
    head = chain[0]
    stages = tuple(
        FusedStage(
            name=node.name,
            prompt_template=node.prompt_template,
            system_prompt=node.system_prompt,
            max_new_tokens=node.execution.max_new_tokens,
        )
        for node in chain
    )
    widest = max(stage.max_new_tokens for stage in stages)
    return FusedAgentNodeConfig(
        name=FUSED_NAME_SEPARATOR.join(node.name for node in chain),
        task=head.task,
        description=head.description,
        queue_capacity=head.queue_capacity,
        retry=head.retry,
        model=head.model,
        execution=head.execution.model_copy(update={"max_new_tokens": widest}),
        prompt_template=head.prompt_template,
        system_prompt=head.system_prompt,
        stages=stages,
    )
