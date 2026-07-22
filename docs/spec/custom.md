# 使用 Crater 自定义作业搭建 Ray 集群

Crater 的一个自定义作业对应一个 Pod。搭建 Ray 集群时，先提交一个 Head 作业，再提交若干个
Worker 作业；需要运行计算入口时，可以另外提交 Driver 作业，或者直接在 Head 所在环境中执行。

这种方式不会自动扩缩容，也不会自动发现 Head。所有 Worker 和 Driver 都必须连接同一个明确且
可达的 Ray Head 地址。

## 准备工作

所有作业应满足以下条件：

- 使用相同或兼容的 Ray 版本。
- 挂载相同的项目与数据目录，或使用包含相同依赖的镜像。
- Head 与 Worker Pod 之间网络互通。
- 使用统一端口；以下模板使用 Ray 默认端口 `6379`。

Ray 除 Head 端口外还会使用节点管理、对象管理和 Worker 进程端口。如果集群配置了
NetworkPolicy 或防火墙，需要允许这些 Pod 之间的双向通信。只放通 `6379` 可能不足以运行任务。

## 启动 Head 作业

创建一个自定义作业作为专用 Head。将项目目录和虚拟环境目录替换成实际路径：

```sh
set -eu

PROJECT_DIR=/path/to/project
VENV_DIR="$PROJECT_DIR/.venv"
RAY_PORT="${RAY_PORT:-6379}"

cd "$PROJECT_DIR"
. "$VENV_DIR/bin/activate"

RAY_HEAD_ADDRESS="$(hostname -i | awk '{print $1}')"
printf 'RAY_HEAD_ADDRESS=%s\nRAY_PORT=%s\n' "$RAY_HEAD_ADDRESS" "$RAY_PORT"

exec ray start \
  --head \
  --port="$RAY_PORT" \
  --num-cpus=0 \
  --disable-usage-stats \
  --block
```

`--block` 让 Head 进程持续占据前台，避免启动命令结束后 Pod 退出。`--num-cpus=0` 表示该专用
Head 不承接普通 Ray Task；如果希望 Head 同时参与计算，应将它改为该作业实际申请的 CPU 数量。

Head 启动后，从作业日志复制以下两项：

```text
RAY_HEAD_ADDRESS=<Head Pod IP>
RAY_PORT=6379
```

Pod IP 适合快速搭建临时集群，但 Head Pod 重建后地址可能变化。长时间运行的集群应优先使用指向
Head 的稳定 Service DNS；无论使用 IP 还是域名，最终都必须指向同一个 Head。

## 启动 Worker 作业

每个 Worker 都是独立的自定义作业。创建 Worker 前，在作业环境变量中配置：

```text
RAY_HEAD_ADDRESS=<从 Head 日志复制的地址，不包含端口>
RAY_PORT=6379
RAY_NUM_CPUS=<该 Worker 作业实际申请的 CPU 数>
```

所有 Worker 可以使用同一份启动模板：

```sh
set -u

PROJECT_DIR=/path/to/project
VENV_DIR="$PROJECT_DIR/.venv"
RAY_PORT="${RAY_PORT:-6379}"

: "${RAY_HEAD_ADDRESS:?请配置 RAY_HEAD_ADDRESS}"
: "${RAY_NUM_CPUS:?请配置 RAY_NUM_CPUS}"

cd "$PROJECT_DIR"
. "$VENV_DIR/bin/activate"

until ray start \
  --address="${RAY_HEAD_ADDRESS}:${RAY_PORT}" \
  --num-cpus="$RAY_NUM_CPUS" \
  --block; do
  printf '等待 Ray Head %s:%s\n' "$RAY_HEAD_ADDRESS" "$RAY_PORT" >&2
  sleep 2
done
```

Head 尚未就绪时，Worker 每两秒重试一次。连接成功后，`--block` 会保持 Ray Worker 和 Pod 运行。

`--num-cpus` 声明的是 Ray 调度资源，不会替 Pod 申请 CPU。它不应大于 Crater 中为该 Worker
实际申请的 CPU 数量；不需要显式控制时，也可以删除这个参数及 `RAY_NUM_CPUS`。

需要更多 Worker 时，继续提交使用相同 Head 地址的自定义作业即可。

## 连接 Driver 作业

Driver 必须使用和 Worker 相同的 Head 地址。可以为 Driver 自定义作业配置：

```text
RAY_HEAD_ADDRESS=<Head 地址>
RAY_PORT=6379
```

通用启动模板如下：

```sh
set -eu

PROJECT_DIR=/path/to/project
VENV_DIR="$PROJECT_DIR/.venv"
RAY_PORT="${RAY_PORT:-6379}"

: "${RAY_HEAD_ADDRESS:?请配置 RAY_HEAD_ADDRESS}"

cd "$PROJECT_DIR"
. "$VENV_DIR/bin/activate"

export RAY_ADDRESS="${RAY_HEAD_ADDRESS}:${RAY_PORT}"
exec python /path/to/entrypoint.py
```

应用代码应显式读取该地址，不要使用可能失效的硬编码 Pod IP：

```python
import os

import ray

ray.init(address=os.environ["RAY_ADDRESS"])
```

`ray.init(address="auto")` 只适用于当前环境已经能够发现 Ray 集群的情况。跨自定义作业连接时，
显式传入 `RAY_ADDRESS` 更可靠。

## 验证集群

在能够访问 Head 的任意作业中执行：

```sh
ray status --address="${RAY_HEAD_ADDRESS}:${RAY_PORT}"
```

也可以在 Driver 中检查存活节点：

```python
import ray

print([(node["NodeManagerAddress"], node["Alive"]) for node in ray.nodes()])
```

必须确认 Head 和预期的所有 Worker 都出现在同一个集群中。Worker 命令成功启动并不一定表示它
加入了目标集群；如果不同作业使用了不同且互不等价的 Head 地址，它们会形成不同的 Ray 集群。

