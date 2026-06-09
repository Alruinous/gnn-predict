项目开发环境为容器环境

项目使用的 shell 为 zsh，zsh 中的 $HOST 环境变量可以得到容器的 pod 名称。

项目使用的 python 虚拟环境在 .venv 目录下。

http://192.168.5.60:31110 为集群 Prometheus 地址，根据 https://help.aliyun.com/zh/ack/ack-managed-and-ack-dedicated/user-guide/introduction-to-metrics 介绍的 DCGM 指标，可以获取到容器所用 GPU 对应宿主机 GPU 的编号，以便在查询指标时使用。

如果有CPU、内存或DCGM等指标的查询需求，可以参考 https://github.com/YiD11/prometheus_monitor 的代码。