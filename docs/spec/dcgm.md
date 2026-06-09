GPU监控2.0基于Exporter+Prometheus+Grafana体系打造更为丰富的GPU可观性场景，支持您基于容器服务开发的GPU Exporter监控指标构建Grafana大盘。本文介绍GPU监控2.0指标的详细信息。

## 指标说明

GPU监控2.0使用的GPU Exporter在兼容开源DCGM Exporter提供的监控指标的基础上，根据某些业务场景，增加了自定义指标。关于DCGM Exporter的更多信息，请参见[DCGM Exporter](https://github.com/NVIDIA/dcgm-exporter)。

GPU监控2.0中所涉及的GPU监控指标包括[DCGM支持的指标](#section-ts3-5l7-lwm)和[自定义指标](#section-oin-6mf-6j0)。

## **计费说明**

GPU监控使用自定义指标会引起额外的费用。

为避免产生额外的费用，建议在启用此功能前，仔细阅读[阿里云Prometheus的计费概述](https://help.aliyun.com/zh/arms/prometheus-monitoring/product-overview/billing-overview-2)，了解自定义指标的收费策略。费用将根据您的集群规模和应用数量等因素产生变动。您可以通过[资源消耗统计功能](https://help.aliyun.com/zh/arms/prometheus-monitoring/product-overview/view-resource-usage)，监控和管理您的资源使用情况。

## DCGM支持的指标

### **利用率（Utilization）**

| **指标名称** | **指标类型** | **单位** | **说明** |
| --- | --- | --- | --- |
| DCGM\\_FI\\_DEV\\_GPU\\_UTIL | Gauge | %   | 表示GPU利用率，即在一个周期时间内（1s或1/6s，根据GPU产品而定），一个或多个核函数处于Active的时间。 该指标仅能够展示有核函数在用的GPU资源，但无法展示具体的使用情况。 |
| DCGM\\_FI\\_DEV\\_MEM\\_COPY\\_UTIL | Gauge | %   | 表示内存带宽利用率。 以英伟达GPU V100为例，其最大内存带宽为900 GB/sec，如果当前的内存带宽为450 GB/sec，则内存带宽利用率为50%。 |
| DCGM\\_FI\\_DEV\\_ENC\\_UTIL | Gauge | %   | 表示编码器利用率。 |
| DCGM\\_FI\\_DEV\\_DEC\\_UTIL | Gauge | %   | 表示解码器利用率。 |

### **内存（Memory）**

| **指标名称** | **指标类型** | **单位** | **说明** |
| --- | --- | --- | --- |
| DCGM\\_FI\\_DEV\\_FB\\_FREE | Gauge | MiB | 表示帧缓存（Framebuffer Memory）剩余数。 **说明** 帧缓存一般也被称为显存。 |
| DCGM\\_FI\\_DEV\\_FB\\_USED | Gauge | MiB | 表示帧缓存已使用数。 该值与nvidia-smi命令中Memory-Usage的已使用值对应。 |

### **剖析（Profiling）**

| **指标名称** | **指标类型** | **单位** | **说明** |
| --- | --- | --- | --- |
| DCGM\\_FI\\_PROF\\_GR\\_ENGINE\\_ACTIVE | Gauge | %   | 表示在一个时间间隔内，Graphics或Compute引擎处于Active的时间占比。 该值表示所有Graphics和Compute引擎的平均值。 Graphics或Compute引擎处于Active是指Graphics或Compute Context绑定到线程，并且Graphics或Compute Context处于Busy状态。 |
| DCGM\\_FI\\_PROF\\_SM\\_ACTIVE | Gauge | %   | 表示在一个时间间隔内，至少一个线程束在一个SM（Streaming Multiprocessor）上处于Active的时间占比。 该值表示所有SM的平均值，且该值对每个块的线程数不敏感。 线程束处于Active是指一个线程束被调度且分配资源后的状态，可能是在Computing、也可能是非Computing状态（例如等待内存请求）。 该值小于0.5表示未高效利用GPU，大于0.8是必要的。 假设一个GPU有N个SM： - 一个核函数在整个时间间隔内使用N个线程块运行在所有的SM上，此时该值为1（100%）。 - 一个核函数在一个时间间隔内运行N/5个线程块，此时该值为0.2。 - 一个核函数使用N个线程块，在一个时间间隔内，仅运行了1/5个周期的时间，此时该值为0.2。 |
| DCGM\\_FI\\_PROF\\_SM\\_OCCUPANCY | Gauge | %   | 表示在一个时间间隔内，驻留在SM上的线程束与该SM最大可驻留线程束的比例。 该值表示一个时间间隔内的所有SM的平均值。 占用率越高不代表GPU使用率越高。只有在GPU内存带宽受限的工作负载（DCGM\\_FI\\_PROF\\_DRAM\\_ACTIVE）情况下，更高的占用率表示更有效的GPU使用率。 |
| DCGM\\_FI\\_PROF\\_PIPE\\_TENSOR\\_ACTIVE | Gauge | %   | 表示Tensor（HMMA/IMMA） Pipe处于Active状态的周期分数。 该值表示一个时间间隔内的平均值，而不是瞬时值。 较高的值表示Tensor Cores的利用率较高。 该值为1（100%）表示在整个时间间隔内每隔一个指令周期发出一个Tensor指令（两个周期完成一条指令）。 假设该值为0.2（20%），可能有如下情况： - 在整个时间间隔内，有20%的SM的Tensor Core以100%的利用率运行。 - 在整个时间间隔内，有100%的SM的Tensor Core以20%的利用率运行。 - 在整个时间间隔的1/5时间内，有100%的SM上的Tensor Core以100%利用率运行。 - 其他组合模式。 |
| DCGM\\_FI\\_PROF\\_PIPE\\_FP64\\_ACTIVE | Gauge | %   | 表示FP64（双精度）Pipe处于Active状态的周期分数。 该值表示一个时间间隔内的平均值，而不是瞬时值。 较高的值代表FP64 Cores有较高的利用率。 该值为 1（100%）表示在整个时间间隔内上每四个周期（以Volta类型卡为例）执行一次FP64指令。 假设该值为0.2（20%），可能有如下情况： - 在整个时间间隔内，有20%的SM的FP64 Core以100%的利用率运行。 - 在整个时间间隔内，有100%的SM的FP64 Core以20%的利用率运行。 - 在整个时间间隔的1/5时间内，有100%的SM上的FP64 Core以100%利用率运行。 - 其他组合模式。 |
| DCGM\\_FI\\_PROF\\_PIPE\\_FP32\\_ACTIVE | Gauge | %   | 表示乘加操作FMA（Fused Multiply-Add）管道处于Active的周期分数，乘加操作包括FP32（单精度）和整数。 该值表示一个时间间隔内的平均值，而不是瞬时值。 较高的值代表FP32 Cores有较高的利用率。 该值为1（100%）表示在整个时间间隔内上每两个周期（Volta类型卡为例）执行一次FP32指令。 假设该值为0.2（20%），可能有如下情况： - 在整个时间间隔内，有20%的SM的FP32 Core以100%的利用率运行。 - 在整个时间间隔内，有100%的SM的FP32 Core以20%的利用率运行。 - 在整个时间间隔的1/5时间内，有100%的SM上的FP32 Core以100%利用率运行。 - 其他组合模式。 |
| DCGM\\_FI\\_PROF\\_PIPE\\_FP16\\_ACTIVE | Gauge | %   | 表示FP16（半精度）管道处于Active的周期分数。 该值表示一个时间间隔内的平均值，而不是瞬时值。 较高的值代表FP16 Cores有较高的利用率。 该值为 1 (100%) 表示在整个时间间隔内上每两个周期（Volta类型卡为例）执行一次FP16指令。 假设该值为0.2（20%），可能有如下情况： - 在整个时间间隔内，有20%的SM的FP16 Core以100%的利用率运行。 - 在整个时间间隔内，有100%的SM的FP16 Core以20%的利用率运行。 - 在整个时间间隔的1/5时间内，有100%的SM上的FP16 Core以100%利用率运行。 - 其他组合模式。 |
| DCGM\\_FI\\_PROF\\_DRAM\\_ACTIVE | Gauge | %   | 表示内存带宽利用率（Memory BW Utilization）是将数据发送到设备内存或从设备内存接收数据的周期分数。 该值表示时间间隔内的平均值，而不是瞬时值。 较高的值表示设备内存的利用率较高。 该值为1（100%）表示在整个时间间隔内的每个周期执行一条 DRAM 指令（实际上，峰值约为 0.8 (80%) 是可实现的最大值）。 假设该值为0.2（20%），表示20%的周期在时间间隔内读取或写入设备内存。 |
| - DCGM\\_FI\\_PROF\\_PCIE\\_TX\\_BYTES - DCGM\\_FI\\_PROF\\_PCIE\\_RX\\_BYTES | Counter | B/s | 表示通过PCIe总线传输/接收的数据速率，包括协议标头和数据有效负载。 该值表示一个时间间隔内的平均值，而不是瞬时值。 该速率在时间间隔内平均。例如，在1秒内传输1 GB数据，则无论以恒定速率还是突发传输数据，速率都是1 GB/s。理论上的最大PCIe Gen3带宽为每通道985 MB/s。 |
| - DCGM\\_FI\\_PROF\\_NVLINK\\_RX\\_BYTES - DCGM\\_FI\\_PROF\\_NVLINK\\_TX\\_BYTES | Counter | B/s | 表示通过NVLink传输/接收的数据速率，不包括协议标头。 该值表示一个时间间隔内的平均值，而不是瞬时值。 该速率在时间间隔内平均。例如，在1秒内传输1 GB数据，则无论以恒定速率还是突发传输数据，速率都是1 GB/s。理论上，最大NVLink Gen2带宽为每个方向每个链路25 GB/s。 |

### **频率（Clock）**

| **指标名称** | **指标类型** | **单位** | **说明** |
| --- | --- | --- | --- |
| DCGM\\_FI\\_DEV\\_SM\\_CLOCK | Gauge | MHz | 表示SM时钟频率。 |
| DCGM\\_FI\\_DEV\\_MEM\\_CLOCK | Gauge | MHz | 表示内存时钟频率。 |
| DCGM\\_FI\\_DEV\\_APP\\_SM\\_CLOCK | Gauge | MHz | 表示SM应用时钟频率。 |
| DCGM\\_FI\\_DEV\\_APP\\_MEM\\_CLOCK | Gauge | MHz | 表示内存应用时钟频率。 |
| DCGM\\_FI\\_DEV\\_CLOCK\\_THROTTLE\\_REASONS | Gauge | MHz | 表示时钟频率降低的原因。 |

### **XID错误&违规行为（XidError & Violations）**

| **指标名称** | **指标类型** | **单位** | **说明** |
| --- | --- | --- | --- |
| DCGM\\_FI\\_DEV\\_XID\\_ERRORS | Gauge | \\- | 表示一段时间内，最后发生的XID错误号。 |
| DCGM\\_FI\\_DEV\\_POWER\\_VIOLATION | Counter | μs（微秒） | 表示因功率上限而导致的违规，该值为违规的时间。 |
| DCGM\\_FI\\_DEV\\_THERMAL\\_VIOLATION | Counter | μs（微秒） | 表示由于热限制导致的违规，该值为违规的时间。 |
| DCGM\\_FI\\_DEV\\_SYNC\\_BOOST\\_VIOLATION | Counter | μs（微秒） | 表示由于同步提升限制而导致的违规，该值为违规的时间。 |
| DCGM\\_FI\\_DEV\\_BOARD\\_LIMIT\\_VIOLATION | Counter | μs（微秒） | 表示由于电路板限制而导致的违规，该值为违规的时间。 |
| DCGM\\_FI\\_DEV\\_LOW\\_UTIL\\_VIOLATION | Counter | μs（微秒） | 表示低利用率限制导致的违规，该值为违规的时间。 |
| DCGM\\_FI\\_DEV\\_RELIABILITY\\_VIOLATION | Counter | μs（微秒） | 表示电路板可靠性限制导致违规，该值为违规的时间。 |

### **BAR1**

| **指标名称** | **指标类型** | **单位** | **说明** |
| --- | --- | --- | --- |
| DCGM\\_FI\\_DEV\\_BAR1\\_USED | Gauge | MB  | 表示已使用的BAR1。 |
| DCGM\\_FI\\_DEV\\_BAR1\\_FREE | Gauge | MB  | 表示剩余的BAR1。 |

### **温度&功率（Temperature & Power）**

| **指标名称** | **指标类型** | **单位** | **说明** |
| --- | --- | --- | --- |
| DCGM\\_FI\\_DEV\\_MEMORY\\_TEMP | Gauge | C   | 表示内存温度。 |
| DCGM\\_FI\\_DEV\\_GPU\\_TEMP | Gauge | C   | 表示GPU温度。 |
| DCGM\\_FI\\_DEV\\_POWER\\_USAGE | Gauge | W   | 表示功率。 |
| DCGM\\_FI\\_DEV\\_TOTAL\\_ENERGY\\_CONSUMPTION | Counter | J   | 表示从驱动加载开始，已消耗的能量。 |

### **停用的内存页面（Retired Pages）**

| **指标名称** | **指标类型** | **单位** | **说明** |
| --- | --- | --- | --- |
| DCGM\\_FI\\_DEV\\_RETIRED\\_SBE | Gauge | \\- | 表示由单比特错误（Single Bit Error）而停用的页面。 |
| DCGM\\_FI\\_DEV\\_RETIRED\\_DBE | Gauge | \\- | 表示由双比特错误（Double Bit Error）而停用的页面。 |

## 自定义指标

| **指标名称** | **指标类型** | **单位** | **说明** |
| --- | --- | --- | --- |
| DCGM\\_CUSTOM\\_PROCESS\\_SM\\_UTIL | Gauge | %   | 表示GPU线程的SM利用率。 |
| DCGM\\_CUSTOM\\_PROCESS\\_MEM\\_COPY\\_UTIL | Gauge | %   | 表示GPU线程的内存拷贝利用率。 |
| DCGM\\_CUSTOM\\_PROCESS\\_ENCODE\\_UTIL | Gauge | %   | 表示GPU线程的编码器利用率。 |
| DCGM\\_CUSTOM\\_PROCESS\\_DECODE\\_UTIL | Gauge | %   | 表示GPU线程的解码器利用率。 |
| DCGM\\_CUSTOM\\_PROCESS\\_MEM\\_USED | Gauge | MiB | 表示GPU线程当前使用的显存。 |
| DCGM\\_CUSTOM\\_CONTAINER\\_MEM\\_ALLOCATED | Gauge | MiB | 表示为容器分配显存。 |
| DCGM\\_CUSTOM\\_CONTAINER\\_CP\\_ALLOCATED | Gauge | \\- | 表示为容器分配的一张GPU卡上部分算力占该GPU卡总算力的比例。值的区间为\\[0, 1\\]。 独占GPU或者共享GPU只申请显存时，该指标的值为0，表示不限制算力。 假设GPU卡有100算力，为一个容器分配30算力，那么为该容器分配的算力比例为30/100=0.3。 |
| DCGM\\_CUSTOM\\_DEV\\_FB\\_TOTAL | Gauge | MiB | 表示GPU卡的总显存。 |
| DCGM\\_CUSTOM\\_DEV\\_FB\\_ALLOCATED | Gauge | \\- | 表示GPU卡已分配显存占总显存的比例，值的区间为\\[0, 1\\]。 |
| DCGM\\_CUSTOM\\_ALLOCATE\\_MODE | Gauge | \\- | 表示节点运行的模式，有如下几种： - 0：代表值为None，表示当前节点没有GPU Pod在运行。 - 1：代表值为Exclusive，表示当前节点的GPU Pod以独占GPU方式运行。 - 2：代表值为Share，表示当前节点GPU Pod以共享GPU方式运行。 |

## 将被弃用的指标

| **将被弃用的指标** | **替换的新指标** | **说明** |
| --- | --- | --- |
| nvidia\\_gpu\\_temperature\\_celsius | DCGM\\_FI\\_DEV\\_GPU\\_TEMP |     |
| nvidia\\_gpu\\_power\\_usage\\_milliwatts | DCGM\\_FI\\_DEV\\_POWER\\_USAGE |     |
| nvidia\\_gpu\\_sharing\\_memory | DCGM\\_CUSTOM\\_DEV\\_FB\\_ALLOCATED \\* DCGM\\_CUSTOM\\_DEV\\_FB\\_TOTAL | 每一张GPU设备显存被申请的比例 \\* 该设备显存的总量 = 该设备被申请的显存数 |
| nvidia\\_gpu\\_memory\\_used\\_bytes | DCGM\\_FI\\_DEV\\_FB\\_USED |     |
| nvidia\\_gpu\\_memory\\_total\\_bytes | DCGM\\_CUSTOM\\_DEV\\_FB\\_TOTAL |     |
| nvidia\\_gpu\\_memory\\_allocated\\_bytes | DCGM\\_CUSTOM\\_DEV\\_FB\\_ALLOCATED \\* DCGM\\_CUSTOM\\_DEV\\_FB\\_TOTAL | 每一张GPU设备显存被申请的比例 \\* 该设备显存的总量 = 该设备被申请的显存数 |
| nvidia\\_gpu\\_duty\\_cycle | DCGM\\_FI\\_DEV\\_GPU\\_UTIL |     |
| nvidia\\_gpu\\_allocated\\_num\\_devices | sum(DCGM\\_CUSTOM\\_DEV\\_FB\\_ALLOCATED) | sum(节点上每一张GPU设备显存被申请的比例) = 该节点上被申请的总卡数 |
| nvidia\\_gpu\\_num\\_devices | DCGM\\_FI\\_DEV\\_COUNT |     |