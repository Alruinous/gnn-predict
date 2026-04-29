# 特征提取迁移

## 背景

当前项目完成了 docs/migrate/remaining_migration_summary.md 和 docs/migrate/gnn_model_migration_result.md的迁移。即从/home/wangjh/gnn-schedule/gnn_model和/home/wangjh/gnn-schedule/gen_archs两个旧项目迁移到当下。

## 目标

现在res目录下已经有各变体实验的CSV数据集。在过往实验中，会使用/home/wangjh/gnn-schedule/gnn_model/src/data的代码手工执行CSV数据集ETL成pytorch geometric dataset的工作。我希望当前这个项目也是如此。

但源项目的CSV schema和当前CSV schema已有很大不同，有必要先分析现有数据表的schema和了解docs/monitoring-csv-schema.md的解释。

重要提示：源项目中的ETL代码有些比较混乱甚至有的是错误的。README也有误导性，因为开发者长期不更新文档，也不删除错误或无用代码。

## /home/wangjh/gnn-schedule/gnn_model/src/data解释

### /home/wangjh/gnn-schedule/gnn_model/src/data/dataset.py

定义了Dataset类，由于当时数据集太大，无法将全量数据集加载到内存中，加之当时代码有内存泄露情况，因此使用IterableDataset实现。但当前项目不一定非要这样实现，因为IterableDataset不利于分batch和shuffle等操作。

### /home/wangjh/gnn-schedule/gnn_model/src/data/enhanced_data_processor.py

是错误的代码，与之关联的代码都需要注意辨别。

### /home/wangjh/gnn-schedule/gnn_model/src/data/extract.py

存放了ETL主要入口，当时为了方便测试，ETL处理模式区分为
- 单进程
- 多进程
- ray分布式
在测试阶段，我调试时习惯使用单进程，实际处理全量数据时我习惯使用ray分布式。多进程模式虽然处理速度较快，但实际发现当时的数据集规模，依然不合适。

ETL过程会先将读取目标目录下的所有CSV数据表，然后枚举每个文件，将表内一个batch的数据行发送到worker进行处理，最终会按照模型类型分成多个pickl序列化文件。每个文件内都是 `list[torch_geometric.data.Data]` 的数据结构。这部分逻辑应该可以被当前项目借鉴。

对于一行数据记录，其包含一次变体实验的：
- 实验类型，训练还是推理
- 模型ONNX路径
- 实验超参数，比如batch_size等
- 实验过程中的各硬件指标，即目标字段
等等

在源项目中/home/wangjh/gnn-schedule/gnn_model/src/data/extract.py的process_csv是处理单个CSV的部分，将每个数据记录行转成一种object。在extract_feature_target中处理这个object，其中onnx_to_pyg_data最为核心。

onnx_to_pyg_data负责读取ONNX模型文件，提取ONNX计算图的全图特征、边特征、节点特征，分别对应extract_graph_features、extract_edge_features和extract_node_features。而extract_node_embedding_features废弃，因为其特征不显著，这种废弃的方法不应该在当前项目中被借鉴。

extract_node_features中会采集onnx_tool提取的各节点特征，我记得onnx_tool能采集memory和macs等信息。其中flops=2*macs也是一个重要特征。但是extract_node_features也有一些错误代码。

比如使用OPERATOR_PERFORMANCE_COEFF获取一个节点的算子特点，但/home/wangjh/gnn-schedule/gnn_model/src/data/const.py中OPERATOR_PERFORMANCE_COEFF等一些重要数据是捏造的。其文件中只有有关显卡规格的硬件参数是可靠的数据。其他很多与算子特性相关的数据都不可靠。

同时extract_node_features的一些特征的编排也不合理，比如显卡特征，这个显然属于全局特征，不应该被重复采集到每个节点上。类似的错误特征编排在源代码中可能还有一些。

去除掉错误的特征后，节点特征维数应该不会很大。

extract_graph_features和extract_edge_features的提取过程相对简答，应该有问题的代码不多。


