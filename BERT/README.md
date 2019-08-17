# BERT For TensorFlow 使用指南

BERT (Bidirectional Encoder Representations from Transformers) 是一种预训练语言表示的新方法，它可以在各种自然语言处理（NLP）任务中获得最先进的结果。本文介绍如何在TensorFlow平台上BERT的使用，其中包括训练过程和推理过程的使用说明。更多细节参考：

-官方源码: [BERT](https://github.com/google-research/bert)

-论文: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) 

-英伟达BERT Docker [NVIDIA's BERT](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT)

另外，请使用BERT模型的默认参数执行以下步骤，预先训练或微调您的问题应答模型



## BERT镜像构建

首先，从Github中拉取[modelzoo](https://github.com/xxmyjk/modelzoo)项目

```bash
git clone https://github.com/xxmyjk/modelzoo
```

切换至[tf-bert](https://github.com/xxmyjk/modelzoo/tree/tf-bert)分支

```bash
cd modelzoo
git checkout -b tf-bert origin/tf-bert
```

进入BERT目录，构建BERT镜像

```bash
cd benchmark/tensorflow/nlp/bert
bash scripts/docker/build.sh
```



## 数据集和已训练模型

本项目提供了脚本用于数据集的下载，验证和提取。数据集包括SQuaD数据集，用于预训练的Wikipedia和BookCorpus数据集，已训练的模型。

需下载数据集和已训练模型，请执行：

```bash
bash scripts/data_download.sh  
```

该脚本启动Docker挂载当前目录，并将数据集下载到主机上的`data/`文件夹。

**注意**：数据集为170 GB +，下载时间超过15小时。 用户也可自行下载，请遵循如下目录格式。

- [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/) - `data/squad/v1.1`
- [SQuaD v2.0](https://rajpurkar.github.io/SQuAD-explorer/)  - `data/squad/v2.0`
- [BERT Large](https://ngc.nvidia.com/catalog/models?orderBy=modifiedDESC&query=BERT) - `data/pretrained_models_google/uncased_L-24_H-1024_A-16`
- [BERT Base](https://ngc.nvidia.com/catalog/models?orderBy=modifiedDESC&query=BERT) - `data/pretrained_models_google/uncased_L-12_H-768_A-12`
- [BERT](https://ngc.nvidia.com/catalog/models?orderBy=modifiedDESC&query=BERT) - `data/pretrained_models_google/uncased_L-24_H-1024_A-16`
- [Wikipedia](https://dumps.wikimedia.org/) - `data/wikipedia_corpus/final_tfrecords_sharded`
- [Books Corpus](https://yknzhu.wixsite.com/mbweb) -  `data/bookcorpus/final_tfrecords_sharded`



## BERT镜像运行

构建容器映像并下载数据后，可以按如下方式启动交互式BERT容器： 

```bash
bash scripts/docker/launch.sh
```

`launch.sh`脚本假定用户下载数据和模型后且位于上述规定位置中。



## 预训练过程

BERT旨在预先训练语言表示的深度双向表示。 以下脚本可用于任何语料库上的预训练语言表示。

在BERT容器内，用户可以使用以下脚本运行预训练。
```bash
bash scripts/run_pretraining.sh <train_batch_size_per_gpu> <eval_batch_size> <learning_rate_per_gpu> <precision> <use_xla> <num_gpus> <warmup_steps> <train_steps> <save_checkpoint_steps> <create_logfile>
```

对于FP16 使用XLA，运行：

```bash
bash scripts/run_pretraining.sh 14 8 5e-5 fp16 true 8 5000 2285000 5000 true
```

对于FP32 不使用XLA，运行：

```bash
bash scripts/run_pretraining.sh 6 6 2e-5 fp32 false 8 2000 5333333 5000 true
```



## 微调过程

上述预训练的BERT表示可以通过一个额外的输出层进行微调，以用于最先进的问答系统。 在BERT容器内，用户可以使用以下脚本为SQuaD运行fine-tuning。

```bash
bash scripts/run_squad.sh <batch_size_per_gpu> <learning_rate_per_gpu> <precision> <use_xla> <num_gpus> <seq_length> <doc_stride> <bert_model> <squad_version> <checkpoint> <epochs>
```

对于FP16 使用XLA训练SQuAD 1.1，运行：

```bash
bash scripts/run_squad.sh 10 5e-6 fp16 true 8 384 128 large 1.1  
data/pretrained_models_google/uncased_L-24_H-1024_A-16/bert_model
.ckpt 1.1
```

对于FP32 不使用XLA训练SQuAD 2.0，运行：

```bash
bash scripts/run_squad.sh 5 5e-6 fp32 false 8 384 128 large 1.1 data/pretrained_models_google/uncased_L-24_H-1024_A-16/bert_model.ckpt 2.0
```



## 验证/评估过程

`run_squad_inference.sh`脚本在针对SQuaD微调过程的checkpoint上运行推理，并根据完全匹配和F1得分评估预测的有效性。

```bash
bash scripts/run_squad_inference.sh <init_checkpoint> <batch_size> <precision> <use_xla> <seq_length> <doc_stride> <bert_model> <squad_version>
```

对于FP16 使用XLA推理SQuAD 2.0，运行：
```bash
bash scripts/run_squad_inference.sh /results/model.ckpt 8 fp16 true 384 128 large 2.0
```

对于FP32 不使用XLA推理SQuAD 1.1，运行：
```bash
bash scripts/run_squad_inference.sh /results/model.ckpt 8 fp32 false 384 128 large 1.1
```
