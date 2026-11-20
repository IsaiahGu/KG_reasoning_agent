# 多条推导按可靠度合并的多跳问答智能体

这个目录只保留最终方法需要的代码、训练好的模型和论文用的结果记录。早期阶段的代码和已删除的超图模块都打包在 `archive/` 里，可以按路径恢复。

- 结果、消融和与已发表方法的对照：[paper/RESULTS.md](paper/RESULTS.md)
- 方法设计和分析过程：[DESIGN_V2.md](DESIGN_V2.md)
- 和原计划不一致的地方（按时间记录）：[DEVIATIONS.md](DEVIATIONS.md)
- 已发表结果的出处和设置：[RELATED_RESULTS.md](RELATED_RESULTS.md)

项目早期叫 HyperBind（源码包名 `hyperbind` 沿用至今），超图模块删除后方法里已经没有超图，写论文时需要另起名字。

## 方法

对每个问题跑四条互相独立的推导，再用一个小的可靠度模型决定采信哪个答案。读者、规划和绑定都用同一个 Qwen3.5-9B，检索用 NV-Embed-v2，模型权重都不训练。

| 推导 | 做什么 | 运行标签 |
|---|---|---|
| 稠密直读 | 用问题检索 5 段，直接读出答案（1 次调用） | `a0` |
| 逐跳回路 A1 | 把问题拆成几跳；每一跳检索、绑定一个带原文引用的答案，再代入下一跳；最后读一遍收集到的证据 | `single` |
| 记忆规划第 1 轮 | 同一个回路，拆解时用“自建规划记忆”里最相似的训练题的拆法作示例。记忆来自 A1 在训练题上答对的拆解，不用拆解标注 | `selfmem_answer` |
| 记忆规划第 2 轮 | 同上，记忆再加入第 1 轮在训练题上答对的拆解 | `selfmem2_answer` |

- **可靠度模型**：17 个不需要标准答案就能算的特征（回路是否走完、引用是否逐字出现在原文、检索分数、拆解形状等）上的逻辑回归，每个数据集用 1,500 道训练题拟合。稠密直读用一个固定的先验值。
- **合并**：答案相同的推导归为一组，组的支持度是“至少一条推导可靠”的概率（noisy-OR），取支持度最高的一组。
- **按需展开**：先跑稠密直读和 A1，两者答案一致就停，不一致才跑两条记忆规划。
- **迁移设置**：NQ 和 PopQA 没有训练题，可靠度模型和规划记忆都用 HotpotQA 的。

早先方法里还有第五条推导“超图检索回路”，在四个测试集上都没有增益，已从方法和代码里删除（`configs/final.json`）。跑测试集之前预先登记的配置在 `configs/frozen_v2.json`，最终方法就是其中预先登记的“去掉超图检索回路”那一行，两行结果都报告。

## 目录

| 路径 | 内容 |
|---|---|
| `src/hyperbind/` | `chain_agent.py`（逐跳回路、规划记忆、稠密检索）、`support.py`（可靠度模型和合并）、`baseline_dense.py`（稠密直读和计分）、`data.py`、`splits.py`、`llm.py`、`textmatch.py`、`binding_scorer.py`（跳级特征）、向量和嵌入服务相关模块；同读者基线：`baseline_ircot.py`（IRCoT）、`baseline_hipporag.py` 及 `hipporag_*.py`（官方 HippoRAG 2 的封装） |
| `scripts/` | 运行和出表的入口，见下面“复现” |
| `tests/` | 保留模块的单元测试 |
| `configs/` | `final.json`（最终方法）、`frozen_v2.json`（预先登记的配置）、`support/`（可靠度模型）、`queue_v1.json` 和 `queue_v2.json`（实际跑过的全部命令）、划分设置、环境锁定文件 |
| `paper/` | 论文用的结果记录 `RESULTS.md`、表格 `tables/`、逐题原始数据 `raw/`、全部数字 `results.json` |
| `reports/` | 脚本生成的报告，以及模型和数据的来源记录（代码运行时会读其中几个文件） |
| `probe/` | 分析用的只读脚本（开发集和测试集的差距、错误根源、超图检索是否互补） |
| `preparation/` | 下载数据和模型、建环境的脚本 |
| `runs/`（服务器） | 全部运行记录 `runs/chain/`、规划记忆 `runs/plan_memory/`、向量、大模型调用缓存、队列日志 |
| `models/`、`envs/`、`data/`（服务器） | 基础模型、三个 Python 环境、数据集 |
| `archive/` | `legacy_2026-11-18.tar.gz`（早期阶段的代码、测试、配置、笔记）、`hypergraph_2026-11-18.tar.gz`（超图模块的代码、测试、配置）；服务器上另有 `legacy_runs_2026-11-18.tar.gz`（早期阶段的小运行目录） |

## 训练好的模型

| 模型 | 位置 | 大小 |
|---|---|---|
| 可靠度模型（MuSiQue、2Wiki、HotpotQA 各一个） | `configs/support/<数据集>_chain_v1.json` | 各约 1.4 KB |
| 自建规划记忆（每个数据集两轮） | `runs/plan_memory/<数据集>_self_answer.json`、`..._self_answer2.json` | 各约 0.2 MB |
| 跳级打分器（只用于一个被弃用的变体） | `configs/scorers/musique_l2_1500.json` | 2 KB |

基础模型只在服务器上：Qwen3.5-9B（规划、绑定和阅读）、NV-Embed-v2（向量）。

## 复现

都在服务器的项目根目录下运行，解释器是 `envs/main/bin/python`。需要大模型的步骤由队列统一调度，队列会自己启动和关闭模型服务。

| 步骤 | 命令 |
|---|---|
| 跑队列（断点续跑） | 删除 `runs/queue/STOP` 后 `supervisorctl start hyperbind_queue`，状态看 `runs/queue/STATUS.txt` |
| 单独跑一条推导 | `scripts/run_chain.py --dataset musique --split test --tag single --candidates 1` |
| 拟合可靠度模型 | `scripts/fit_support_model.py --dataset <数据集> --tags single_fit --out configs/support/<数据集>_chain_v1.json` |
| 建规划记忆 | `scripts/build_plan_memory.py --dataset <数据集> --admit answer --tags single_fit --out runs/plan_memory/<数据集>_self_answer.json` |
| 自洽性对照的一次采样 | `scripts/run_chain.py --dataset musique --split test --tag sc07_1 --candidates 1 --temperature 0.7 --seed 1` |
| 同读者 IRCoT（测试集） | `scripts/run_ircot_test.py --dataset musique` |
| 同读者 HippoRAG 2（测试集） | `envs/embed/bin/python -m hyperbind.baseline_hipporag --split test --approved --workers 16 --phase openie --dataset musique`，之后依次 `encode-index`（需先停掉模型服务）、`build`、`run`，最后 `scripts/score_hipporag_test.py --dataset musique` |
| 预先登记配置的结果表 | `PYTHONPATH=src envs/main/bin/python -B scripts/test_report.py` |
| 论文表格和逐题数据 | `PYTHONPATH=src envs/main/bin/python -B scripts/paper_results.py` |
| 单元测试 | `PYTHONPATH=src envs/main/bin/python -m pytest tests -q --ignore=tests/test_hipporag_preflight.py`；HippoRAG 2 的预检测试要用嵌入环境：`PYTHONPATH=src:tests envs/embed/bin/python -m unittest tests.test_hipporag_preflight` |
| 看一组任务的状态 | `python3 scripts/queue_state.py <任务名> ...` |

每个任务的完整命令行在 `configs/queue_v2.json` 里。其中建超图索引和超图回路的任务已经跑完，脚本随超图模块归档了，要重跑得先从归档里取回。2026-11-18 晚追加了论文需要的补充实验（同读者基线、自洽性对照、回路设计消融），见 [DEVIATIONS.md](DEVIATIONS.md) 末尾。

## 从归档里取回文件

```bash
tar -tzf archive/hypergraph_2026-11-18.tar.gz
```

```bash
tar -xzf archive/hypergraph_2026-11-18.tar.gz src/hyperbind/hyper_retriever.py
```

`legacy_2026-11-18.tar.gz` 里有早期 P0–P2 阶段的代码和测试，重排器和 Qwen 向量的适配代码，基线的修复工具，早期笔记和评审意见。同读者基线（HippoRAG 2、IRCoT）的实现已经取回到 `src/hyperbind/`。超图索引的数据已经删除，取回代码后用 `scripts/build_test_graph_index.py` 可以重建（需要重新下载 GLiNER large-v2.1）。
