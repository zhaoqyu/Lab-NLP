# 2026-07-10 项目进展汇报稿

> **数据角色更正（2026-08-27）：** 当前 pipeline 使用 KVS 的 `train/eval`
> 训练 DPO/HyPO，只使用 AITA 做跨数据集价值观变化测试。以下内容保留为历史会议记录，
> 实际命令与说明请以 `value_alignment/README.md` 为准。

## 一句话总结

我们现在有了一个比较完整的 value alignment 实验 pipeline：AITA 用来做 DPO/HyPO preference training 和行为评估，KVS 用来做 survey-style intrinsic evaluation，训练部分调用官方 HyPO implementation，并且模型可以在 Qwen、Mistral、Llama 之间切换。

## 已完成模块

### 1. Data Pipeline

- 已实现 AITA 到 DPO/HyPO 数据格式转换：
  - `prompt`
  - `chosen`
  - `rejected`
  - `value`
  - `high_standard_stance`
  - `low_standard_stance`
- 支持按 value 过滤。
- 支持 train/eval/test split。
- 本地已经生成过 4 个主 value 的 split：
  - `Security_personal`
  - `Benevolence_caring`
  - `Universalism_concern`
  - `Self_direction_action`
- 当前 split 大小：
  - train: 1600
  - eval: 200
  - test: 200
- validator 检查通过，0 errors。

### 2. Synthetic Data Generation

- 已实现 teacher LLM synthetic preference generation 脚本。
- 支持：
  - diverse value-profile personas
  - target value conditioning
  - OpenAI-compatible API
  - local endpoint via `OPENAI_BASE_URL`
  - dry-run 模式
- 我们不保存长 hidden CoT，而是保存 concise public rationale。
- 这样更适合报告中表述为 rationale generation，而不是暴露 chain-of-thought。

### 3. Model & Training

- 训练脚本使用官方 HyPO repo 的 `DPOTrainer` 和 `DPOConfig`。
- 我们自己的代码只做 wrapper，不重写核心 HyPO 算法。
- 支持：
  - standard DPO
  - HyPO / Hybrid-DPO
  - LoRA
  - config file
  - model aliases
- 可切换模型：
  - `qwen2.5-7b`
  - `mistral-7b`
  - `llama3.1-8b`

### 4. Intrinsic Evaluation: KVS

- 已实现 KVS survey prompt conversion。
- 已实现模型 1-6 rating evaluation。
- 已实现 base vs trained comparison：
  - target value mean shift
  - per-value mean/std
  - other values' variance

### 5. Extrinsic Evaluation: AITA

- 已实现 AITA Probability Gain evaluation。
- 对 NTA / YTA / Neutral 三个 label 计算概率分布。
- 比较训练前后模型对 high-standard stance 的概率变化。
- 输出：
  - per-example result
  - per-value accuracy
  - probability gain

### 6. MACHIAVELLI

- 已添加 MACHIAVELLI setup notes。
- 已添加 Hugging Face model agent template。
- 这个部分比较重，需要单独下载 benchmark game data，所以目前作为 optional transfer evaluation。

## 我可以在 meeting 里这样说

> We now have an end-to-end modular pipeline. AITA is used for preference-pair training and behavioral evaluation, while KVS is used for intrinsic survey-style value evaluation. For training, we call the official HyPO implementation instead of reimplementing the algorithm. The model is configurable, so we can switch between Qwen2.5-7B-Instruct, Mistral-7B-Instruct, and Llama-3.1-8B-Instruct. I also added synthetic teacher generation with value-profile personas and concise public rationales, plus validators to check label consistency before training.

中文版本：

> 我们现在已经有一个端到端的模块化 pipeline。AITA 用于 preference-pair training 和 behavioral evaluation，KVS 用于 intrinsic survey-style value evaluation。训练部分调用官方 HyPO implementation，不自己重写核心算法。模型名称通过配置或 CLI 传入，可以切换 Qwen、Mistral 和 Llama。synthetic generation 部分也已经有初版，支持 teacher LLM、不同 value-profile persona 和 concise public rationale，并且有 validator 检查数据质量。

## 还需要 GPU/Cluster 才能完成的部分

- 下载并运行 7B 模型。
- 跑 DPO / HyPO training。
- 跑完整 KVS evaluation。
- 跑完整 AITA probability gain evaluation。
- MACHIAVELLI benchmark 的完整运行。

## 下一步建议

1. 先用 `qwen2.5-7b` 跑一轮 smoke-test training。
2. 对比 base vs DPO vs HyPO 的 KVS intrinsic score。
3. 对比 base vs DPO vs HyPO 的 AITA probability gain。
4. 如果时间允许，再扩展到 Mistral 和 Llama。
5. 最后把结果整理成 presentation 的三张核心表/图：
   - KVS target value shift
   - other values' variance
   - AITA probability gain
