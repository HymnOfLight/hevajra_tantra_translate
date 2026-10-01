# Claude Opus 5.5 使用说明

本仓库通过 Anthropic 官方 Python SDK 调用 `claude-opus-5-5`。技术细节见英文文档 [`docs/llm-tasks.md`](../docs/llm-tasks.md)。

---

## 1 安装与密钥

```bash
pip install -e ".[dev,claude]"          # claude 附加依赖即 anthropic SDK
export ANTHROPIC_API_KEY=sk-ant-...      # 不要把密钥写进任何文件或提交
hevajra-matrix claude-check              # 发一个极小的请求，打印实际服务的模型、用量和请求 id
```

没有安装 SDK、没有密钥也能使用全部不调用 Claude 的功能（切分、基线、复核表格、统计、报告），以及用已有缓存离线重放（`--offline`）。

## 2 先估算费用，再真正调用

```bash
hevajra-matrix ingest
hevajra-matrix collate --dry-run         # 有密钥时用官方 token 计数接口，没有时按字符估算；不发送任何内容
hevajra-matrix collate --chapters I.1 --replicates 1    # 建议先在一品上实测藏文的 token 消耗
hevajra-matrix collate                   # 全经，每品 3 次重复
```

参考：全经一次对勘约 35 次调用；3 次重复约 105 次调用。按字符估算约 $65；完整周期（含扰动测试）约 $80–140。`config/llm.yaml` 的 `budget_usd: 300` 是硬上限，超过后会拒绝新的、未缓存的调用。

## 3 参数（`config/llm.yaml`）

- 每个任务有自己的 `effort`（推理投入：low/medium/high/xhigh/max）、`max_tokens`、重复次数与 `fallback`。
- 该模型**思考恒开、不接受温度等采样参数**，所以同一请求可能每次结果不同。本仓库的对策：
  1. **响应缓存**（`runs/llm-cache/`）：任何跑过的结果都能 `--offline` 逐字节重放，换台机器、没有密钥也能复算；
  2. **重复**：对勘每品跑 3 次，投票成共识，重复间一致性按"人与人一致性"同样报告；
  3. **工具指纹**：提示词、schema、模型、effort、参数任何一项变化，都视为一个新工具，必须重新过门控。
- 缓存里含有受许可限制的原文，**永远不要提交**；需要让别人复算时请私下存档（如受限访问的 Zenodo）。审计日志 `llm_audit.jsonl` 只记哈希、用量和状态，不含任何原文。

## 4 拒绝与模型替换

- 如果安全分类器拒绝某个请求（`stop_reason == "refusal"`），该窗口的单元记为 `UNALIGNED(refused:<类别>)`（API 未给类别时为 `refused:unspecified`；组件与主题任务用同样的写法），排到人工复核最前面，**不会自动重试或拆分**；需要时可以改小窗口重跑（这会产生新的缓存键）。
- 按您的决定：对勘、组件、主题这类流水线任务开启服务器端 fallback，实验关闭。由于 fallback 对同一内容约一小时内是"粘性"的，凡是实际服务模型不是 `claude-opus-5-5` 的响应，一律标为 `substituted_model`，其单元记为 `UNALIGNED(substituted_model)`；替代模型的草案只在解决表（`resolve`）的 `hint_other_model` 列里作提示，**永不进入测量**；只要构成共识的对勘请求中有一次被替换，门控 G2 就不通过（扰动测试等其他调用不计入）。若希望严格到底，可以把 `config/llm.yaml` 里相应任务的 `fallback` 改为 `false`。
- 拒绝率按任务和主题分别报告。如果敏感段落被拒得更多，这本身就是需要写进论文的发现。

## 5 冻结预注册

在第一次用测试金标准评估 Claude 之前：

```bash
hevajra-matrix prereg freeze             # 把当前五个任务的工具指纹写入 config/preregistration.yaml 并标为已冻结
hevajra-matrix prereg freeze --amend "修改原因"   # 冻结后的任何改动都必须追加修订记录
```

测试集每被评估一次，都会追加到 `data/ledger/test_evaluations.jsonl`；报告会写明"测试集已被 K 个工具版本评过 N 次"。对同一工具指纹、同一预注册、同一得分与门控结果的**重复评分不会追加**（例如复核之后为刷新 G2/G3 再跑 `evaluate`，并没有重新评估测试集），程序会打印"already recorded; not appended again"。按主题组拆分的拒绝率不参与这一比较，因为主题标签通常在测试评分之后才导入。

## 6 Claude 在这里不做什么

- 不给出任何动机归因（"审查""避讳"之类）：那是过度归因实验研究的对象。
- 看不到梵文、大正藏脚注和主题标签（对勘时）；主题预标看不到汉文；实验评分器看不到实验条件。
- 提示词从不要求模型展示或解释它的推理过程。
