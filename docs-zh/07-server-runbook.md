# 服务器运行说明：连通检查与费用估算

本文写给在服务器上运行 Claude 相关步骤的人。开发容器里没有 Claude 的访问权限，所以这几步在服务器上跑完，再把结果包发回来分析。

本次只需要跑两步，加一个可选的小试跑：

1. `claude-check`：发一个极小的请求，确认登录、模型和返回格式都正常。
2. `collate --dry-run`：估算全经对勘要发多少请求、多少 token。这一步**不发送任何内容**。
3. （可选）`--pilot`：在 I.1 一品上真实跑一次对勘，约 1–3 次调用，用来实测用量。

这些步骤全部由一个脚本完成：`tools/server_check.sh`。

---

## 0 两种调用方式（后端）

| 后端 | 用什么付费 | 需要什么 |
|---|---|---|
| `subscription`（订阅） | Claude Max/Pro 包月套餐的用量额度，不另外收费 | 安装 Claude Code，并用**订阅账号**登录 |
| `api` | Anthropic Console 预付的 API 额度 | 环境变量 `ANTHROPIC_API_KEY` |

用 20x 包月额度，就选 `subscription`。

**关于账号**：按 Anthropic 的条款，订阅登录只供购买者本人使用。所以服务器上登录的，应当是运行脚本的那个人自己的订阅账号。如果要用别人的套餐，请那个人自己远程登录并运行，不要互相转交登录凭据。

**关于工具**：两种后端是不同的测量工具，工具指纹不同，响应缓存也分开存放。整个研究从开发集到最终报告应当只用同一种后端。

---

## 1 一次性准备

```bash
git clone https://github.com/HymnOfLight/hevajra_tantra_translate.git
cd hevajra_tantra_translate
git checkout claude/brave-bell-7rb7lh

python3 -m venv .venv && . .venv/bin/activate      # Python 3.10 或更高
pip install -e ".[dev]"                             # 用 api 后端时改为 ".[dev,claude]"
```

**订阅后端**还需要安装 Claude Code 并登录：

```bash
npm install -g @anthropic-ai/claude-code    # 或按官方文档的其他安装方式
claude                                      # 进入后输入 /login，用订阅账号登录，完成后 /exit 退出
```

如果服务器没有浏览器、不方便交互登录，可以在自己的电脑上运行 `claude setup-token` 生成令牌，再在服务器上设置：

```bash
export CLAUDE_CODE_OAUTH_TOKEN=...   # 只放环境变量，不要写进文件，也不要发给别人
```

**注意**：服务器上如果设置了 `ANTHROPIC_API_KEY`，订阅后端会自动忽略它，不会用它扣费。

---

## 2 运行

```bash
bash tools/server_check.sh subscription           # 连通检查＋费用估算
bash tools/server_check.sh subscription --pilot   # 另加 I.1 一品的真实小试跑
```

脚本依次做这几件事：

1. 下载原文（`data/raw/` 已有就跳过）；
2. 切分；
3. 连通检查；
4. 费用估算；
5. 可选的试跑。

最后打印一行：

```
packed: .../server-check-<时间>.tar.gz  (send this file back)
```

**请把这个 `.tar.gz` 文件发回来。** 它里面只有：

- 检查结果 `claude_check.json`；
- 费用估算 `collation_dry_run.json`；
- 调用审计日志 `llm_audit.jsonl`：只有 token 数、模型名、状态，不含任何提示或回答的文字；
- 运行清单 `manifest.json`，以及切分完整性结果 `ingest_g0.json`；
- 各步骤的屏幕输出（`*.log`）和环境信息 `environment.txt`：Python、Claude Code、SDK 的版本，以及是否设置了密钥（只记"是/否"）。

结果包里没有受许可限制的原文，也没有密钥或登录令牌。运行目录 `runs/server-check-<时间>/` 和响应缓存 `runs/llm-cache/` 保留在服务器上，不需要发回。

如果启用了 `--pilot`，试跑的结果会留在服务器上。之后如需进一步分析，再单独约定要发回哪些文件。

---

## 3 怎么看结果

`claude_check.json` 中：

- `"passed": true`：一切正常。
- `"served_model"` 应为 `claude-opus-5-5`；`"fallback_used"` 应为 `false`。
- `"claude_code_version"`：服务器上 Claude Code 的版本。

费用估算在 `dry_run.log` 的最后一行，例如：

```
collate --dry-run: 105 calls (0 cached), 3786924 input tokens (heuristic), assumed 30000 output tokens per call
  -> projected $64.63 (API-equivalent; the subscription bills nothing but uses its allowance); nothing was sent
```

用订阅后端时，这个美元数只是"按 API 价格折算"的参考，并不实际扣费；真正消耗的是套餐的用量额度。额度用了多少，可以在 Claude Code 里输入 `/usage` 查看。试跑前后各看一次，就能知道一品大约占多少额度。

---

## 4 常见问题

| 现象 | 原因与处理 |
|---|---|
| `claude-check needs the claude command` | 没装 Claude Code，或 `claude` 不在 PATH 中 |
| `claude -p failed ... login` 或类似认证错误 | 没登录：运行 `claude` 后 `/login`，或设置 `CLAUDE_CODE_OAUTH_TOKEN` |
| `UsageLimitReached`（"You've hit your session limit" 之类） | 套餐的用量额度暂时用完了。已完成的调用都在缓存里，等额度重置后**重新运行同一条命令**，会从停下的地方继续 |
| `CliTimeout` | 单次调用超过 1 小时被停止（`config/llm.yaml` 的 `subscription.timeout_s`）。不会自动重试；把结果包发回，我们一起判断是否需要调大 |
| `... was produced with the api backend` | 这个运行目录之前是用另一种后端跑的。换一个新的运行目录，或者改用原来的后端 |
| `passed: false`，`status: invalid` | 没有返回结构化结果，可能与 Claude Code 版本有关。把结果包原样发回，我们来调整 `config/llm.yaml` 的 `subscription` 设置 |
| `fallback_used: true` | 实际回答的不是 Opus 5.5，这样的回答不会进入测量。请把结果包发回 |
| 下载原文失败 | 服务器访问 CBETA 或 GitHub 受限，可以手动把 `T18n0892.xml` 与 `derge_rgyud_bum_nga.txt` 放进 `data/raw/` |

---

## 5 以后的正式运行

连通检查通过、费用和额度都确认之后，正式对勘也在服务器上进行，在每条命令后加 `--backend subscription`，例如：

```bash
hevajra-matrix collate --backend subscription
```

也可以把 `config/llm.yaml` 中的 `backend: api` 改成 `backend: subscription`，这样就不用每次加参数。

正式运行前，还要先完成开发集的人工盲标与预注册冻结，见 [`00-overview.md`](00-overview.md) §21。
