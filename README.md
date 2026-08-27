# Mini Coding Agent

一个使用 Python 从零实现的命令行编程智能体。它通过 DeepSeek 的工具调用能力作出行动决策，并由本地程序完成文件查看、代码修改、命令执行和结果反馈，从而迭代完成编程任务。

本项目目前正在开发中。

## 环境要求

- Python 3.9 或更高版本
- DeepSeek API Key

## 安装

项目使用 `uv` 管理虚拟环境和依赖：

```bash
uv sync --extra dev
```

## 凭据安全

API Key 必须通过 `DEEPSEEK_API_KEY` 环境变量提供。不要把真实 API Key 写入代码或提交到仓库。

首次运行前复制环境变量示例文件：

```bash
cp .env.example .env
```

然后只在本地 `.env` 中填写真实的 `DEEPSEEK_API_KEY`。`.env` 已被 Git 忽略，不会上传到仓库。

可以通过以下可选变量调整模型连接参数：

- `DEEPSEEK_MODEL`：模型名称；
- `DEEPSEEK_BASE_URL`：API 接口地址；
- `DEEPSEEK_TIMEOUT_SECONDS`：单次请求超时时间；
- `DEEPSEEK_MAX_RETRIES`：API 客户端最大重试次数。
- `DEEPSEEK_MAX_CONTEXT_CHARS`：消息历史的近似字符预算，默认 200000。

## 运行

在需要 Agent 操作的项目目录中运行：

```bash
uv run --project /path/to/mini-coding-agent mini-agent \
  --workspace . \
  "检查项目并完成指定的编程任务"
```

也可以先进入本仓库，再明确指定目标工作目录：

```bash
uv run mini-agent --workspace /path/to/target-project "修复失败的测试"
```

使用 `--max-steps` 可以限制单次任务的最大模型调用轮数，默认值为 20。

高风险命令默认会被拦截。如果确实需要让 Agent 执行删除、强制 Git 操作等命令，可以显式添加：

```bash
--allow-dangerous-commands
```

该选项会绕过内置命令规则，只应在隔离且可恢复的可信工作目录中使用。

启动后会先显示不含凭据的运行配置，并使用统一标签展示过程：

```text
Mini Coding Agent
模型：deepseek-v4-flash
工作目录：/path/to/target-project
最大步骤：20
上下文预算：200000 字符
安全模式：高风险命令默认拦截

[模型 1/20] 正在生成下一步...
[工具] read_file（path=src/app.py）
[结果] 成功
```

工具状态只展示必要信息。写文件时不打印内容，执行命令时只显示程序名称而不回显完整参数，降低凭据意外出现在终端或视频中的风险。

## 核心流程

1. 命令行接收用户任务和工作目录；
2. Agent 将任务、历史消息和工具定义发送给 DeepSeek；
3. 模型返回工具调用或最终回答；
4. 本地程序校验参数并执行文件或命令工具；
5. 工具结果加入对话历史，再次请求模型；
6. 模型给出最终回答或达到终止条件后结束。

除最大模型轮数外，Agent 还会检测无效循环：第 3 次完全相同的工具调用会被拦截并提示模型调整方案，继续重复则终止；工具连续失败 3 次会警告，达到 4 次则终止。任意成功工具调用都会重置连续失败计数。

每次模型请求前，Agent 会估算消息历史的 JSON 字符数。超过预算时，只压缩最早的完整工具轮次，保留 system 指令、原始用户任务、最近两个工具轮次，以及包含工具名称、关键参数和成功状态的历史摘要。该方法不额外调用模型，也不会拆散 Tool Call 和对应 Tool Result。

当前提供五个本地工具：

- `list_files`：列出目录内容；
- `read_file`：读取 UTF-8 文本文件；
- `write_file`：创建或完整写入文件；
- `replace_in_file`：精确替换已有文本；
- `run_command`：执行带超时和输出限制的 zsh 命令。

## 安全边界

文件工具会解析真实路径，并拒绝绝对路径、`..` 越界和指向工作目录外部的符号链接。

命令工具默认阻止 `sudo`、`rm`、磁盘管理、关机、`git reset --hard`、强制 Git 清理或推送、`find -delete`、下载脚本后直接执行等明显高风险操作。此外，命令固定在工作目录启动，并受到超时和输出长度限制。

这些规则属于保守的启发式防护，并不是容器或操作系统级沙箱。复杂脚本、解释器代码、别名等仍可能绕过规则。运行 Agent 前应使用可恢复的项目副本，并确保工作目录中没有敏感文件。

## 测试

运行本地自动化测试：

```bash
uv run pytest -q
```

普通测试使用假模型响应，不需要 API Key，也不会消耗 API 额度。真实 API 会作为单独的端到端测试执行。

## 当前进度

- 已建立 Python 项目结构；
- 已配置独立虚拟环境和依赖管理；
- 已提供命令行启动入口；
- 已加入基础自动化测试；
- 已实现配置读取和 DeepSeek API 客户端；
- 已实现文件工具、命令工具和统一工具注册；
- 已实现 Agent 主循环和命令行入口；
- 已通过 DeepSeek API 真实端到端验证：创建 Python 文件、执行并检查输出。
