# Mini Coding Agent

一个使用 Python 从零实现的命令行编程智能体。它通过 DeepSeek 的工具调用能力作出行动决策，并由本地程序完成文件查看、代码修改、命令执行和结果反馈，从而迭代完成编程任务。

本项目目前正在开发中。

## 环境要求

- Python 3.9 或更高版本
- DeepSeek API Key

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

## 当前进度

- 已建立 Python 项目结构；
- 已配置独立虚拟环境和依赖管理；
- 已提供命令行启动入口；
- 已加入基础自动化测试；
- 已实现配置读取和 DeepSeek API 客户端；
- Agent 循环和本地工具正在开发中。
