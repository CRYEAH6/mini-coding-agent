"""Command-line interface for Mini Coding Agent."""

import argparse
from pathlib import Path
from typing import Optional, Sequence

from openai import APIError

from mini_agent.agent import CodingAgent, StepLimitError
from mini_agent.client import DeepSeekClient
from mini_agent.config import ConfigurationError, Settings
from mini_agent.tools import ToolRegistry


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="mini-agent",
        description="使用 DeepSeek 和本地工具完成编程任务。",
    )
    parser.add_argument("task", nargs="?", help="需要 Agent 完成的编程任务。")
    parser.add_argument(
        "--workspace",
        default=".",
        help="允许 Agent 操作的工作目录，默认为当前目录。",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="最大模型调用轮数，默认为 20。",
    )
    parser.add_argument(
        "--allow-dangerous-commands",
        action="store_true",
        help="允许绕过高风险命令拦截；仅在可信环境中使用。",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run one coding task and return a process exit code."""
    args = build_parser().parse_args(argv)
    task = args.task or input("请输入编程任务：").strip()

    try:
        settings = Settings.from_env()
        tools = ToolRegistry(
            Path(args.workspace),
            allow_dangerous_commands=args.allow_dangerous_commands,
        )
        client = DeepSeekClient(settings)
        agent = CodingAgent(
            client,
            tools,
            max_steps=args.max_steps,
            max_context_chars=settings.max_context_chars,
            event_handler=print,
        )
        result = agent.run(task)
    except (ConfigurationError, ValueError) as exc:
        print(f"配置错误：{exc}")
        return 2
    except StepLimitError as exc:
        print(f"任务未完成：{exc}")
        return 3
    except APIError as exc:
        print(f"API 请求失败：{exc}")
        return 4
    except RuntimeError as exc:
        print(f"Agent 运行失败：{exc}")
        return 5
    except KeyboardInterrupt:
        print("\n任务已由用户中断。")
        return 130

    print("\n任务完成：")
    print(result.content)
    print(f"\n共调用模型 {result.steps} 次，执行工具 {result.tool_calls} 次。")
    if result.compacted_rounds:
        print(f"上下文压缩了 {result.compacted_rounds} 个较早工具轮次。")
    return 0
