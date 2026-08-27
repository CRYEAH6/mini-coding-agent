"""Command-line interface for Mini Coding Agent."""

import argparse
from pathlib import Path
import time
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
        workspace = Path(args.workspace).expanduser().resolve()
        tools = ToolRegistry(
            workspace,
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
        _print_startup(
            model=settings.model,
            workspace=workspace,
            max_steps=args.max_steps,
            max_context_chars=settings.max_context_chars,
            dangerous_commands=args.allow_dangerous_commands,
        )
        started_at = time.monotonic()
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

    elapsed = time.monotonic() - started_at
    print("\n[完成] 任务已结束")
    print(result.content)
    print(
        f"\n[统计] 模型调用 {result.steps} 次，工具执行 "
        f"{result.tool_calls} 次，耗时 {elapsed:.1f} 秒。"
    )
    if result.compacted_rounds:
        print(f"[统计] 上下文压缩 {result.compacted_rounds} 个较早工具轮次。")
    return 0


def _print_startup(
    *,
    model: str,
    workspace: Path,
    max_steps: int,
    max_context_chars: int,
    dangerous_commands: bool,
) -> None:
    """Print a compact, secret-free run configuration summary."""
    safety = "高风险命令已允许" if dangerous_commands else "高风险命令默认拦截"
    print("Mini Coding Agent")
    print(f"模型：{model}")
    print(f"工作目录：{workspace}")
    print(f"最大步骤：{max_steps}")
    print(f"上下文预算：{max_context_chars} 字符")
    print(f"安全模式：{safety}\n")
