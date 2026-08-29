"""Command-line interface for Mini Coding Agent."""

import argparse
from pathlib import Path
import time
from typing import Callable, Optional, Sequence

from openai import APIError

from mini_agent.agent import AgentResult, CodingAgent, StepLimitError
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
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="任务完成后保留对话并继续接受新需求。",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run one task or start a persistent interactive conversation."""
    args = build_parser().parse_args(argv)
    output = TerminalOutput()

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
            event_handler=output.event,
            text_handler=output.write,
        )
        _print_startup(
            model=settings.model,
            workspace=workspace,
            max_steps=args.max_steps,
            max_context_chars=settings.max_context_chars,
            dangerous_commands=args.allow_dangerous_commands,
        )
    except (ConfigurationError, ValueError) as exc:
        print(f"配置错误：{exc}")
        return 2

    if args.interactive or args.task is None:
        return _run_interactive(agent, output, args.task)

    try:
        _run_task(agent.run, args.task, output)
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
    return 0


class TerminalOutput:
    """Keep streamed model text and structured events readable together."""

    def __init__(self) -> None:
        self._line_open = False
        self._text_seen = False

    @property
    def text_seen(self) -> bool:
        return self._text_seen

    def begin_turn(self) -> None:
        self.finish_line()
        self._text_seen = False

    def write(self, text: str) -> None:
        if not text:
            return
        if not self._line_open:
            print("[回复] ", end="", flush=True)
            self._line_open = True
        print(text, end="", flush=True)
        self._text_seen = True

    def event(self, message: str) -> None:
        self.finish_line()
        print(message)

    def finish_line(self) -> None:
        if self._line_open:
            print()
            self._line_open = False


def _run_task(
    runner: Callable[[str], AgentResult],
    task: str,
    output: TerminalOutput,
) -> None:
    """Run one user turn and print its streamed result and statistics."""
    output.begin_turn()
    started_at = time.monotonic()
    result = runner(task)
    output.finish_line()
    if not output.text_seen:
        print(f"[回复] {result.content}")

    elapsed = time.monotonic() - started_at
    print("[完成] 本轮任务已结束")
    print(
        f"[统计] 模型调用 {result.steps} 次，工具执行 "
        f"{result.tool_calls} 次，耗时 {elapsed:.1f} 秒。"
    )
    if result.compacted_rounds:
        print(f"[统计] 上下文压缩 {result.compacted_rounds} 个较早工具轮次。")
    if result.compacted_turns:
        print(f"[统计] 上下文压缩 {result.compacted_turns} 个较早对话轮次。")


def _run_interactive(
    agent: CodingAgent,
    output: TerminalOutput,
    initial_task: Optional[str],
) -> int:
    """Keep one Agent session alive until the user explicitly exits."""
    print("交互模式已启动。输入 /help 查看命令，输入 /exit 退出。")
    pending_task = initial_task

    while True:
        try:
            task = pending_task if pending_task is not None else input("\n你> ")
            pending_task = None
        except EOFError:
            print("\n会话已结束。")
            return 0
        except KeyboardInterrupt:
            print("\n会话已结束。")
            return 130

        task = task.strip()
        if not task:
            continue
        if task in {"/exit", "/quit"}:
            print("会话已结束。")
            return 0
        if task == "/new":
            agent.reset_session()
            print("已清空对话历史，工作目录和本地文件保持不变。")
            continue
        if task == "/help":
            _print_interactive_help()
            continue

        try:
            _run_task(agent.run_turn, task, output)
        except StepLimitError as exc:
            output.finish_line()
            print(f"本轮未完成：{exc}")
        except APIError as exc:
            output.finish_line()
            print(f"API 请求失败：{exc}")
        except RuntimeError as exc:
            output.finish_line()
            print(f"本轮运行失败：{exc}")
        except KeyboardInterrupt:
            output.finish_line()
            print("本轮已中断，可以继续输入新需求。")


def _print_interactive_help() -> None:
    print("可用命令：")
    print("  /help  显示帮助")
    print("  /new   清空对话历史，但保留工作目录中的文件")
    print("  /exit  结束会话")
    print("  /quit  结束会话")


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
