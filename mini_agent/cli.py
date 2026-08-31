"""Command-line interface for Mini Coding Agent."""

import argparse
from pathlib import Path
import time
from typing import Callable, Optional, Sequence

from openai import APIError

from mini_agent.agent import AgentResult, CodingAgent, StepLimitError
from mini_agent.client import DeepSeekClient
from mini_agent.config import ConfigurationError, Settings
from mini_agent.memory import CATEGORY_LABELS, MemoryError, MemoryStore
from mini_agent.session import (
    OpenedSession,
    SessionError,
    SessionRecord,
    SessionStore,
)
from mini_agent.summary import MAX_SUMMARY_CHARS, SemanticSummarizer
from mini_agent.tools import ToolRegistry
from mini_agent.tools.approval import ApprovalRequest
from mini_agent.tools.sandbox import SANDBOX_MODES, STRICT_MODE


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="mini-agent",
        description="使用 DeepSeek 和本地工具完成编程任务。",
    )
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
        help="允许高风险命令进入人工确认流程；仅在可信环境中使用。",
    )
    parser.add_argument(
        "--sandbox-mode",
        choices=SANDBOX_MODES,
        default=STRICT_MODE,
        help="命令隔离模式：strict 为系统沙箱，policy 仅使用策略检查。",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Start one persistent interactive conversation."""
    args = build_parser().parse_args(argv)
    output = TerminalOutput()

    try:
        settings = Settings.from_env()
        workspace = Path(args.workspace).expanduser().resolve()
        tools = ToolRegistry(
            workspace,
            allow_dangerous_commands=args.allow_dangerous_commands,
            approval_handler=_confirm_tool_action,
            sandbox_mode=args.sandbox_mode,
        )
        client = DeepSeekClient(settings)
        summarizer = SemanticSummarizer(
            client,
            settings.effective_summary_model,
            max_summary_chars=min(
                MAX_SUMMARY_CHARS,
                max(200, settings.max_context_chars // 5),
            ),
            event_handler=output.event,
        )
        memory_store = MemoryStore(workspace)
        agent = CodingAgent(
            client,
            tools,
            max_steps=args.max_steps,
            max_context_chars=settings.max_context_chars,
            event_handler=output.event,
            text_handler=output.write,
            memory_store=memory_store,
            summary_generator=summarizer.summarize,
        )
        session_store = SessionStore(workspace)
        opened_session = session_store.open_active()
        current_session = opened_session.record
        if current_session.messages:
            agent.restore_session(
                current_session.messages,
                current_session.context_state,
            )
        _print_startup(
            model=settings.model,
            workspace=workspace,
            max_steps=args.max_steps,
            max_context_chars=settings.max_context_chars,
            dangerous_commands=args.allow_dangerous_commands,
            sandbox_mode=args.sandbox_mode,
            memory_count=len(memory_store.list_memories()),
            summary_model=settings.effective_summary_model,
        )
        _print_opened_session(opened_session)
    except (ConfigurationError, SessionError, MemoryError, ValueError) as exc:
        print(f"配置错误：{exc}")
        return 2

    return _run_interactive(
        agent,
        output,
        session_store,
        current_session,
        memory_store,
    )


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
    if result.semantic_summaries:
        print(f"[统计] 语义摘要模型调用 {result.semantic_summaries} 次。")
    if result.summary_fallbacks:
        print(f"[统计] 语义摘要回退 {result.summary_fallbacks} 次。")


def _run_interactive(
    agent: CodingAgent,
    output: TerminalOutput,
    session_store: SessionStore,
    current_session: SessionRecord,
    memory_store: MemoryStore,
) -> int:
    """Keep one Agent session alive until the user explicitly exits."""
    print("交互模式已启动。输入 /help 查看命令，输入 /exit 退出。")

    while True:
        try:
            task = input("\n你> ")
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
            saved = _save_current_session(
                session_store,
                current_session,
                agent,
            )
            if saved is None:
                continue
            try:
                current_session = session_store.create()
            except SessionError as exc:
                print(f"新建会话失败：{exc}")
                continue
            agent.reset_session()
            print(f"已创建新会话：{current_session.session_id}")
            print("旧会话和工作目录中的文件保持不变。")
            continue
        if task == "/sessions":
            _print_sessions(session_store, current_session.session_id)
            continue
        if task == "/memories":
            _print_memories(memory_store)
            continue
        if task == "/remember" or task.startswith("/remember "):
            content = _command_argument(task)
            if content is None:
                print("用法：/remember 要长期记住的内容")
                continue
            try:
                record, created = memory_store.remember(content)
                action = "已保存" if created else "已存在，已刷新"
                print(f"{action}长期记忆：{record.memory_id}")
            except (MemoryError, ValueError) as exc:
                print(f"保存长期记忆失败：{exc}")
            continue
        if task == "/forget" or task.startswith("/forget "):
            memory_id = _command_argument(task)
            if memory_id is None:
                print("用法：/forget 记忆ID")
                continue
            try:
                removed = memory_store.forget(memory_id)
                print(f"已删除长期记忆：{removed.memory_id}")
            except MemoryError as exc:
                print(f"删除长期记忆失败：{exc}")
            continue
        if task == "/help":
            _print_interactive_help()
            continue
        if task == "/switch" or task.startswith("/switch "):
            session_id = _command_argument(task)
            if session_id is None:
                print("用法：/switch 会话ID")
                continue
            saved = _save_current_session(
                session_store,
                current_session,
                agent,
            )
            if saved is None:
                continue
            try:
                target = session_store.load(session_id)
                agent.restore_session(target.messages, target.context_state)
                session_store.set_active(target.session_id)
                current_session = target
                print(f"已切换到会话：{target.session_id}")
                print(f"标题：{target.title}（{target.turn_count} 轮）")
            except RuntimeError as exc:
                print(f"切换会话失败：{exc}")
            continue
        if task == "/delete" or task.startswith("/delete "):
            session_id = _command_argument(task)
            if session_id is None:
                print("用法：/delete 会话ID")
                continue
            try:
                target = session_store.load(session_id)
            except SessionError as exc:
                print(f"删除会话失败：{exc}")
                continue
            request = ApprovalRequest(
                tool_name="delete_session",
                action="删除本地对话会话",
                details=(
                    f"会话：{target.session_id}\n"
                    f"标题：{target.title}\n"
                    f"对话轮数：{target.turn_count}"
                ),
                reason="删除后该会话的聊天历史无法由 Agent 恢复。",
            )
            if not _confirm_tool_action(request):
                continue
            if target.session_id == current_session.session_id:
                try:
                    replacement = session_store.create()
                    agent.reset_session()
                    current_session = replacement
                except SessionError as exc:
                    print(f"删除会话失败：无法建立替代会话：{exc}")
                    continue
                try:
                    session_store.delete(target.session_id)
                    print(
                        "当前会话已删除，已创建新会话："
                        f"{replacement.session_id}"
                    )
                except SessionError as exc:
                    print(
                        "已切换到新会话，但旧会话删除失败："
                        f"{exc}"
                    )
            else:
                try:
                    session_store.delete(target.session_id)
                    print(f"已删除会话：{target.session_id}")
                except SessionError as exc:
                    print(f"删除会话失败：{exc}")
            continue
        if task.startswith("/"):
            print("未知命令。输入 /help 查看可用命令。")
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
        finally:
            saved = _save_current_session(
                session_store,
                current_session,
                agent,
            )
            if saved is not None:
                current_session = saved


def _print_interactive_help() -> None:
    print("可用命令：")
    print("  /help  显示帮助")
    print("  /new   创建新会话，保留旧会话和工作目录文件")
    print("  /sessions  查看当前工作目录的历史会话")
    print("  /switch 会话ID  切换到指定会话")
    print("  /delete 会话ID  确认后删除指定会话")
    print("  /remember 内容  手动保存一条长期记忆")
    print("  /memories  查看当前工作目录的长期记忆")
    print("  /forget 记忆ID  删除一条长期记忆")
    print("  /exit  结束会话")
    print("  /quit  结束会话")


def _save_current_session(
    session_store: SessionStore,
    current_session: SessionRecord,
    agent: CodingAgent,
) -> Optional[SessionRecord]:
    """Persist the current Agent state without terminating the CLI on failure."""
    state = agent.export_session()
    try:
        return session_store.save(
            current_session.session_id,
            state["messages"],
            state["context"],
        )
    except (KeyError, SessionError) as exc:
        print(f"会话保存失败：{exc}")
        return None


def _print_sessions(session_store: SessionStore, current_id: str) -> None:
    """Print workspace sessions in most-recently-used order."""
    records = session_store.list_sessions()
    if not records:
        print("当前工作目录没有可用会话。")
        return
    print("当前工作目录的会话：")
    for record in records:
        marker = "*" if record.session_id == current_id else " "
        print(
            f"{marker} {record.session_id} | {record.turn_count} 轮 | "
            f"{record.updated_at} | {record.title}"
        )


def _print_memories(memory_store: MemoryStore) -> None:
    """Print durable memories without exposing storage implementation details."""
    try:
        records = memory_store.list_memories()
    except MemoryError as exc:
        print(f"读取长期记忆失败：{exc}")
        return
    if not records:
        print("当前工作目录还没有长期记忆。")
        return
    print("当前工作目录的长期记忆：")
    for record in records:
        label = CATEGORY_LABELS[record.category]
        source = "手动" if record.source == "manual" else "自动"
        print(f"  {record.memory_id} | {label} | {source} | {record.content}")


def _command_argument(command: str) -> Optional[str]:
    parts = command.split(maxsplit=1)
    if len(parts) != 2 or not parts[1].strip():
        return None
    return parts[1].strip()


def _print_opened_session(opened: OpenedSession) -> None:
    """Report whether startup resumed history or created a new session."""
    if opened.warning:
        print(f"会话警告：{opened.warning}")
    record = opened.record
    if opened.resumed:
        print(f"当前会话：{record.session_id}")
        print(f"已恢复 {record.turn_count} 轮历史对话。\n")
    else:
        print(f"已创建新会话：{record.session_id}\n")


def _confirm_tool_action(request: ApprovalRequest) -> bool:
    """Show a sensitive action and return the user's explicit decision."""
    print(f"[需要确认] {request.action}")
    print(f"原因：{request.reason}")
    print(f"具体内容：{request.details}")
    try:
        answer = input("是否允许？[y/N] ").strip().lower()
    except EOFError:
        print("[确认] 未收到输入，已拒绝。")
        return False

    allowed = answer in {"y", "yes", "是", "允许"}
    status = "已允许。" if allowed else "已拒绝。"
    print(f"[确认] {status}")
    return allowed


def _print_startup(
    *,
    model: str,
    workspace: Path,
    max_steps: int,
    max_context_chars: int,
    dangerous_commands: bool,
    sandbox_mode: str,
    memory_count: int = 0,
    summary_model: Optional[str] = None,
) -> None:
    """Print a compact, secret-free run configuration summary."""
    safety = (
        "高风险命令需人工确认"
        if dangerous_commands
        else "高风险命令默认拦截"
    )
    print("Mini Coding Agent")
    print(f"模型：{model}")
    print(f"工作目录：{workspace}")
    print(f"最大步骤：{max_steps}")
    print(f"上下文预算：{max_context_chars} 字符")
    print(f"摘要模型：{summary_model or model}")
    print(f"安全模式：{safety}\n")
    isolation = "macOS 系统沙箱" if sandbox_mode == STRICT_MODE else "策略检查"
    print(f"命令隔离：{isolation}\n")
    print(f"长期记忆：{memory_count} 条（按工作目录隔离）\n")
