import time
from typing import Any, Dict, Optional, List, TypedDict, Annotated, cast
import operator
import json
import threading
import traceback
import pathlib
import re
import os
from collections import deque
from datetime import datetime, timezone
import uuid

from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.types import Send
import operator
import re
import pathlib
import json
import time
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


class LangGraphNodes:

    @staticmethod
    async def planner_node(llm: BaseChatModel,state: MessagesState) -> Dict[str, Any]:
        planner_prompt = SystemMessage(content=(
            "あなたは実行計画作成者です。ユーザーの依頼を分析し、実行可能なサブタスクに分解してください。\n"
            "\n"    
            "【重要】出力はJSONのみ。余計な説明/見出し/担当者割り当ては書かないでください。\n"
            "次の形式に厳密に従ってください: {\"tasks\": [\"...\", ...]}\n"
            "- tasks は最大6件\n"
            "- 各taskは1〜2文で具体的に（ツールで実行できる粒度）\n"
            "- タスク内でユーザーに質問したり、選択肢(1/2/3)や出力形式(CSV/表/JSON)の指定を求めない（必要なら自分で決めて進める）\n"
            "- なるべく『最終アウトプット』が返るタスクにする（例: 5件を表形式で列挙して一言評価まで完了）\n"
            "- 重要: このワークフローはタスクを並列実行する。タスク間で前段の出力が必要になる依存関係を作らない。依存が避けられない場合は分割せず1つのtaskに統合する\n"
            "- 各taskは単独で完結し、必要な入力（対象ファイル/ディレクトリ）と期待する出力形式まで含める\n"
            "- 実行環境メモ: executor コンテナの作業ディレクトリは /workspace。ファイル参照はホスト絶対パスではなく /workspace からの相対パス（例: /workspace/14-front/package.json）を使う\n"
            "- 『担当エージェント』や『タスクの割り振り』などは出力しない\n"
        ))
        # プランナーにはツールをバインドしない（思考に専念させる）
        response = await llm.ainvoke([planner_prompt] + state["messages"])
        return {"messages": [response]}

    @staticmethod
    async def supervisor_agent(
        llm: BaseChatModel, 
        state: MessagesState, 
        tools: list
        ) -> Dict[str, Any]:
        
        sys_prompt = SystemMessage(content=(
            "あなたは実行責任者です。承認された計画に基づき、直ちにツールを呼び出して実行してください。\n"
            "「了解しました」などの挨拶は不要です。まず最初のステップに必要なツールを呼び出してください。"
        ))
        
        response = await llm.ainvoke([sys_prompt] + state["messages"])
        return {"messages": [response]}


    @staticmethod
    async def planner_summarize_results(
        *,
        llm: BaseChatModel,
        original_request: str,
        results: list[dict[str, Any]],
        raw_summary: str,
    ):
        """並列実行の結果をPlanner視点で要約する（ツール呼び出し無し）。"""
        sys_prompt = SystemMessage(content=(
            "あなたはPlanner（統合責任者）です。以下の実行結果を、ユーザー向けに日本語で簡潔にまとめてください。\n"
            "- 成果物（得られた情報）\n"
            "- 失敗したタスクと原因（分かる範囲）\n"
            "- 次にやるべきこと（最大3点）\n"
            "\n"
            "注意: 事実は結果からのみ述べ、推測は『推測』と明記してください。\n"
        ))

        # LLMへの入力は大きくなりやすいので、ここでは既に圧縮されたサマリ文字列を主に渡す。
        user_prompt = HumanMessage(content=(
            "[元の依頼]\n"
            f"{original_request}\n\n"
            "[実行結果サマリ]\n"
            f"{raw_summary}\n"
        ))
        return await llm.ainvoke([sys_prompt, user_prompt])

class ParallelAgentWorkflow:
    @staticmethod
    def should_continue(state: MessagesState):
        """次の遷移先を決定するルーティング関数"""
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:        
            return "tools" # ツール呼び出しがあればtoolsノードへ
        return END         # なければ会話終了

    def __init__(self,         
            include_planner: bool = True,
            include_planner_summary: bool = True,
            tools: Optional[list] = None,
            parallel: bool = False,
            max_parallel: int = 4,
) -> None:
        self.include_planner = include_planner
        self.include_planner_summary = include_planner_summary
        self.tools = tools
        self.parallel = parallel
        self.max_parallel = max_parallel

    def create_graph(self) -> StateGraph:
        if self.parallel:
            return self._create_parallel_graph_(
                include_planner=self.include_planner,
                include_planner_summary=self.include_planner_summary,
                tools=self.tools,
                max_parallel=self.max_parallel,
            )
        else:
            return self._create_simple_graph_(include_planner=self.include_planner, tools=self.tools)

    def _create_simple_graph_(self, include_planner: bool = False, tools: Optional[list] = None) -> StateGraph:
        workflow = StateGraph(MessagesState)


        async def create_agent_node(state: MessagesState):
            return await LangGraphNodes.supervisor_agent(state, tools=tools)
        workflow.add_node("agent", create_agent_node)
        workflow.add_node("tools", ToolNode(effective_tools))

        if include_planner:
            workflow.add_node("planner", LangGraphNodes.planner_node)
            workflow.add_edge(START, "planner")
            workflow.add_edge("planner", "agent")
        else:
            workflow.add_edge(START, "agent")

        workflow.add_conditional_edges("agent", self.should_continue)
        workflow.add_edge("tools", "agent")

        return workflow


    def _create_parallel_graph_(
        self,
        *,
        include_planner: bool = True,
        include_planner_summary: bool = True,
        tools: Optional[list] = None,
        max_parallel: int = 4,
    ) -> StateGraph:
        """計画をサブタスクに分割し、最大 max_parallel 件を並列実行して統合するPoCグラフ。"""
        if max_parallel < 1:
            raise ValueError("max_parallel must be >= 1")

        effective_tools = local_tools if tools is None else tools
        tools_by_name = {getattr(t, "name", str(i)): t for i, t in enumerate(effective_tools)}

        async def plan_to_tasks(state: ParallelExecutionState) -> Dict[str, Any]:
            # planner が有効な場合は最後のメッセージ（planner出力）から抽出
            msgs = state.get("messages") or []
            text = ""
            if include_planner and msgs:
                text = getattr(msgs[-1], "content", "") or ""
            elif msgs:
                text = getattr(msgs[0], "content", "") or ""

            tasks = _extract_tasks_from_plan_text(text)
            return {
                "tasks": tasks,
                "task_queue": tasks,
                "current_batch": [],
                "results": [],
            }

        async def dispatch(state: ParallelExecutionState) -> Dict[str, Any]:
            queue = list(state.get("task_queue") or [])
            batch = queue[:max_parallel]
            rest = queue[max_parallel:]
            return {"current_batch": batch, "task_queue": rest}

        def route_after_dispatch(state: ParallelExecutionState):
            batch = state.get("current_batch") or []
            if not batch:
                return "join"
            # NOTE:
            # Send() の payload は worker 側 state として渡されるため、
            # 共有コンテキスト（source_dirs）が欠落すると executor が /workspace を準備できない。
            shared: Dict[str, Any] = {}
            if state.get("source_dirs") is not None:
                shared["source_dirs"] = state.get("source_dirs")
            if state.get("trace_id") is not None:
                shared["trace_id"] = state.get("trace_id")
            # 対話制御（HITL）を worker にも伝播する
            if state.get("auto_approve") is not None:
                shared["auto_approve"] = bool(state.get("auto_approve"))

            return [Send("worker", {**shared, "task": t}) for t in batch]

        async def worker(state: ParallelExecutionState) -> Dict[str, Any]:
            task = (state.get("task") or "").strip()
            msgs = state.get("messages") or []
            original_request = getattr(msgs[0], "content", "") if msgs else ""

            started_at = time.time()
            # NOTE: stream_mode="updates" だと worker の state update は完了時にしか見えないため、
            # 並列性が分かるように開始ログを標準出力へ出す（PoC）。
            print(f"🧩 worker started task={task!s} at={started_at:.3f}")

            prompt = (
                "あなたは実行担当の自律型コーディングエージェントです。\n"
                "次のサブタスクを実行してください。必要ならファイルを読み、結果を日本語で報告してください。\n\n"
                f"[サブタスク]\n{task}\n\n"
                f"[元の依頼]\n{original_request}\n"
            )

            source_dirs = state.get("source_dirs")
            auto_approve = bool(state.get("auto_approve"))
            trace_id = state.get("trace_id")

            workspace_path = None
            if isinstance(source_dirs, list) and source_dirs:
                workspace_path = str(source_dirs[0].resolve())

            # ツール選択（ローカル優先、次にzip、最後に通常）
            tool = None
            tool_args: Dict[str, Any] = {"prompt": prompt}

            if not auto_approve and "run_executor_local_hitl" in tools_by_name:
                tool = tools_by_name["run_executor_local_hitl"]
                if source_dirs is not None:
                    tool_args["source_dirs"] = [str(p) for p in source_dirs]
                tool_args["timeout"] = 600
                if trace_id:
                    tool_args["trace_id"] = trace_id

            elif "run_executor_local" in tools_by_name:
                tool = tools_by_name["run_executor_local"]
                if source_dirs is not None:
                    tool_args["source_dirs"] = [str(p) for p in source_dirs]
                tool_args["timeout"] = 600
                if trace_id:
                    tool_args["trace_id"] = trace_id
            elif "run_autonomous_agent_executor" in tools_by_name:
                tool = tools_by_name["run_autonomous_agent_executor"]
                if not workspace_path:
                    raise RuntimeError(
                        "workspace_path is required for run_autonomous_agent_executor. "
                        "Provide source_dirs (shared workspace) in the workflow state."
                    )
                tool_args["workspace_path"] = workspace_path
                tool_args["timeout"] = 600
                if trace_id:
                    tool_args["trace_id"] = trace_id
            else:
                raise RuntimeError(f"No suitable executor tool found. tools={list(tools_by_name.keys())}")

            result = await tool.ainvoke(tool_args)

            ended_at = time.time()
            print(f"🧩 worker ended   task={task!s} at={ended_at:.3f} elapsed={ended_at - started_at:.3f}s")
            return {
                "results": [
                    {
                        "task": task,
                        "started_at": started_at,
                        "ended_at": ended_at,
                        "elapsed_sec": round(ended_at - started_at, 3),
                        "tool": getattr(tool, "name", None),
                        "result": result,
                    }
                ]
            }

        async def join(state: ParallelExecutionState) -> Dict[str, Any]:
            results = state.get("results") or []
            lines: list[str] = []
            lines.append(f"並列実行の結果サマリ (max_parallel={max_parallel}):")
            for i, item in enumerate(results, start=1):
                task = item.get("task")
                tool_name = item.get("tool")
                elapsed = item.get("elapsed_sec")
                res = item.get("result")
                status = res.get("status") if isinstance(res, dict) else None
                stdout = res.get("stdout") if isinstance(res, dict) else None
                tail = ""
                if isinstance(stdout, str) and stdout.strip():
                    tail = stdout.strip().splitlines()[-1]
                lines.append(f"{i}. task={task!s} tool={tool_name!s} elapsed={elapsed!s}s status={status!s} last={tail!s}")

            report = "\n".join(lines)
            # 次段の summarizer で使えるよう state に保持しておく
            return {"raw_summary": report}

        async def planner_summarize(state: ParallelExecutionState) -> Dict[str, Any]:
            """join の raw_summary と results をもとに、Plannerがユーザー向けに最終報告を生成する。"""
            msgs = state.get("messages") or []
            original_request = getattr(msgs[0], "content", "") if msgs else ""
            raw_summary = (state.get("raw_summary") or "").strip()
            results = state.get("results") or []

            if not include_planner_summary:
                # 旧挙動互換: raw_summary をそのまま返す
                return {"messages": [AIMessage(content=raw_summary)]}

            # Planner(LLM)で要約
            summary_msg = await LangGraphNodes.planner_summarize_results(
                original_request=original_request,
                results=results,
                raw_summary=raw_summary,
            )

            # 生サマリも併記（デバッグ/検証用）
            final_text = (getattr(summary_msg, "content", "") or "").strip()
            if raw_summary:
                final_text = f"{final_text}\n\n---\n[詳細サマリ]\n{raw_summary}".strip()
            return {"messages": [AIMessage(content=final_text)]}

        workflow = StateGraph(ParallelExecutionState)
        workflow.add_node("plan_to_tasks", plan_to_tasks)
        workflow.add_node("dispatch", dispatch)
        workflow.add_node("worker", worker)
        workflow.add_node("join", join)
        workflow.add_node("planner_summarize", planner_summarize)

        if include_planner:
            workflow.add_node("planner", LangGraphNodes.planner_node)
            workflow.add_edge(START, "planner")
            workflow.add_edge("planner", "plan_to_tasks")
        else:
            workflow.add_edge(START, "plan_to_tasks")

        workflow.add_edge("plan_to_tasks", "dispatch")
        workflow.add_conditional_edges("dispatch", route_after_dispatch)
        workflow.add_edge("worker", "dispatch")
        workflow.add_edge("join", "planner_summarize")
        workflow.add_edge("planner_summarize", END)

        return workflow

class ParallelExecutionState(TypedDict, total=False):
    # Keep LangGraph message semantics
    messages: Annotated[list, add_messages]

    # Optional execution context
    source_dirs: Optional[list[pathlib.Path]]

    # Trace context (SV実行全体の相関ID)
    trace_id: Optional[str]

    # Parallel task execution
    tasks: list[str]
    task_queue: list[str]
    current_batch: list[str]
    task: str
    results: Annotated[list[Dict[str, Any]], operator.add]
    raw_summary: str

    # CLI controls
    auto_approve: bool


def _extract_tasks_from_plan_text(plan_text: str, *, max_tasks: int = 50) -> list[str]:
    """Planner出力(Markdown/JSON/自然文)からサブタスク文字列を抽出する（PoC）。

    目的:
    - 章見出し/担当割り振り/補足説明などを「サブタスク」と誤認して大量実行しない
    - JSON({"tasks": [...]}) が来た場合はそれを最優先で使う
    """
    text = (plan_text or "").strip()
    if not text:
        return []

    def _clean_task(s: str) -> str:
        s = (s or "").strip()
        if not s:
            return ""
        # 全体が **...** / `...` の場合は剥がす
        m_bold = re.fullmatch(r"\*\*(.+)\*\*", s)
        if m_bold:
            s = m_bold.group(1).strip()
        m_code = re.fullmatch(r"`(.+)`", s)
        if m_code:
            s = m_code.group(1).strip()
        # 末尾の句読点/コロンを軽く正規化
        s = s.rstrip("：:")
        return s

    def _is_meta_task(s: str) -> bool:
        if not s:
            return True
        if s.startswith("担当エージェント"):
            return True
        if "タスクの割り振り" in s or "タスク割り振り" in s:
            return True
        return False

    def _dedupe_keep_order(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            key = item.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    # 1) JSON を最優先
    for candidate in (text,):
        try:
            if candidate.lstrip().startswith("{"):
                obj = json.loads(candidate)
                if isinstance(obj, dict) and isinstance(obj.get("tasks"), list):
                    raw_tasks = [t for t in (obj.get("tasks") or []) if isinstance(t, str)]
                    cleaned = [_clean_task(t) for t in raw_tasks]
                    cleaned = [t for t in cleaned if t and not _is_meta_task(t)]
                    cleaned = _dedupe_keep_order(cleaned)
                    return cleaned[:max_tasks]
        except Exception:
            pass

    # JSONが文章中に埋まっているケース（最初の { 〜 最後の } を雑に試す）
    try:
        start = text.find("{")
        end = text.rfind("}")
        if 0 <= start < end:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict) and isinstance(obj.get("tasks"), list):
                raw_tasks = [t for t in (obj.get("tasks") or []) if isinstance(t, str)]
                cleaned = [_clean_task(t) for t in raw_tasks]
                cleaned = [t for t in cleaned if t and not _is_meta_task(t)]
                cleaned = _dedupe_keep_order(cleaned)
                return cleaned[:max_tasks]
    except Exception:
        pass

    # 2) Markdown/自然文: 「タスクリスト」セクションのトップレベル項目だけ拾う
    start_markers = ("タスクリスト", "タスク一覧", "task list", "tasks")
    stop_markers = ("タスクの割り振り", "タスク割り振り", "割り振り", "担当エージェント")

    def _extract_top_level_items(lines: list[str], *, require_tasks_section: bool) -> list[str]:
        in_tasks_section = not require_tasks_section
        collected: list[str] = []

        for raw_line in lines:
            stripped = raw_line.strip()
            if not stripped:
                continue

            normalized = stripped.lstrip("#").strip()
            if not in_tasks_section and any(m in normalized.lower() for m in start_markers):
                in_tasks_section = True
                continue
            if in_tasks_section and any(m in normalized for m in stop_markers):
                break
            if not in_tasks_section:
                continue

            # ネストした箇条書き（インデントあり）は爆発しやすいので捨てる
            leading_spaces = len(raw_line) - len(raw_line.lstrip(" "))
            if leading_spaces >= 2:
                continue

            m_num = re.match(r"^\s*\d+[\.|\)]\s+(.+)$", raw_line)
            if m_num:
                task = _clean_task(m_num.group(1))
                if task and not _is_meta_task(task):
                    collected.append(task)
            else:
                m_bullet = re.match(r"^\s*[-\*]\s+(.+)$", raw_line)
                if m_bullet:
                    task = _clean_task(m_bullet.group(1))
                    if task and not _is_meta_task(task):
                        collected.append(task)

            if len(collected) >= max_tasks:
                break

        return collected

    lines = text.splitlines()
    tasks = _extract_top_level_items(lines, require_tasks_section=True)
    if not tasks:
        # タスクリストの見出しが無い場合に備えて、全体からトップレベル箇条書きだけ拾う
        tasks = _extract_top_level_items(lines, require_tasks_section=False)

    tasks = _dedupe_keep_order(tasks)
    if tasks:
        return tasks[:max_tasks]

    # 3) 最後の手段: 全体を1タスクとして扱う
    return [text]
