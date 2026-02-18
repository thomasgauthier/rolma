# This contains code derived from [ROMA](https://github.com/sentient-agi/ROMA), licensed under Apache 2.0. The code has been modified and combined for standalone execution.

import asyncio
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

import dspy
from pydantic import BaseModel, Field
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.tree import Tree as RichTree

from modules import aggregator, atomizer, executor, planner
from telemetry import trace_and_capture

lm_kwargs = {
    "model": "openrouter/deepseek/deepseek-v3.2",
}

lm = dspy.LM(**lm_kwargs)

dspy.configure(lm=lm)


def create_roma_context(goal: str, depth: int = 0, max_depth: int = 3) -> str:
    """Creates the standard ROMA XML context string."""

    return f"""<context>
<fundamental_context>
  <overall_objective>{goal}</overall_objective>
  <recursion>
    <current_depth>{depth}</current_depth>
    <max_depth>{max_depth}</max_depth>
    <at_limit>{"true" if depth >= max_depth else "false"}</at_limit>
  </recursion>
</fundamental_context>
</context>"""


console = Console()


@dataclass
class NodeTrace:
    node_id: str
    parent_id: Optional[str]
    depth: int
    node_type: str
    goal: str
    result: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    start_time: Optional[str] = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    end_time: Optional[str] = None
    plan_data: Optional[Dict] = None  # Store subtasks and graph here

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


# --- Utilities & Validators ---


async def _retry_acall(
    component: Any,
    validator: Optional[Callable[[Any], None]] = None,
    _number_of_retries: int = 25,
    **kwargs,
):
    import dspy

    last_exception = None
    for attempt in range(_number_of_retries):
        try:
            if attempt > 0:
                lm = dspy.settings.lm.copy()
                lm.cache = False
                with dspy.context(lm=lm):
                    result = await component.acall(**kwargs)
            else:
                result = await component.acall(**kwargs)

            if validator:
                validator(result)

            return result

        except Exception as e:
            last_exception = e
            wait_time = 2**attempt
            console.print(
                f"[bold red]⚠️  {component.__class__.__name__} failed (Attempt {attempt + 1}). "
                f"Error: {str(e)}. Retrying...[/]"
            )
            await asyncio.sleep(wait_time)
    raise last_exception


def validate_plan(prediction):
    subtasks = prediction.subtasks
    graph = prediction.dependencies_graph or {}
    num_tasks = len(subtasks)
    if num_tasks == 0:
        raise ValueError("Planner returned 0 subtasks.")
    for node_str, deps in graph.items():
        node_idx = int(node_str)
        if not (0 <= node_idx < num_tasks):
            raise IndexError(f"Graph key '{node_idx}' out of bounds.")
        for dep_str in deps:
            dep_idx = int(dep_str)
            if not (0 <= dep_idx < num_tasks):
                raise IndexError(f"Dependency '{dep_idx}' out of bounds.")
    return True


def validate_execution_output(prediction):
    """Prevents the 'None' result crash by ensuring output exists."""
    if not hasattr(prediction, "output") or not isinstance(prediction.output, str):
        raise ValueError("Executor returned None or missing output field.")
    return True


def safe_serialize(obj):
    # Handle Enums first
    if isinstance(obj, Enum):
        return obj.value
    # Handle Dataclasses
    if hasattr(obj, "__dataclass_fields__"):
        return {k: safe_serialize(v) for k, v in asdict(obj).items()}
    # Handle Pydantic/DSPy models
    if hasattr(obj, "dict"):
        return {k: safe_serialize(v) for k, v in obj.dict().items()}
    # Handle Lists/Tuples
    if isinstance(obj, (list, tuple)):
        return [safe_serialize(i) for i in obj]
    # Handle Dictionaries
    if isinstance(obj, dict):
        return {k: safe_serialize(v) for k, v in obj.items()}
    return obj


def dir_to_dict(path: str):
    """
    Recursively copies a directory structure into a Python dictionary.
    """
    if not os.path.exists(path):
        return None

    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            return "<BINARY_OR_NON_UTF8_CONTENT>"
        except Exception as e:
            return f"<ERROR_READING_FILE: {str(e)}>"

    result_dict = {}
    try:
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            content = dir_to_dict(item_path)
            if content is not None:
                result_dict[item] = content
    except PermissionError:
        return "<PERMISSION_DENIED>"

    return result_dict


# --- Main Orchestrator ---


class DAGOrchestrator:
    def __init__(
        self,
        max_depth: int = 3,
        checkpoint_path: str = "checkpoint.json",
        knowledge_base_path: str = "/knowledge_base/",
    ):
        # Assuming atomizer, planner, executor, aggregator are defined globally
        self.atomizer = atomizer
        self.planner = planner
        self.executor = executor
        self.aggregator = aggregator

        self.max_depth = max_depth
        self.checkpoint_path = checkpoint_path
        self.nodes: Dict[str, NodeTrace] = {}
        self.root_id: Optional[str] = None
        self.knowledge_base = dir_to_dict(knowledge_base_path)

        self._load_checkpoint()

    def _get_node_id(self, goal: str, parent_id: Optional[str]) -> str:
        """Generates a stable ID for the same task in the same position."""
        hash_input = f"{parent_id or 'root'}-{goal}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]

    def _save_checkpoint(self):
        """Saves current execution state to disk, handling Enums and Dataclasses."""
        try:
            # We wrap the whole dictionary in safe_serialize
            data = {
                "root_id": self.root_id,
                "nodes": {
                    nid: safe_serialize(node) for nid, node in self.nodes.items()
                },
            }
            with open(self.checkpoint_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            console.print(f"[bold red]❌ Failed to save checkpoint: {e}[/]")

    def _hydrate_plan(self, plan_data: Dict) -> SimpleNamespace:
        """
        Converts the JSON-serialized plan data back into an object
        structure compatible with the rest of the logic (SimpleNamespace).
        """
        # Reconstruct subtasks as SimpleNamespaces so we can access .goal via dot notation
        subtasks = []
        for st_dict in plan_data.get("subtasks", []):
            # If it's already an object (rare but possible), leave it
            if not isinstance(st_dict, dict):
                subtasks.append(st_dict)
            else:
                subtasks.append(SimpleNamespace(**st_dict))

        return SimpleNamespace(
            subtasks=subtasks,
            dependencies_graph=plan_data.get("dependencies_graph", {}),
        )

    def _load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, "r") as f:
                    data = json.load(f)
                self.root_id = data.get("root_id")
                for nid, n_dict in data.get("nodes", {}).items():
                    self.nodes[nid] = NodeTrace.from_dict(n_dict)
                console.print(
                    f"[bold cyan]🔄 Resumed from checkpoint: {len(self.nodes)} nodes loaded.[/]"
                )
            except Exception as e:
                console.print(f"[bold red]❌ Failed to load checkpoint: {e}[/]")

    def _pretty_log(self, role: str, title: str, content: str, color: str = "cyan"):
        # Extra safety: Rich cannot render None.
        display_content = str(content) if content is not None else "[NULL CONTENT]"
        panel = Panel(
            display_content,
            title=f"[bold {color}]{role}: {title}[/]",
            border_style=color,
            padding=(1, 2),
            expand=False,
        )
        console.print(panel)

    @trace_and_capture
    async def atomize(self, goal: str, context: str, node_id: str) -> SimpleNamespace:
        """
        Determines if a task is atomic (EXECUTE) or complex (PLAN).
        Normalizes the output to ensure 'is_atomic' always exists.
        """
        existing_node = self.nodes[node_id]

        # --- RESUME LOGIC ---
        # 1. Strongest signal: Plan data exists on disk -> Must be a PLAN.
        if existing_node.plan_data:
            return SimpleNamespace(is_atomic=False, node_type="PLAN")

        # 2. Previous decision exists but unfinished -> Respect the saved decision.
        if existing_node.node_type not in ["PENDING", "UNKNOWN"]:
            # Derive is_atomic from the string type
            return SimpleNamespace(
                is_atomic=(existing_node.node_type == "EXECUTE"),
                node_type=existing_node.node_type,
            )

        # --- FRESH LOGIC ---
        with console.status(
            f"[bold green]Atomizing task {node_id[:8]}...", spinner="dots"
        ):
            # Result usually contains just 'node_type' (Enum or str)
            raw_res = await _retry_acall(self.atomizer, goal=goal, context=context)

            # Normalize: Ensure we work with strings and derive boolean
            determined_type = str(raw_res.node_type).upper()
            is_atomic = determined_type == "EXECUTE"

            # Update State & Checkpoint
            existing_node.node_type = determined_type
            self._save_checkpoint()

            # Return normalized object matching the Resume Logic structure
            return SimpleNamespace(
                is_atomic=is_atomic,
                node_type=determined_type,
                reasoning=getattr(raw_res, "reasoning", None),
            )

    @trace_and_capture
    async def solve(
        self,
        goal: str,
        depth: int = 0,
        parent_context: Optional[str] = None,
        initial_goal: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> str:
        # 1. Identify Node
        node_id = self._create_node(goal, depth, parent_id)
        existing_node = self.nodes[node_id]

        # 2. Check Completion (Fast Exit)
        if existing_node.result is not None:
            console.print(f"[dim]⏩ Skipping {node_id[:8]} (Already Complete)[/]")
            return existing_node.result

        # Setup Context
        initial_goal = initial_goal or goal
        context = parent_context or f"Goal: {goal}"

        # 3. Determine Path (Atomize)
        # Now guaranteed to have .is_atomic and .node_type
        atom_decision = await self.atomize(goal=goal, context=context, node_id=node_id)

        # 4. Route Execution
        if atom_decision.is_atomic:
            # --- EXECUTION BRANCH ---
            # Double check: if atomize said EXECUTE, we trust it.
            result = await self.execute(
                goal=goal, execution_context=context, node_id=node_id, depth=depth
            )

        else:
            # --- PLANNING BRANCH ---

            # A) Acquire Plan (Load or Generate)
            if existing_node.plan_data:
                console.print(f"[bold magenta]📖 Hydrating plan for {node_id[:8]}[/]")
                plan_res = self._hydrate_plan(existing_node.plan_data)
            else:
                # If we are here, atomize said "PLAN", but we haven't generated the plan yet.
                plan_res = await self.plan(goal=goal, context=context, depth=depth)

                # Critical State Transition: Save Plan Data IMMEDIATELY
                existing_node.plan_data = {
                    "subtasks": [safe_serialize(st) for st in plan_res.subtasks],
                    "dependencies_graph": plan_res.dependencies_graph,
                }
                # Reinforce node_type in case of manual intervention
                existing_node.node_type = "PLAN"
                self._save_checkpoint()

            # B) Solve Subtasks (Recursive)
            # We pass the plan_res down. Results are stored in the *child* nodes on disk.
            # We re-collect them here into a list for the aggregator.
            if depth >= self.max_depth:
                sub_results = await self._force_execute_subtasks(
                    plan_res.subtasks, context, node_id, depth
                )
            else:
                sub_results = await self._solve_subtasks(
                    plan_res.subtasks,
                    plan_res.dependencies_graph or {},
                    depth,
                    context,
                    initial_goal,
                    node_id,
                )

            # C) Aggregate
            result = await self.aggregate(
                goal=goal,
                plan_res=plan_res,
                sub_results=sub_results,
                context=context,
                node_id=node_id,
                depth=depth,
            )

        # 5. Root Visualization
        if depth == 0:
            console.print(
                f"\n[bold reverse green] 🏁 RUN COMPLETE [/] Total Nodes: {len(self.nodes)}\n"
            )
            self.render_tree()

        return result

    @trace_and_capture
    async def execute(
        self,
        goal: str,
        execution_context: str,
        node_id: str,
        depth: int,
        node_type: str = "EXECUTE",
    ) -> str:
        self._pretty_log("WORKER", f"Executing Node {node_id}", goal, "yellow")

        with console.status(
            "[bold yellow]LLM is generating response...", spinner="earth"
        ):
            # Use validator to prevent storing 'None' in the trace
            res_obj = await _retry_acall(
                self.executor,
                goal=goal,
                execution_context=execution_context,
                context=self.knowledge_base,
                validator=validate_execution_output,
            )
            res = res_obj.output

        self._pretty_log("RESULT", "Worker Output", res, "green")
        self.nodes[node_id].node_type, self.nodes[node_id].result = (
            node_type,
            res,
        )
        self.nodes[node_id].end_time = datetime.now(timezone.utc).isoformat()
        self._save_checkpoint()  # Save after every worker success
        return res

    @trace_and_capture
    async def plan(self, goal: str, context: str, depth: int) -> Any:
        self._pretty_log("PLANNER", "Developing Strategy", goal, "magenta")
        plan_res = await _retry_acall(
            self.planner, goal=goal, context=context, validator=validate_plan
        )

        table = Table(show_header=True, header_style="bold magenta", box=None)
        table.add_column("Step", style="dim")
        table.add_column("Sub-Goal")
        for i, st in enumerate(plan_res.subtasks):
            table.add_row(f"#{i}", st.goal)

        console.print(
            Panel(
                table,
                title="[magenta]Proposed Subtasks[/]",
                border_style="magenta",
            )
        )

        return plan_res

    @trace_and_capture
    async def aggregate(
        self, goal, plan_res, sub_results, context, node_id, depth
    ) -> str:
        self._pretty_log(
            "AGGREGATOR",
            "Synthesizing Final Answer",
            f"Goal: {goal[:100]}...",
            "blue",
        )

        with console.status("[bold blue]Synthesizing subtasks...", spinner="dots"):
            res = (
                await _retry_acall(
                    self.aggregator,
                    original_goal=goal,
                    subtasks_results=sub_results,
                    context=context,
                )
            ).synthesized_result

        self._pretty_log("FINAL", "Aggregated Result", res, "bold blue")
        self.nodes[node_id].result = res
        self.nodes[node_id].end_time = datetime.now(timezone.utc).isoformat()
        self._save_checkpoint()
        return res

    def _create_node(self, goal, depth, parent_id) -> str:
        nid = self._get_node_id(goal, parent_id)
        if nid not in self.nodes:
            self.nodes[nid] = NodeTrace(
                node_id=nid,
                parent_id=parent_id,
                depth=depth,
                node_type="PENDING",
                goal=goal,
            )
        if not self.root_id:
            self.root_id = nid
        if parent_id in self.nodes and nid not in self.nodes[parent_id].child_ids:
            self.nodes[parent_id].child_ids.append(nid)
        return nid

    async def _solve_subtasks(
        self, subtasks, deps, depth, context, init_goal, parent_id
    ):
        completed = {}
        for idx in self._topological_sort(subtasks, deps):
            st = subtasks[idx]
            console.print(f"\n[bold cyan]▶ Starting Subtask {idx}[/]")
            st_ctx = self._inject_deps(
                context, deps.get(str(idx), []), subtasks, completed
            )
            st.result = await self.solve(
                goal=st.goal,
                depth=depth + 1,
                parent_context=st_ctx,
                initial_goal=init_goal,
                parent_id=parent_id,
            )
            completed[idx] = st.result
        return subtasks

    async def _force_execute_subtasks(self, subtasks, context, parent_id, depth):
        for st in subtasks:
            nid = self._create_node(st.goal, depth + 1, parent_id)
            st.result = await self.execute(
                goal=st.goal,
                execution_context=context,
                node_id=nid,
                depth=depth + 1,
                node_type="EXECUTE_MAX_DEPTH",
            )
        return subtasks

    def _topological_sort(self, subtasks, dependencies_graph) -> List[int]:
        n = len(subtasks)
        in_degree = [0] * n
        adj_list = {i: [] for i in range(n)}
        for node_str, deps in dependencies_graph.items():
            node = int(node_str)
            for dep_str in deps:
                dep = int(dep_str)
                adj_list[dep].append(node)
                in_degree[node] += 1
        queue = [i for i in range(n) if in_degree[i] == 0]
        result = []
        while queue:
            node = queue.pop(0)
            result.append(node)
            for neighbor in adj_list[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        return result if len(result) == n else list(range(n))

    def _inject_deps(self, context, dep_indices, subtasks, completed_results) -> str:
        if not dep_indices:
            return context
        dep_xml = []
        for idx_str in dep_indices:
            idx = int(idx_str)
            if idx in completed_results:
                dep_xml.append(
                    f"    <dependency>\n      <goal>{subtasks[idx].goal}</goal>\n      <output>{completed_results[idx]}</output>\n    </dependency>"
                )
        if not dep_xml:
            return context
        block = f"<executor_specific>\n  <dependency_results>\n{chr(10).join(dep_xml)}\n  </dependency_results>\n</executor_specific>\n"
        return (
            context.replace("</context>", f"{block}</context>")
            if "</context>" in context
            else context + "\n" + block
        )

    def render_tree(self):
        if not self.root_id:
            return
        root_data = self.nodes[self.root_id]
        tree = RichTree(f"[bold blue]Root Task: {root_data.goal[:80]}...")

        def _build(nid, rich_node):
            n = self.nodes[nid]
            for cid in n.child_ids:
                child = self.nodes[cid]
                is_exec = "EXECUTE" in child.node_type
                color = "yellow" if is_exec else "magenta"
                icon = "⚡" if is_exec else "🧠"
                branch = rich_node.add(
                    f"[{color}]{icon} [{cid}] {child.goal[:60]}...[/]"
                )
                _build(cid, branch)

        _build(self.root_id, tree)
        console.print(
            Panel(tree, title="Task Execution Trace", border_style="bright_black")
        )


async def answer(query):
    orchestrator = DAGOrchestrator(max_depth=3)

    return await orchestrator.solve(
        goal=f"""<user_question>
{query}
</user_question>"""
    )


await answer(query="")


async def main():
    while True:
        query = input(
            "Enter a question about your knowledge base (or 'quit' to exit): "
        ).strip()

        if query.lower() in ("quit", "q", "exit"):
            print("Goodbye!")
            break

        if not query:
            print("Please enter a valid question.\n")
            continue

        print("\nProcessing your question...\n")
        result = await answer(query=query)
        print("\n" + "=" * 50)
        print("FINAL ANSWER:")
        print("=" * 50)
        print(result)
        print("\n")


# Run the top-level async function
if __name__ == "__main__":
    asyncio.run(main())
