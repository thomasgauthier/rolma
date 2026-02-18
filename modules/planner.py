# Copyright 2026 Sentient Labs

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file contains code derived from ROMA (https://github.com/sentient-agi/ROMA),
# licensed under Apache 2.0. The code has been modified for knowledge base execution.

from enum import Enum
from typing import Dict, List, Optional

import dspy
from pydantic import BaseModel, Field


class TaskType(Enum):
    RETRIEVE = "RETRIEVE"
    WRITE = "WRITE"
    THINK = "THINK"


class SubTask(BaseModel):
    goal: str = Field(..., min_length=1, description="Precise subtask objective")
    task_type: TaskType = Field(..., description="Type of subtask")
    dependencies: List[str] = Field(
        default_factory=list, description="List of subtask IDs this depends on"
    )
    result: Optional[str] = Field(
        default=None, description="Result of subtask execution (for aggregation)"
    )
    context_input: Optional[str] = Field(
        default=None,
        description="Context from dependent tasks (left-to-right flow)",
    )


class PlannerSignature(dspy.Signature):
    """# REPOSITORY CONTEXT

    Working with `/knowledge_base/` - a knowledge base containing documents from one more sources.

    # Planner — Instruction Prompt

    Role
    Plan a goal into minimal, parallelizable subtasks with a precise, acyclic dependency graph. Do not execute; only plan.

    Available Tools
    If web search tools are available to you, you can use them during planning to:
    - Research current events, trends, or market data when planning tasks that require up-to-date information
    - Verify task requirements or gather context before decomposing complex goals
    - Find relevant documentation, best practices, or domain-specific knowledge to inform your planning
    - Improve the quality and accuracy of RETRIEVE task definitions

    Output Contract (strict)
    - Return only: `subtasks` and `dependencies_graph`. No extra keys, no prose.
    - `subtasks`: list[SubTask]. Each SubTask MUST include:
      - `goal`: imperative, concrete objective for the subtask.
      - `task_type`: one of "THINK", "RETRIEVE", "WRITE".
      - `dependencies`: list[str] of subtask IDs it depends on.
      - `context_input` (optional): brief note on what to consume from dependencies; omit when unnecessary.
    - `dependencies_graph`: dict[str, list[str]] | null
      - Keys and values are subtask IDs as 0-based indices encoded as strings, e.g., "0", "1".
      - Must be acyclic and consistent with each SubTask's `dependencies`.
      - Use empty lists for independent subtasks; set to `{}` if no dependencies, or `null` if not needed.
    - Do not add fields like `id` or `result`. The list index is the subtask ID.

    Task Type Guidance (MECE)
    - THINK: reasoning, derivations, comparisons, validations; no external retrieval.
    - RETRIEVE: fetch/verify external info where freshness, citations, or lookup are essential (replaces "SEARCH").
    - WRITE: produce prose/structured text when inputs are known (emails, outlines, drafts, summaries).

    Decomposition Principles
    - Minimality: Decompose only as much as necessary to reach the goal.
    - MECE: Subtasks should not overlap; together they fully cover the goal.
    - Parallelization: Prefer independent subtasks with a final synthesis step; add dependencies only when required.
    - Granularity: For common tasks, prefer 3–8 total subtasks; keep the number of artefact-producing steps (WRITE/CODE_INTERPRET/IMAGE_GENERATION) to 1–5 unless complexity justifies more.
    - Determinism: Each subtask should have a clear, verifiable completion condition.

    Dependency Rules
    - Use 0-based indices as strings for IDs ("0", "1", ...). The index in `subtasks` is the ID.
    - A subtask may only depend on earlier IDs when linear order is natural; otherwise make independent and merge later.
    - Keep the graph acyclic; avoid chains longer than necessary.
    - Ensure `dependencies_graph` matches each SubTask's `dependencies` exactly.

    Context Flow
    - Outputs from dependencies are available to dependents; do not recompute.
    - When a dependent needs specific artefacts (numbers, citations, outlines), state this succinctly in `context_input`.
    - Numeric values from other subtasks are provided after those subtasks complete; reference them rather than re-deriving.

    Edge Cases
    - If the goal is already atomic, return the minimal valid plan (often 1–3 subtasks) rather than inflating to 3–8.
    - If key requirements are unspecified, add an early THINK step to enumerate assumptions or a RETRIEVE step to collect missing facts.

    Strict Output Shape
    {
      "subtasks": [SubTask, ...],
      "dependencies_graph": {"<id>": ["<id>", ...], ...} | {}
    }

    Do not execute any steps, and do not include reasoning or commentary in the output.
    """

    goal: str = dspy.InputField(
        description="Task that needs to be decomposed into subtasks through planner"
    )
    context: str = dspy.InputField(default=None, description="Execution context (XML)")
    subtasks: List[SubTask] = dspy.OutputField(
        description="List of generated subtasks from planner"
    )
    dependencies_graph: Dict[str, List[str]] = dspy.OutputField(
        default=None,
        description="Task dependency mapping. Keys are subtask indices as strings (e.g., '0', '1'), values are lists of dependency indices as strings. Example: {'1': ['0'], '2': ['0', '1']}",
    )


planner_demos = [
    dspy.Example(
        goal="What is the capital of France?",
        subtasks=[
            SubTask(
                goal="State the capital of France.",
                task_type=TaskType.THINK,
                dependencies=[],
                result=None,
                context_input=None,
            )
        ],
        dependencies_graph={"0": []},
    ).with_inputs("goal"),
    dspy.Example(
        goal="What is the current price of Bitcoin in USD?",
        subtasks=[
            SubTask(
                goal="Fetch the current BTCUSD spot price from a reputable financial source...",
                task_type=TaskType.RETRIEVE,
                dependencies=[],
                result=None,
                context_input="Return price, currency, source, and timestamp.",
            ),
            SubTask(
                goal="Format as 'BTCUSD: <price> USD — <source> <timestamp>'...",
                task_type=TaskType.WRITE,
                dependencies=["0"],
                result=None,
                context_input="Use the fetched price, source, and timestamp from 0.",
            ),
        ],
        dependencies_graph={"0": [], "1": ["0"]},
    ).with_inputs("goal"),
    dspy.Example(
        goal="Create a 1-page privacy policy and a separate cookie policy for my blog.",
        subtasks=[
            SubTask(
                goal="Draft a clear 1-page privacy policy...",
                task_type=TaskType.WRITE,
                dependencies=[],
                result=None,
                context_input=None,
            ),
            SubTask(
                goal="Draft a concise cookie policy...",
                task_type=TaskType.WRITE,
                dependencies=[],
                result=None,
                context_input=None,
            ),
            SubTask(
                goal="Bundle both documents into a single markdown deliverable...",
                task_type=TaskType.WRITE,
                dependencies=["0", "1"],
                result=None,
                context_input="Use the full texts from 0 and 1...",
            ),
        ],
        dependencies_graph={"0": [], "1": [], "2": ["0", "1"]},
    ).with_inputs("goal"),
    dspy.Example(
        goal="Collect Apple and Microsoft's latest quarterly results and compare their guidance side-by-side.",
        subtasks=[
            SubTask(
                goal="Retrieve Apple's most recent quarterly results...",
                task_type=TaskType.RETRIEVE,
                dependencies=[],
                result=None,
                context_input="Include figures, date, source name, and URL.",
            ),
            SubTask(
                goal="Retrieve Microsoft's most recent quarterly results...",
                task_type=TaskType.RETRIEVE,
                dependencies=[],
                result=None,
                context_input="Include figures, date, source name, and URL.",
            ),
            SubTask(
                goal="Create a compact table comparing Apple vs. Microsoft...",
                task_type=TaskType.THINK,
                dependencies=["0", "1"],
                result=None,
                context_input="Use the retrieved metrics and citations from 0 and 1.",
            ),
            SubTask(
                goal="Wrap the table with a 2–3 sentence summary and list citations underneath.",
                task_type=TaskType.WRITE,
                dependencies=["2"],
                result=None,
                context_input="Insert the table from 2...",
            ),
        ],
        dependencies_graph={"0": [], "1": [], "2": ["0", "1"], "3": ["2"]},
    ).with_inputs("goal"),
]


planner = dspy.ChainOfThought(PlannerSignature)

planner.predict.demos = planner_demos
