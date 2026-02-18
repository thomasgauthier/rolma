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
from typing import List, Optional

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


class AggregatorSignature(dspy.Signature):
    """
    # Aggregator — Instruction Prompt

    Role
    Synthesize child subtask results into a single, high-quality answer that directly satisfies the original goal. Do not re-plan or re-execute subtasks.

    Synthesis Principles
    - Goal alignment: Answer precisely what `original_goal` asks for (scope, units, format).
    - Evidence-driven: Use only provided child `result` content; do not invent facts.
    - Fidelity: Preserve key details, numbers, and constraints surfaced by child results.
    - Dependency-aware: Respect implicit ordering from dependencies; later synthesis may rely on earlier computations.
    - Concision with completeness: Be as brief as possible while fully satisfying the goal.

    Consistency & Math Rules
    - If percentages are "of previous step," apply compounding; otherwise treat as of the original baseline unless explicitly stated.
    - Keep arithmetic consistent across child results; do not re-derive if a definitive figure is provided.
    - Resolve conflicts by preferring:
      1) More precise/explicit computations over vague statements;
      2) Later synthesis steps that consolidate earlier ones;
      3) Results that explicitly reference required constraints of the goal.

    Formatting Guidelines
    - Match any format implied by `original_goal` (bullets, table, or a short paragraph). If no format is specified, provide a clear paragraph.
    - Include units, rounding, and labeling exactly as requested; round at the end.
    - If child results contain citations or source notes, retain them compactly at the end.

    Strict Output Shape
    Return only the synthesized_result string. No extra keys, no markdown fences, no commentary.
    """

    original_goal: str = dspy.InputField(description="Original goal of the task")
    subtasks_results: List[SubTask] = dspy.InputField(
        description="List of completed subtask results to synthesize"
    )
    context: Optional[str] = dspy.InputField(
        default=None, description="Execution context (XML)"
    )
    synthesized_result: str = dspy.OutputField(description="Final synthesized output")


aggregator = dspy.ChainOfThought(AggregatorSignature)
