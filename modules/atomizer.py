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
from typing import Optional

import dspy


class NodeType(Enum):
    PLAN = "PLAN"
    EXECUTE = "EXECUTE"


class AtomizerSignature(dspy.Signature):
    """
    # Atomizer — Instruction Prompt

    Role
    Classify the goal as ATOMIC or NOT and set `node_type`. Do not solve the task.

    Available Executors (for atomic tasks only)
    - Think, Search, Write

    Decision Rules
    - Atomic (→ EXECUTE) iff ALL are true:
      1) Single deliverable — exactly one answer/artefact/transformation.
      2) Single executor suffices — exactly one of Think OR Search OR Write can produce the final output in one pass.
      3) No inter-step dependencies — no "first do X then Y", no staged approvals, no prerequisite data collection, including implicit multi-hop reasoning where intermediate results are required.
      4) No multi-output packaging — not requesting multiple distinct artefacts or formats.
      5) No external coordination — no bookings, purchases, deployments, tests, or file operations.

    Notes
    - Needing web retrieval or citations does not always force planning; if one Search pass can do it, it's atomic.

    When to choose PLAN (→ PLAN)
    - Any multi-step sequencing (outline→draft, generate→evaluate→select, research A & B → compare).
    - Multiple deliverables or formats.
    - Parallel subtasks to be synthesized.
    - Clarification required before executing the goal.
    - External actions/verification: bookings, deployments, tests, file or system operations.
    - Long procedural projects with dependencies.
     - Implicit multi-hop dependencies — chained lookups or intermediate computations are needed to reach the answer.

    Tie-breaker
    - If a single executor can reasonably deliver the end result in one pass, choose EXECUTE; otherwise PLAN.

    Strict Output Contract
    - Return ONLY this JSON object (no prose, no extra keys, no markdown):
    {
      "is_atomic": true|false,
      "node_type": "EXECUTE"|"PLAN"
    }

    Compliance
    - Do not design plans, pick executors, or add explanations.
    - Do not solve or partially solve the task.
    - Output exactly the two fields above, nothing else.
    """

    goal: str = dspy.InputField(description="Task to atomize")
    context: Optional[str] = dspy.InputField(
        default=None, description="Execution context (XML)"
    )
    is_atomic: bool = dspy.OutputField(
        description="True if task can be executed directly"
    )
    node_type: NodeType = dspy.OutputField(
        description="Type of node to process (PLAN or EXECUTE)"
    )


atomizer_demos = [
    dspy.Example(
        goal="Compute 23 × 47.",
        is_atomic=True,
        node_type=NodeType.EXECUTE,
    ).with_inputs("goal"),
    dspy.Example(
        goal="What is the current price of Bitcoin in USD?",
        is_atomic=True,
        node_type=NodeType.EXECUTE,
    ).with_inputs("goal"),
    dspy.Example(
        goal="Translate to Japanese: 'I love ramen.'",
        is_atomic=True,
        node_type=NodeType.EXECUTE,
    ).with_inputs("goal"),
    dspy.Example(
        goal="Outline a 10-chapter book and then write Chapter 1.",
        is_atomic=False,
        node_type=NodeType.PLAN,
    ).with_inputs("goal"),
    dspy.Example(
        goal=(
            "Recommend the best laptop for me under $1500—ask me 5 questions first, then decide."
        ),
        is_atomic=False,
        node_type=NodeType.PLAN,
    ).with_inputs("goal"),
    dspy.Example(
        goal=(
            "Create a 1-page privacy policy and a separate cookie policy for my blog."
        ),
        is_atomic=False,
        node_type=NodeType.PLAN,
    ).with_inputs("goal"),
]


atomizer = dspy.ChainOfThought(AtomizerSignature)

atomizer.predict.demos = atomizer_demos
