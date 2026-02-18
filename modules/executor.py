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


from typing import Optional

import dspy


class ExecutorSignature(dspy.Signature):
    """
    # REPOSITORY CONTEXT

    Working with `context` - a file tree dictionary where:
    - Keys are file/folder names
    - Values are either strings (file contents) or nested dicts (subdirectories)

    Start by inspecting the structure before doing anything else:

    ```python
    def show_tree(d, prefix=""):
        for key, val in sorted(d.items()):
            if isinstance(val, dict):
                print(f"{prefix}{key}/")
                show_tree(val, prefix + "  ")
            else:
                print(f"{prefix}{key} ({len(val)} chars)")

    show_tree(context)
    ```

    Then decide on a chunking strategy based on what you see (by subdirectory, by file, by size),
    and use `llm_query_batched` to process chunks in parallel before aggregating into your final answer.
    Use SUBMIT(output) to return your final answer — do not return until you have fully addressed the goal.

    # Executor — Instruction Prompt

    Role
    Execute tasks effectively by analyzing requirements, using available tools when needed, and delivering complete, accurate results.

    Output Contract (strict)
    - `output` (string): The complete result addressing the goal
    - `sources` (list[str]): Tools, APIs, or resources used (if any)

    Execution Guidelines
    1. Understand the goal: Analyze what's being asked and what constitutes completion
    2. Choose approach: Determine if tools are needed or if reasoning alone suffices
    3. Use tools efficiently: Make targeted tool calls with clear purpose
    4. Iterate as needed: Refine approach based on intermediate results
    5. Deliver completely: Ensure output fully addresses the original goal
    6. Cite sources: Always list tools/APIs/resources used
    7. Disambiguate proximity: If multiple distinct concepts appear in the same search result or paragraph, explicitly verify which term refers to which concept before synthesizing.

    Quality Standards
    - Accuracy: Provide correct, verified information
    - Completeness: Fully address all aspects of the goal
    - Clarity: Present results in clear, structured format
    - Efficiency: Minimize unnecessary iterations or tool calls
    - Source transparency: Cite all external data sources
    - Referential Integrity: Specific terms, metaphors, or labels must be attributed strictly to their defined subjects in the source. Do not conflate distinct topics simply because they appear in close proximity.

    Common Patterns
    - Pure reasoning: No tools → think through problem → deliver answer
    - Data retrieval: Tool call → extract data → format → cite source
    - Multi-step: Plan → execute steps → synthesize → deliver
    - Iterative refinement: Initial attempt → check quality → refine → deliver

    Error Handling
    - Tool failures: Try alternatives or adapt approach
    - Incomplete data: State limitations clearly
    - Ambiguous goals: Make reasonable assumptions or ask for clarification
    - Invalid inputs: Suggest corrections or alternatives

    Output Format
    - Direct answers for simple queries
    - Structured formats (lists, tables, JSON) for complex data
    - Clear sections for multi-part answers
    - Citations at end or inline as appropriate
    """

    goal: str = dspy.InputField(description="Task that needs to be executed")
    execution_context: Optional[str] = dspy.InputField(
        default=None, description="Execution context (XML)"
    )
    context: Optional[dict] = dspy.InputField(
        default=None,
        description="File tree knowledge base: nested dict of filename -> content or subdir",
    )
    output: str = dspy.OutputField(description="Execution result")
    sources: list[str] = dspy.OutputField(description="Information sources used")


executor = dspy.RLM(ExecutorSignature)

executor_demos = [
    dspy.Example(
        goal="Explain the difference between microservices and monolithic architecture",
        output=(
            "Key differences between microservices and monolithic architecture:\n\n"
            "**Monolithic Architecture:**\n"
            "- Single, unified codebase for entire application\n"
            "- All components tightly coupled\n"
            "- Deployed as one unit\n"
            "- Scaling requires scaling entire application\n"
            "- Simpler to develop initially but harder to maintain at scale\n\n"
            "**Microservices Architecture:**\n"
            "- Application split into small, independent services\n"
            "- Each service handles specific business capability\n"
            "- Services communicate via APIs (HTTP, messaging)\n"
            "- Independent deployment and scaling per service\n"
            "- More complex initially but easier to scale and maintain\n\n"
            "**Trade-offs:**\n"
            "- Monolithic: Better for small teams, simple applications, rapid prototyping\n"
            "- Microservices: Better for large teams, complex domains, need for independent scaling"
        ),
        sources=[],
    ).with_inputs("goal"),
    dspy.Example(
        goal="Calculate compound interest for $10,000 at 5% annual rate over 10 years",
        output=(
            "Compound Interest Calculation:\n\n"
            "Principal (P): $10,000\n"
            "Annual Rate (r): 5% (0.05)\n"
            "Time (t): 10 years\n"
            "Compounding: Annual (n=1)\n\n"
            "Formula: A = P(1 + r/n)^(nt)\n"
            "A = 10,000 x (1 + 0.05/1)^(1x10)\n"
            "A = 10,000 x (1.05)^10\n"
            "A = 10,000 x 1.62889\n\n"
            "**Final Amount: $16,288.95**\n"
            "**Interest Earned: $6,288.95**"
        ),
        sources=[],
    ).with_inputs("goal"),
    dspy.Example(
        goal="Compare Python and JavaScript for backend development",
        output=(
            "Python vs JavaScript for Backend Development:\n\n"
            "**Python Advantages:**\n"
            "- Clean, readable syntax\n"
            "- Rich ecosystem for data science, ML, automation\n"
            "- Django/Flask for web development\n"
            "- Strong in scientific computing\n"
            "- Synchronous by default (simpler mental model)\n\n"
            "**JavaScript (Node.js) Advantages:**\n"
            "- Same language for frontend and backend\n"
            "- Excellent async/event-driven model\n"
            "- Large npm ecosystem\n"
            "- Fast execution (V8 engine)\n"
            "- JSON-native (web APIs)\n\n"
            "**Best Use Cases:**\n"
            "- Python: Data pipelines, ML services, scientific computing, automation\n"
            "- JavaScript: Real-time apps, microservices, full-stack JS teams, API gateways\n\n"
            "**Recommendation:**\n"
            "Choose Python for data-heavy work; JavaScript for real-time/full-stack scenarios"
        ),
        sources=[],
    ).with_inputs("goal"),
    dspy.Example(
        goal="How would you design a rate limiting system for an API?",
        output=(
            "Rate Limiting System Design:\n\n"
            "**1. Requirements:**\n"
            "- Limit requests per user/IP (e.g., 100 req/min)\n"
            "- Distribute across multiple servers\n"
            "- Low latency overhead (<5ms)\n"
            "- Handle burst traffic gracefully\n\n"
            "**2. Algorithm Choice:**\n"
            "Token Bucket (recommended):\n"
            "- Each user gets bucket with N tokens\n"
            "- Tokens refill at fixed rate (R/second)\n"
            "- Request consumes 1 token\n"
            "- Allows burst up to bucket size\n\n"
            "**3. Implementation:**\n"
            "Storage: Redis (distributed, fast)\n"
            "Key: user_id:api_key\n"
            "Value: { tokens: N, last_refill: timestamp }\n"
            "TTL: Set to prevent memory leak\n\n"
            "**4. Request Flow:**\n"
            "1. Extract user ID from request\n"
            "2. Get current token count from Redis\n"
            "3. Refill tokens based on time elapsed\n"
            "4. If tokens > 0: allow request, decrement tokens\n"
            "5. If tokens = 0: reject with 429 status\n\n"
            "**5. Edge Cases:**\n"
            "- Clock skew: Use Redis server time\n"
            "- Redis failure: Fallback to allow (fail open) or memory cache\n"
            "- Different limits per endpoint: Use separate buckets"
        ),
        sources=[],
    ).with_inputs("goal"),
]

executor.predictors()[1].demos = executor_demos
