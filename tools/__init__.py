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

import asyncio
import math
import os
import signal

MAX_TERMINAL_OUTPUT_LEN = 21_500


def calculate_token_length(text: str) -> int:
    """
    Estimates token count using Google's gemini-cli heuristic.
    - CJK characters = ~1 token
    - Other characters = ~0.25 tokens
    - Massive strings (>1M chars) = len / 4 fallback
    """
    if not text:
        return 0

    if len(text) > 1_000_000:
        return int(len(text) / 4)

    token_count = 0.0
    for char in text:
        code = ord(char)
        is_cjk = (
            (0x4E00 <= code <= 0x9FFF)
            or (0x3040 <= code <= 0x309F)
            or (0x30A0 <= code <= 0x30FF)
            or (0xAC00 <= code <= 0xD7AF)
        )
        token_count += 1.0 if is_cjk else 0.25

    return math.floor(token_count)


async def terminal(command: str):
    """
    Execute a shell command asynchronously via /bin/bash with timeout cleanup.
    """
    print(f"[Running Shell]: {command}")

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            preexec_fn=os.setsid,
            executable="/bin/bash",
        )

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)

            output = stdout.decode().strip() or stderr.decode().strip()

            if not output:
                return "Command executed successfully (no output)."

            if calculate_token_length(output) > MAX_TERMINAL_OUTPUT_LEN:
                original_len = calculate_token_length(output)
                output = output[:MAX_TERMINAL_OUTPUT_LEN]
                output += (
                    f"\n... [Output truncated. Original length: {original_len} chars]"
                )

            print(f"[Command Output]: {output[:200]}...")
            return output

        except asyncio.TimeoutError:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass

            return "[ERROR]: Command timed out after 30 seconds."

    except Exception as e:
        print(f"[ERROR in terminal]: {type(e).__name__}: {e}")
        return f"Error executing command: {e}"
