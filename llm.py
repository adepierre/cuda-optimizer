import json
import os
from dataclasses import dataclass
from typing import Optional

import openai
import config


@dataclass
class SubmitResult:
    code: str
    commit_message: Optional[str] = None
    summary: Optional[str] = None
    tool_call_id: str = ""
    tool_call_message: Optional[dict] = None
    usage: Optional[dict] = None


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "submit_code",
            "description": "Submit the complete new impl.cu file with a commit message and summary. Use this for the initial optimization proposal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Complete new content for impl.cu.",
                    },
                    "commit_message": {
                        "type": "string",
                        "description": "Short one-liner (<=72 chars) describing the change.",
                    },
                    "summary": {
                        "type": "string",
                        "description": "2-5 sentences: bottleneck identified, what was changed, expected outcome.",
                    },
                },
                "required": ["code", "commit_message", "summary"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fix_code",
            "description": "Submit a fixed version of impl.cu after a build or validation error. Only provide the corrected code; no commit message or summary needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Complete fixed content for impl.cu.",
                    },
                },
                "required": ["code"],
            },
        },
    },
]


def system_prompt() -> str:
    prompt_template = config.SYSTEM_PROMPT_PATH.read_text()
    interface_base = (config.COMMON_DIR / "interface_base.hpp").read_text()
    interface_specific = (config.COMMON_DIR / "interface_specific.hpp").read_text()

    parts = [prompt_template]

    parts.append("## interface_base.hpp\n```cpp\n" + interface_base + "```\n")
    parts.append("## interface_specific.hpp\n```cpp\n" + interface_specific + "```\n")

    return "\n".join(parts)


def build_user_message(
    impl_source: str,
    best_timing: dict,
    history: list[dict],
    ncu_metrics: str,
    retry_error: Optional[str] = None,
) -> str:
    parts = []

    if history:
        parts.append("## Iteration history\n")
        for entry in history:
            status = entry.get("status", "unknown")
            timing_ms = entry.get("timing_ms")
            max_error = entry.get("max_error")
            summary = entry.get("summary", "")
            timing_str = f"{timing_ms} ms" if timing_ms is not None else "N/A"
            err_str = f", max_error={max_error}" if max_error is not None else ""
            parts.append(f"- Iter {entry.get('iter', '?')} [{status}] {timing_str}{err_str} - {summary}")
        parts.append("")

    parts.append("## NCU metrics\n")
    parts.append(ncu_metrics)
    parts.append("")

    parts.append("## Current impl.cu\n```cpp\n" + impl_source + "```\n")

    parts.append("## Best timing so far\n")
    for k, v in best_timing.items():
        if k == "per_seed" and isinstance(v, list):
            for seed, val in v:
                parts.append(f"- seed={seed}: {val} ms")
        else:
            parts.append(f"- {k}: {v} ms")
    parts.append("")

    if retry_error:
        parts.append("## Your last submission failed, fix it rather than trying something new:\n" + retry_error[:2000])

    return "\n".join(parts)


def run_turn(messages: list[dict]) -> SubmitResult:
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or "sk-no-key-required", base_url=config.OPENAI_BASE_URL)

    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="required",
        timeout=3600,
    )

    usage = None
    if response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        print(
            f"[LLM] tokens: {usage['total_tokens']} "
            f"({usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion)"
        )

    tool_calls = response.choices[0].message.tool_calls

    if not tool_calls:
        raise ValueError("No tool calls in response")

    tool_call = tool_calls[0]
    args = json.loads(tool_call.function.arguments)

    tool_name = tool_call.function.name

    if tool_name == "submit_code":
        commit_message = args.get("commit_message")
        summary = args.get("summary")
    elif tool_name == "fix_code":
        commit_message = None
        summary = None
    else:
        raise ValueError(f"Unknown tool: {tool_name}")

    assistant_msg = response.choices[0].message

    return SubmitResult(
        code=args["code"],
        commit_message=commit_message,
        summary=summary,
        tool_call_id=tool_call.id,
        tool_call_message={
            "role": "assistant",
            "content": assistant_msg.content,
            "tool_calls": assistant_msg.tool_calls,
        },
        usage=usage,
    )


def apply(result: SubmitResult) -> tuple[bool, str]:
    try:
        config.IMPL_PATH.write_text(result.code, encoding="utf-8")
        return True, ""
    except OSError as e:
        return False, str(e)
