"""
Simple form-filling agent using create_agent.

Same form template, simpler code. The agent drives the conversation naturally,
validates answers via tools, and tracks form progress.

Multi-turn: each invoke is one exchange (agent asks, user answers, agent validates).
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from agents.shared import load_template, JudgeResult

load_dotenv()

# ---------------------------------------------------------------------------
# Tools + middleware
# ---------------------------------------------------------------------------

def make_form_tools_and_middleware(template: dict, judge_model: str = "gpt-4o-mini"):
    """Create the validate_answer tool and form-tracking middleware."""

    judge = ChatOpenAI(model=judge_model, temperature=0).with_structured_output(JudgeResult)

    # Shared state for tracking — middleware reads this after each validation
    form_progress = {"answered": {}, "total": len(template["questions"])}

    @tool
    def validate_answer(question_id: str, answer: str) -> str:
        """Validate the respondent's answer for a specific form question.

        Call this after the respondent answers a question. The judge will
        evaluate against the template's acceptance criteria.

        Args:
            question_id: The id of the question being validated
            answer: The respondent's answer to evaluate
        """
        # Find the question in the template
        q = next((q for q in template["questions"] if q["id"] == question_id), None)
        if not q:
            return json.dumps({"error": f"Unknown question_id: {question_id}"})

        result: JudgeResult = judge.invoke([
            {
                "role": "system",
                "content": (
                    "You are a strict but fair judge evaluating form responses.\n"
                    "Evaluate ONLY against the provided criteria.\n"
                    f"Criteria: {q['judge_prompt']}"
                ),
            },
            {
                "role": "user",
                "content": f"Question: {q['question']}\nAnswer: {answer}",
            },
        ])

        if result.sufficient:
            form_progress["answered"][question_id] = answer

        return json.dumps({
            "sufficient": result.sufficient,
            "feedback": result.feedback,
            "questions_remaining": form_progress["total"] - len(form_progress["answered"]),
        })

    @tool
    def get_form_status() -> str:
        """Check which questions have been answered and which remain.

        Call this to see the current state of the form.
        """
        answered_ids = set(form_progress["answered"].keys())
        remaining = [
            {"id": q["id"], "question": q["question"]}
            for q in template["questions"]
            if q["id"] not in answered_ids
        ]
        return json.dumps({
            "answered": form_progress["answered"],
            "remaining": remaining,
            "complete": len(remaining) == 0,
        })

    # Middleware: enforce that questions are answered in template order
    question_ids = [q["id"] for q in template["questions"]]

    @wrap_tool_call
    def enforce_order_middleware(request, handler):
        tc = request.tool_call
        if tc["name"] == "validate_answer":
            qid = tc["args"].get("question_id")
            answered_ids = set(form_progress["answered"].keys())

            for expected_id in question_ids:
                if expected_id not in answered_ids:
                    if qid != expected_id:
                        return (
                            f"Cannot validate '{qid}' yet. "
                            f"Answer '{expected_id}' first. "
                            f"Questions must be answered in order."
                        )
                    break

        return handler(request)

    return [validate_answer, get_form_status], [enforce_order_middleware]


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

def build_system_prompt(template: dict) -> str:
    questions_block = ""
    for i, q in enumerate(template["questions"], 1):
        questions_block += f"\n{i}. id=\"{q['id']}\" — {q['question']}\n"

    return f"""You are a form-filling assistant for: "{template['name']}"
Description: {template.get('description', '')}

## Rules
1. Ask questions ONE AT A TIME, in order. Start with the first unanswered question.
2. After the respondent answers, call `validate_answer` with their answer and the question_id.
3. If sufficient: acknowledge briefly, then ask the next question.
4. If NOT sufficient: share the feedback, rephrase the question with guidance, and give a concrete example of a good answer. Do NOT repeat the same question verbatim.
5. When all questions pass validation, call `get_form_status` to confirm, then output a clean summary.
6. Be conversational but focused. Don't skip ahead or ask multiple questions at once.

## Questions (ask in this order)
{questions_block}
"""


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def create_form_agent(
    template_path: str,
    model: str = "openai:gpt-4o",
    judge_model: str = "gpt-4o-mini",
):
    """Create a form-filling agent from a template file."""
    template = load_template(template_path)
    tools, middleware = make_form_tools_and_middleware(template, judge_model)

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=build_system_prompt(template),
        checkpointer=MemorySaver(),
        middleware=middleware,
    )
    return agent, template


# ---------------------------------------------------------------------------
# Interactive runner
# ---------------------------------------------------------------------------

def run_interactive(template_path: str, thread_id: str = "form-session-1"):
    """Run the form agent interactively. Multi-turn conversation."""
    agent, template = create_form_agent(template_path)
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n--- {template['name']} ---")
    print(f"    {template.get('description', '')}")
    print(f"    ({len(template['questions'])} questions)\n")

    # Kick off — agent asks the first question
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "I'm ready to fill out the form."}]},
        config=config,
    )
    print(f"\nAgent: {result['messages'][-1].content}\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Exiting.")
            break

        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )
        print(f"\nAgent: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    template = sys.argv[1] if len(sys.argv) > 1 else "templates/loan_application.json"
    run_interactive(template)
