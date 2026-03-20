"""
Deterministic form-filling agent using StateGraph with HITL interrupts.

Two pieces:
1. Template = JSON/YAML config defining questions + LLM-as-Judge prompts
2. Graph = deterministic loop that collects answers via interrupt(), validates
   with LLM-as-Judge, and re-asks (with feedback) until sufficient.

Flow:
  pick_question -> ask_question -> judge -> [sufficient?]
                                             yes -> pick_question
                                             no  -> ask_question (with feedback)
  pick_question (all done) -> END

The graph controls all flow. The LLM only runs inside the judge node.
Single invocation with Command(resume=answer) for each user response.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from typing import Literal
from typing_extensions import TypedDict

from agents.shared import load_template, JudgeResult

load_dotenv()

# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class FormState(TypedDict):
    template: dict                   # the loaded template
    answers: dict                    # {question_id: accepted_answer}
    current_index: int               # index of the question being asked
    judge_feedback: str              # feedback from last failed validation
    complete: bool                   # whether all questions are done
    last_answer: str                 # holds the answer between ask and judge


# ---------------------------------------------------------------------------
# LLM-as-Judge
# ---------------------------------------------------------------------------

def make_judge(model: str = "gpt-4o-mini") -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=0).with_structured_output(JudgeResult)


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def pick_question(state: FormState) -> Command[Literal["ask_question", "__end__"]]:
    """Deterministically find the next unanswered question. Pure code, no LLM."""
    questions = state["template"]["questions"]
    answers = state.get("answers", {})

    for i, q in enumerate(questions):
        if q["id"] not in answers:
            return Command(
                update={"current_index": i, "judge_feedback": ""},
                goto="ask_question",
            )

    # All questions answered
    return Command(update={"complete": True}, goto=END)


def ask_question(state: FormState) -> Command[Literal["judge"]]:
    """Interrupt to ask the user the current question. Collects their answer."""
    questions = state["template"]["questions"]
    q = questions[state["current_index"]]
    feedback = state.get("judge_feedback", "")

    prompt = {"question": q["question"], "question_id": q["id"]}
    if feedback:
        prompt["previous_feedback"] = feedback
        prompt["hint"] = "Your previous answer wasn't sufficient. See the feedback above and try again."

    # Pause execution — surfaces prompt to caller, resumes with user's answer
    answer = interrupt(prompt)

    return Command(
        update={"last_answer": answer},
        goto="judge",
    )


def make_judge_node(judge_model: str = "gpt-4o-mini"):
    """Create the judge node with a configurable model."""
    judge = make_judge(judge_model)

    def judge_node(state: FormState) -> Command[Literal["pick_question", "ask_question"]]:
        """Run LLM-as-Judge on the user's answer. Only LLM call in the graph."""
        questions = state["template"]["questions"]
        q = questions[state["current_index"]]
        answer = state.get("last_answer", "")

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
            # Accept the answer, move to next question
            answers = {**state.get("answers", {}), q["id"]: answer}
            return Command(
                update={"answers": answers, "judge_feedback": ""},
                goto="pick_question",
            )
        else:
            # Reject — loop back to ask_question with feedback
            return Command(
                update={"judge_feedback": result.feedback},
                goto="ask_question",
            )

    return judge_node


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def create_form_graph(template_path: str, judge_model: str = "gpt-4o-mini"):
    """Build the deterministic form graph from a template file."""
    template = load_template(template_path)
    judge_node = make_judge_node(judge_model)

    graph = StateGraph(FormState)
    graph.add_node("pick_question", pick_question)
    graph.add_node("ask_question", ask_question)
    graph.add_node("judge", judge_node)

    graph.add_edge(START, "pick_question")
    # All other routing is handled by Command(goto=...) in nodes

    compiled = graph.compile(checkpointer=InMemorySaver())
    return compiled, template


# ---------------------------------------------------------------------------
# Interactive runner
# ---------------------------------------------------------------------------

def print_form_summary(state: dict, template: dict):
    """Print a summary of the completed form."""
    print("\n" + "=" * 50)
    print(f"  {template['name']} — COMPLETE")
    print("=" * 50)
    for q in template["questions"]:
        print(f"\n  {q['question']}")
        print(f"  → {state['answers'].get(q['id'], '(unanswered)')}")
    print("\n" + "=" * 50)


def run_interactive(template_path: str, thread_id: str = "form-session-1"):
    """Run the form interactively. Single graph invocation with interrupt/resume."""
    graph, template = create_form_graph(template_path)
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n--- {template['name']} ---")
    print(f"    {template.get('description', '')}")
    print(f"    ({len(template['questions'])} questions)\n")

    # First invocation — graph runs until first interrupt
    result = graph.invoke(
        {
            "template": template,
            "answers": {},
            "current_index": 0,
            "judge_feedback": "",
            "complete": False,
        },
        config=config,
    )

    while True:
        # Check if complete (no interrupt)
        interrupts = result.get("__interrupt__", [])
        if not interrupts:
            print_form_summary(result, template)
            break

        # Display the interrupt prompt
        prompt = interrupts[0].value
        if "previous_feedback" in prompt:
            print(f"  [Feedback: {prompt['previous_feedback']}]")
            print(f"  {prompt['hint']}")
        print(f"\n  Q: {prompt['question']}")

        user_input = input("  A: ").strip()
        if user_input.lower() in ("quit", "exit"):
            print("Exiting.")
            break

        # Resume with the user's answer
        result = graph.invoke(Command(resume=user_input), config=config)


if __name__ == "__main__":
    template = sys.argv[1] if len(sys.argv) > 1 else "templates/loan_application.json"
    run_interactive(template)
