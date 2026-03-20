"""
Tests for form_graph.py — deterministic StateGraph with HITL.

Mocks the LLM judge so tests run without API keys and are fully deterministic.
The graph's flow control (pick_question, ask_question routing) is pure code,
so we only need to mock the judge node's LLM calls.
"""

import json
import yaml
import pytest
from unittest.mock import patch, MagicMock

from langgraph.types import Command

from agents.form_graph import (
    create_form_graph,
    pick_question,
    FormState,
)
from agents.shared import JudgeResult, load_template


# ---------------------------------------------------------------------------
# Shared templates
# ---------------------------------------------------------------------------

LOAN_TEMPLATE = {
    "name": "Commercial Loan Application",
    "description": "Collect key details for commercial lending review",
    "questions": [
        {
            "id": "business_name",
            "question": "What is the legal name of the business entity applying for the loan?",
            "judge_prompt": "Must be a specific legal business name (e.g., 'Acme Corp LLC').",
        },
        {
            "id": "loan_amount",
            "question": "What is the requested loan amount and currency?",
            "judge_prompt": "Must include a specific dollar amount. Reject answers without a number.",
        },
        {
            "id": "loan_purpose",
            "question": "What is the intended use of the loan proceeds?",
            "judge_prompt": "Must describe a specific business purpose. Reject generic answers like 'business needs'.",
        },
    ],
}

SINGLE_Q_TEMPLATE = {
    "name": "Quick Check",
    "description": "Single question form",
    "questions": [
        {
            "id": "only",
            "question": "What is your account number?",
            "judge_prompt": "Must be a numeric account number, 8-12 digits.",
        },
    ],
}


def _initial_state(template=None):
    t = template or LOAN_TEMPLATE
    return {
        "template": t,
        "answers": {},
        "current_index": 0,
        "judge_feedback": "",
        "complete": False,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def loan_json(tmp_path):
    p = tmp_path / "loan.json"
    p.write_text(json.dumps(LOAN_TEMPLATE))
    return str(p)


@pytest.fixture
def loan_yaml(tmp_path):
    p = tmp_path / "loan.yaml"
    p.write_text(yaml.dump(LOAN_TEMPLATE))
    return str(p)


def _build_graph(template_path, judge_responses):
    """Build graph with a mocked judge that returns scripted responses."""
    mock_judge = MagicMock()
    mock_judge.invoke.side_effect = judge_responses
    with patch("agents.form_graph.make_judge", return_value=mock_judge):
        graph, template = create_form_graph(template_path)
    return graph, template, mock_judge


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------


def test_load_json(loan_json):
    t = load_template(loan_json)
    assert t["name"] == "Commercial Loan Application"
    assert len(t["questions"]) == 3


def test_load_yaml(loan_yaml):
    t = load_template(loan_yaml)
    assert t["name"] == "Commercial Loan Application"
    assert len(t["questions"]) == 3


def test_json_and_yaml_equivalent(loan_json, loan_yaml):
    tj = load_template(loan_json)
    ty = load_template(loan_yaml)
    assert tj["name"] == ty["name"]
    for j, y in zip(tj["questions"], ty["questions"]):
        assert j["id"] == y["id"]
        assert j["question"] == y["question"]


# ---------------------------------------------------------------------------
# pick_question (pure logic, no mock needed)
# ---------------------------------------------------------------------------


def test_pick_question_selects_first_unanswered():
    state: FormState = {
        "template": LOAN_TEMPLATE,
        "answers": {},
        "current_index": 99,
        "judge_feedback": "old",
        "complete": False,
    }
    cmd = pick_question(state)
    assert cmd.goto == "ask_question"
    assert cmd.update["current_index"] == 0
    assert cmd.update["judge_feedback"] == ""


def test_pick_question_skips_answered():
    state: FormState = {
        "template": LOAN_TEMPLATE,
        "answers": {"business_name": "Acme Corp LLC"},
        "current_index": 0,
        "judge_feedback": "",
        "complete": False,
    }
    cmd = pick_question(state)
    assert cmd.goto == "ask_question"
    assert cmd.update["current_index"] == 1  # loan_amount


def test_pick_question_all_done():
    state: FormState = {
        "template": LOAN_TEMPLATE,
        "answers": {
            "business_name": "Acme Corp LLC",
            "loan_amount": "$2.5M USD",
            "loan_purpose": "Equipment purchase",
        },
        "current_index": 0,
        "judge_feedback": "",
        "complete": False,
    }
    cmd = pick_question(state)
    assert cmd.goto == "__end__"
    assert cmd.update["complete"] is True


# ---------------------------------------------------------------------------
# Full graph: first interrupt
# ---------------------------------------------------------------------------


def test_first_interrupt_asks_business_name(loan_json):
    graph, _, _ = _build_graph(loan_json, [])
    config = {"configurable": {"thread_id": "t-first"}}

    result = graph.invoke(_initial_state(), config)

    interrupts = result.get("__interrupt__", [])
    assert len(interrupts) == 1
    assert interrupts[0].value["question_id"] == "business_name"
    assert "legal name" in interrupts[0].value["question"].lower()
    assert "previous_feedback" not in interrupts[0].value


# ---------------------------------------------------------------------------
# Full graph: happy path — all answers sufficient on first try
# ---------------------------------------------------------------------------


def test_happy_path_loan_application(loan_json):
    responses = [
        JudgeResult(sufficient=True, feedback="Valid business name"),
        JudgeResult(sufficient=True, feedback="Clear amount"),
        JudgeResult(sufficient=True, feedback="Specific purpose"),
    ]
    graph, _, mock = _build_graph(loan_json, responses)
    config = {"configurable": {"thread_id": "t-happy"}}

    result = graph.invoke(_initial_state(), config)
    assert result["__interrupt__"][0].value["question_id"] == "business_name"

    result = graph.invoke(Command(resume="Meridian Capital Partners LLC"), config)
    assert result["__interrupt__"][0].value["question_id"] == "loan_amount"

    result = graph.invoke(Command(resume="$5M USD"), config)
    assert result["__interrupt__"][0].value["question_id"] == "loan_purpose"

    result = graph.invoke(Command(resume="Commercial real estate acquisition in downtown Chicago"), config)
    assert result.get("complete") is True
    assert result["answers"] == {
        "business_name": "Meridian Capital Partners LLC",
        "loan_amount": "$5M USD",
        "loan_purpose": "Commercial real estate acquisition in downtown Chicago",
    }
    assert mock.invoke.call_count == 3


# ---------------------------------------------------------------------------
# Full graph: vague loan amount rejected, specific one accepted
# ---------------------------------------------------------------------------


def test_vague_loan_amount_rejected(loan_json):
    responses = [
        JudgeResult(sufficient=True, feedback="OK"),
        JudgeResult(sufficient=False, feedback="No dollar amount provided. Must include a specific number like '$2.5M'."),
        JudgeResult(sufficient=True, feedback="Clear amount"),
        JudgeResult(sufficient=True, feedback="OK"),
    ]
    graph, _, mock = _build_graph(loan_json, responses)
    config = {"configurable": {"thread_id": "t-vague-amount"}}

    # Q1: business_name — accepted
    result = graph.invoke(_initial_state(), config)
    result = graph.invoke(Command(resume="Acme Corp LLC"), config)
    assert result["__interrupt__"][0].value["question_id"] == "loan_amount"

    # Q2: vague answer — rejected with feedback
    result = graph.invoke(Command(resume="a lot of money"), config)
    interrupt_val = result["__interrupt__"][0].value
    assert interrupt_val["question_id"] == "loan_amount"
    assert "previous_feedback" in interrupt_val
    assert "dollar amount" in interrupt_val["previous_feedback"].lower()

    # Q2: specific answer — accepted
    result = graph.invoke(Command(resume="$2.5M USD"), config)
    assert result["__interrupt__"][0].value["question_id"] == "loan_purpose"

    # Q3: accepted
    result = graph.invoke(Command(resume="Working capital for seasonal inventory"), config)
    assert result.get("complete") is True
    assert result["answers"]["loan_amount"] == "$2.5M USD"


# ---------------------------------------------------------------------------
# Full graph: multiple rejections on loan purpose
# ---------------------------------------------------------------------------


def test_generic_purpose_rejected_twice(loan_json):
    responses = [
        JudgeResult(sufficient=True, feedback="OK"),
        JudgeResult(sufficient=True, feedback="OK"),
        JudgeResult(sufficient=False, feedback="Too generic. 'Business needs' is not specific enough."),
        JudgeResult(sufficient=False, feedback="Still vague. Describe what specifically the funds will be used for."),
        JudgeResult(sufficient=True, feedback="Specific and clear"),
    ]
    graph, _, mock = _build_graph(loan_json, responses)
    config = {"configurable": {"thread_id": "t-purpose-reject"}}

    result = graph.invoke(_initial_state(), config)
    result = graph.invoke(Command(resume="Acme Corp LLC"), config)
    result = graph.invoke(Command(resume="$1M"), config)
    assert result["__interrupt__"][0].value["question_id"] == "loan_purpose"

    # Attempt 1: too generic
    result = graph.invoke(Command(resume="business needs"), config)
    assert "generic" in result["__interrupt__"][0].value["previous_feedback"].lower()

    # Attempt 2: still vague
    result = graph.invoke(Command(resume="growth"), config)
    assert "vague" in result["__interrupt__"][0].value["previous_feedback"].lower()

    # Attempt 3: specific — accepted
    result = graph.invoke(Command(resume="Purchase of CNC milling equipment for manufacturing expansion"), config)
    assert result.get("complete") is True
    assert mock.invoke.call_count == 5


# ---------------------------------------------------------------------------
# Full graph: single-question form
# ---------------------------------------------------------------------------


def test_single_question_form(tmp_path):
    p = tmp_path / "single.json"
    p.write_text(json.dumps(SINGLE_Q_TEMPLATE))

    responses = [JudgeResult(sufficient=True, feedback="Valid account number")]
    graph, _, _ = _build_graph(str(p), responses)
    config = {"configurable": {"thread_id": "t-single"}}

    result = graph.invoke(_initial_state(SINGLE_Q_TEMPLATE), config)
    assert result["__interrupt__"][0].value["question_id"] == "only"

    result = graph.invoke(Command(resume="12345678"), config)
    assert result.get("complete") is True
    assert result["answers"]["only"] == "12345678"


# ---------------------------------------------------------------------------
# Full graph: empty answer goes through judge (not skipped)
# ---------------------------------------------------------------------------


def test_empty_answer_judged_not_skipped(loan_json):
    responses = [
        JudgeResult(sufficient=False, feedback="Answer is empty, provide a business name"),
        JudgeResult(sufficient=True, feedback="OK"),
        JudgeResult(sufficient=True, feedback="OK"),
        JudgeResult(sufficient=True, feedback="OK"),
    ]
    graph, _, mock = _build_graph(loan_json, responses)
    config = {"configurable": {"thread_id": "t-empty"}}

    result = graph.invoke(_initial_state(), config)

    # Empty string → judge still called, rejects it
    result = graph.invoke(Command(resume=""), config)
    assert result["__interrupt__"][0].value["question_id"] == "business_name"
    assert "empty" in result["__interrupt__"][0].value["previous_feedback"].lower()

    # Real answer
    result = graph.invoke(Command(resume="Acme Corp LLC"), config)
    result = graph.invoke(Command(resume="$1M"), config)
    result = graph.invoke(Command(resume="Equipment"), config)
    assert result.get("complete") is True
    assert mock.invoke.call_count == 4


# ---------------------------------------------------------------------------
# Full graph: answers preserved exactly as given
# ---------------------------------------------------------------------------


def test_answers_preserved_verbatim(loan_json):
    responses = [
        JudgeResult(sufficient=True, feedback="OK"),
        JudgeResult(sufficient=True, feedback="OK"),
        JudgeResult(sufficient=True, feedback="OK"),
    ]
    graph, _, _ = _build_graph(loan_json, responses)
    config = {"configurable": {"thread_id": "t-verbatim"}}

    result = graph.invoke(_initial_state(), config)
    result = graph.invoke(Command(resume="  Acme Corp LLC  "), config)
    result = graph.invoke(Command(resume="$2,500,000.00 USD"), config)
    result = graph.invoke(Command(resume="Working capital\nfor seasonal inventory"), config)

    # Whitespace, formatting, newlines preserved — judge decides validity
    assert result["answers"]["business_name"] == "  Acme Corp LLC  "
    assert result["answers"]["loan_amount"] == "$2,500,000.00 USD"
    assert "\n" in result["answers"]["loan_purpose"]
