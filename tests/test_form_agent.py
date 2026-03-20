"""
Tests for form_agent.py — create_agent with custom middleware.

Tests the tools and middleware directly with mocked LLM judge.
The agent loop is opaque (create_agent), so we focus on the components:
validate_answer, get_form_status, prompt building, template loading.
"""

import json
import yaml
import pytest
from unittest.mock import patch, MagicMock

from agents.form_agent import (
    make_form_tools_and_middleware,
    build_system_prompt,
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
            "judge_prompt": "Must describe a specific business purpose. Reject generic answers.",
        },
    ],
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


def _make_tools(template, judge_responses):
    """Build tools with a mocked judge returning scripted responses."""
    mock_judge = MagicMock()
    mock_judge.invoke.side_effect = judge_responses

    with patch("agents.form_agent.ChatOpenAI") as MockChatOpenAI:
        instance = MockChatOpenAI.return_value
        instance.with_structured_output.return_value = mock_judge
        tools, middleware = make_form_tools_and_middleware(template)

    validate_answer = tools[0]
    get_form_status = tools[1]
    return validate_answer, get_form_status, mock_judge


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


# ---------------------------------------------------------------------------
# validate_answer: sufficient answers
# ---------------------------------------------------------------------------


def test_validate_business_name_sufficient():
    validate, _, mock = _make_tools(LOAN_TEMPLATE, [
        JudgeResult(sufficient=True, feedback="Valid legal entity name"),
    ])

    result = json.loads(validate.invoke({
        "question_id": "business_name",
        "answer": "Meridian Capital Partners LLC",
    }))

    assert result["sufficient"] is True
    assert result["questions_remaining"] == 2
    assert mock.invoke.call_count == 1


def test_validate_loan_amount_sufficient():
    validate, _, _ = _make_tools(LOAN_TEMPLATE, [
        JudgeResult(sufficient=True, feedback="Clear dollar amount"),
    ])

    result = json.loads(validate.invoke({
        "question_id": "loan_amount",
        "answer": "$2.5M USD",
    }))

    assert result["sufficient"] is True
    assert result["questions_remaining"] == 2


# ---------------------------------------------------------------------------
# validate_answer: insufficient answers
# ---------------------------------------------------------------------------


def test_validate_vague_business_name_rejected():
    validate, _, _ = _make_tools(LOAN_TEMPLATE, [
        JudgeResult(sufficient=False, feedback="'my company' is not a legal entity name"),
    ])

    result = json.loads(validate.invoke({
        "question_id": "business_name",
        "answer": "my company",
    }))

    assert result["sufficient"] is False
    assert "legal entity" in result["feedback"]
    assert result["questions_remaining"] == 3  # nothing accepted


def test_validate_no_dollar_amount_rejected():
    validate, _, _ = _make_tools(LOAN_TEMPLATE, [
        JudgeResult(sufficient=False, feedback="No dollar amount provided"),
    ])

    result = json.loads(validate.invoke({
        "question_id": "loan_amount",
        "answer": "a lot",
    }))

    assert result["sufficient"] is False
    assert result["questions_remaining"] == 3


def test_validate_generic_purpose_rejected():
    validate, _, _ = _make_tools(LOAN_TEMPLATE, [
        JudgeResult(sufficient=False, feedback="'business needs' is too generic"),
    ])

    result = json.loads(validate.invoke({
        "question_id": "loan_purpose",
        "answer": "business needs",
    }))

    assert result["sufficient"] is False


# ---------------------------------------------------------------------------
# validate_answer: edge cases
# ---------------------------------------------------------------------------


def test_validate_unknown_question_id():
    validate, _, _ = _make_tools(LOAN_TEMPLATE, [])

    result = json.loads(validate.invoke({
        "question_id": "nonexistent",
        "answer": "foo",
    }))

    assert "error" in result
    assert "nonexistent" in result["error"]


def test_validate_empty_answer_goes_to_judge():
    validate, _, mock = _make_tools(LOAN_TEMPLATE, [
        JudgeResult(sufficient=False, feedback="Answer is empty"),
    ])

    result = json.loads(validate.invoke({
        "question_id": "business_name",
        "answer": "",
    }))

    assert result["sufficient"] is False
    assert mock.invoke.call_count == 1  # judge was still called


# ---------------------------------------------------------------------------
# get_form_status
# ---------------------------------------------------------------------------


def test_status_initially_all_remaining():
    _, status, _ = _make_tools(LOAN_TEMPLATE, [])

    result = json.loads(status.invoke({}))

    assert result["complete"] is False
    assert len(result["remaining"]) == 3
    assert result["answered"] == {}


def test_status_tracks_accepted_answers():
    validate, status, _ = _make_tools(LOAN_TEMPLATE, [
        JudgeResult(sufficient=True, feedback="OK"),
    ])

    validate.invoke({"question_id": "business_name", "answer": "Acme Corp LLC"})
    result = json.loads(status.invoke({}))

    assert result["answered"] == {"business_name": "Acme Corp LLC"}
    assert len(result["remaining"]) == 2
    assert result["complete"] is False


def test_status_complete_after_all_accepted():
    validate, status, _ = _make_tools(LOAN_TEMPLATE, [
        JudgeResult(sufficient=True, feedback="OK"),
        JudgeResult(sufficient=True, feedback="OK"),
        JudgeResult(sufficient=True, feedback="OK"),
    ])

    validate.invoke({"question_id": "business_name", "answer": "Acme Corp LLC"})
    validate.invoke({"question_id": "loan_amount", "answer": "$5M"})
    validate.invoke({"question_id": "loan_purpose", "answer": "Equipment purchase"})

    result = json.loads(status.invoke({}))
    assert result["complete"] is True
    assert len(result["remaining"]) == 0


def test_rejected_answer_not_tracked():
    """Insufficient answers should NOT appear in form progress."""
    validate, status, _ = _make_tools(LOAN_TEMPLATE, [
        JudgeResult(sufficient=False, feedback="Bad"),
        JudgeResult(sufficient=True, feedback="Good"),
    ])

    # Rejected — should NOT appear
    validate.invoke({"question_id": "business_name", "answer": "bad"})
    result = json.loads(status.invoke({}))
    assert "business_name" not in result["answered"]

    # Accepted — should appear
    validate.invoke({"question_id": "business_name", "answer": "Acme Corp LLC"})
    result = json.loads(status.invoke({}))
    assert result["answered"]["business_name"] == "Acme Corp LLC"


# ---------------------------------------------------------------------------
# questions_remaining count
# ---------------------------------------------------------------------------


def test_remaining_decrements_on_acceptance():
    validate, _, _ = _make_tools(LOAN_TEMPLATE, [
        JudgeResult(sufficient=True, feedback="OK"),
        JudgeResult(sufficient=True, feedback="OK"),
        JudgeResult(sufficient=True, feedback="OK"),
    ])

    r1 = json.loads(validate.invoke({"question_id": "business_name", "answer": "X"}))
    assert r1["questions_remaining"] == 2

    r2 = json.loads(validate.invoke({"question_id": "loan_amount", "answer": "$1M"}))
    assert r2["questions_remaining"] == 1

    r3 = json.loads(validate.invoke({"question_id": "loan_purpose", "answer": "Equip"}))
    assert r3["questions_remaining"] == 0


def test_remaining_unchanged_on_rejection():
    validate, _, _ = _make_tools(LOAN_TEMPLATE, [
        JudgeResult(sufficient=False, feedback="Bad"),
    ])

    r = json.loads(validate.invoke({"question_id": "business_name", "answer": "bad"}))
    assert r["questions_remaining"] == 3


# ---------------------------------------------------------------------------
# build_system_prompt
# ---------------------------------------------------------------------------


def test_system_prompt_contains_loan_info():
    prompt = build_system_prompt(LOAN_TEMPLATE)

    assert "Commercial Loan Application" in prompt
    assert "business_name" in prompt
    assert "loan_amount" in prompt
    assert "loan_purpose" in prompt
    assert "legal name" in prompt.lower()


def test_system_prompt_contains_rules():
    prompt = build_system_prompt(LOAN_TEMPLATE)

    assert "ONE AT A TIME" in prompt
    assert "validate_answer" in prompt
    assert "rephrase" in prompt.lower() or "Rephrase" in prompt


# ---------------------------------------------------------------------------
# Judge receives correct context per question
# ---------------------------------------------------------------------------


def test_judge_receives_business_name_criteria():
    validate, _, mock = _make_tools(LOAN_TEMPLATE, [
        JudgeResult(sufficient=True, feedback="OK"),
    ])

    validate.invoke({"question_id": "business_name", "answer": "Acme Corp LLC"})

    msgs = mock.invoke.call_args[0][0]
    assert "legal business name" in msgs[0]["content"].lower()
    assert "Acme Corp LLC" in msgs[1]["content"]


def test_judge_receives_loan_amount_criteria():
    validate, _, mock = _make_tools(LOAN_TEMPLATE, [
        JudgeResult(sufficient=True, feedback="OK"),
    ])

    validate.invoke({"question_id": "loan_amount", "answer": "$5M"})

    msgs = mock.invoke.call_args[0][0]
    assert "dollar amount" in msgs[0]["content"].lower()
    assert "$5M" in msgs[1]["content"]
