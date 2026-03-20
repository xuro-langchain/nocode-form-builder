# Form Agent

A form-filling agent with LLM-as-Judge validation, powered by LangGraph. Includes a Streamlit UI and CLI runners for building templates and filling out forms interactively.

## Setup

```bash
uv sync
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

## Streamlit UI

```bash
uv run streamlit run frontend/app.py
```

Two tabs:

- **Template Builder** — Create or edit form templates (questions + judge prompts). Load existing templates from `templates/`, add/remove questions, and save as JSON or YAML.
- **Form Filler** — Select a template, answer questions one at a time, and get real-time LLM judge validation. Rejected answers show feedback; accepted answers advance to the next question.

## CLI runners

### StateGraph agent (deterministic, HITL interrupts)

```bash
uv run python -m agents.form_graph                              # defaults to templates/loan_application.json
uv run python -m agents.form_graph templates/my_template.yaml   # custom template
```

### create_agent agent (conversational, middleware-based)

```bash
uv run python -m agents.form_agent                              # defaults to templates/loan_application.json
uv run python -m agents.form_agent templates/my_template.yaml   # custom template
```

## Project structure

```
agents/
  shared.py          # load_template(), JudgeResult (shared utilities)
  form_graph.py      # StateGraph with interrupt()/Command(resume=) HITL loop
  form_agent.py      # create_agent with validate_answer tool + middleware
templates/
  loan_application.json
  loan_application.yaml
frontend/
  app.py             # Streamlit UI (Template Builder + Form Filler)
tests/
  test_form_graph.py
  test_form_agent.py
```

## Tests

```bash
uv run pytest tests/ -v
```

All tests use mocked LLM calls — no API key required.
