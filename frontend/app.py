"""
Streamlit UI for the form-filling agent.

Tab 1: Template Builder — create/edit form templates (questions + judge prompts)
Tab 2: Form Filler — interact via Graph (HITL interrupts) or Agent (chat-based)
"""

import json
import yaml
import streamlit as st
from pathlib import Path

from agents.form_graph import create_form_graph
from agents.form_agent import create_form_agent
from agents.shared import load_template
from langgraph.types import Command

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_templates() -> list[str]:
    """Return template filenames from templates/ directory."""
    if not TEMPLATES_DIR.exists():
        return []
    return sorted(
        f.name for f in TEMPLATES_DIR.iterdir()
        if f.suffix in (".json", ".yaml", ".yml")
    )


def _clear_ff_state():
    """Remove all Form Filler session state keys."""
    for key in list(st.session_state.keys()):
        if key.startswith("ff_") and key != "ff_mode":
            del st.session_state[key]


# ---------------------------------------------------------------------------
# Tab 1: Template Builder
# ---------------------------------------------------------------------------

def template_builder_tab():
    st.header("Template Builder")

    # --- Load existing template ---
    templates = list_templates()
    load_col, _ = st.columns([2, 3])
    with load_col:
        load_choice = st.selectbox(
            "Load existing template",
            ["(new template)"] + templates,
            key="tb_load",
        )

    if load_choice != "(new template)" and st.session_state.get("_tb_last_loaded") != load_choice:
        t = load_template(str(TEMPLATES_DIR / load_choice))
        st.session_state["tb_name"] = t.get("name", "")
        st.session_state["tb_desc"] = t.get("description", "")
        questions = t.get("questions", [])
        st.session_state["tb_questions"] = questions
        st.session_state["_tb_last_loaded"] = load_choice
        # Clear old widget keys, then pre-populate from loaded data
        for k in list(st.session_state.keys()):
            if k.startswith(("q_id_", "q_text_", "q_judge_")):
                del st.session_state[k]
        for i, q in enumerate(questions):
            st.session_state[f"q_id_{i}"] = q.get("id", "")
            st.session_state[f"q_text_{i}"] = q.get("question", "")
            st.session_state[f"q_judge_{i}"] = q.get("judge_prompt", "")
        st.rerun()

    # Initialize defaults
    if "tb_name" not in st.session_state:
        st.session_state["tb_name"] = ""
    if "tb_desc" not in st.session_state:
        st.session_state["tb_desc"] = ""
    if "tb_questions" not in st.session_state:
        st.session_state["tb_questions"] = [{"id": "", "question": "", "judge_prompt": ""}]

    # --- Form metadata ---
    st.session_state["tb_name"] = st.text_input("Form name", value=st.session_state["tb_name"])
    st.session_state["tb_desc"] = st.text_input("Description", value=st.session_state["tb_desc"])

    st.subheader("Questions")

    questions = st.session_state["tb_questions"]

    for i, q in enumerate(questions):
        # Use the widget key value if it exists (user already typed), else fall back to list
        display_id = st.session_state.get(f"q_id_{i}", q.get("id", "")) or "(untitled)"
        with st.expander(f"Question {i + 1}: {display_id}", expanded=True):
            st.text_input("ID", value=q.get("id", ""), key=f"q_id_{i}")
            st.text_area("Question text", value=q.get("question", ""), key=f"q_text_{i}")
            st.text_area("Judge prompt", value=q.get("judge_prompt", ""), key=f"q_judge_{i}")

            if st.button("Remove", key=f"q_rm_{i}"):
                questions.pop(i)
                # Clean up widget keys for removed and subsequent questions
                for k in list(st.session_state.keys()):
                    if k.startswith(("q_id_", "q_text_", "q_judge_")):
                        del st.session_state[k]
                st.rerun()

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Add question"):
            questions.append({"id": "", "question": "", "judge_prompt": ""})
            st.rerun()

    # --- Save ---
    st.divider()
    save_col1, save_col2 = st.columns(2)
    with save_col1:
        filename = st.text_input("Filename (without extension)", value="my_template")
    with save_col2:
        fmt = st.radio("Format", ["JSON", "YAML"], horizontal=True)

    if st.button("Save template", type="primary"):
        # Read current values from widget keys (source of truth after user edits)
        saved_questions = []
        for i in range(len(questions)):
            qid = st.session_state.get(f"q_id_{i}", "")
            if qid:
                saved_questions.append({
                    "id": qid,
                    "question": st.session_state.get(f"q_text_{i}", ""),
                    "judge_prompt": st.session_state.get(f"q_judge_{i}", ""),
                })

        template_data = {
            "name": st.session_state["tb_name"],
            "description": st.session_state["tb_desc"],
            "questions": saved_questions,
        }

        TEMPLATES_DIR.mkdir(exist_ok=True)
        ext = ".json" if fmt == "JSON" else ".yaml"
        out_path = TEMPLATES_DIR / (filename + ext)

        if fmt == "JSON":
            out_path.write_text(json.dumps(template_data, indent=2))
        else:
            out_path.write_text(yaml.dump(template_data, default_flow_style=False, sort_keys=False))

        st.success(f"Saved to `{out_path.relative_to(TEMPLATES_DIR.parent)}`")


# ---------------------------------------------------------------------------
# Tab 2: Form Filler — Graph mode
# ---------------------------------------------------------------------------

def _form_filler_graph(selected: str):
    """Graph mode: deterministic HITL with interrupt/resume."""

    if st.button("Start Form", key="ff_graph_start") or st.session_state.get("ff_active"):
        if not st.session_state.get("ff_active") or st.session_state.get("ff_current_template") != selected:
            with st.spinner("Loading template and building graph..."):
                graph, template = create_form_graph(str(TEMPLATES_DIR / selected))
                config = {"configurable": {"thread_id": f"streamlit-graph-{selected}"}}

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

                st.session_state["ff_active"] = True
                st.session_state["ff_current_template"] = selected
                st.session_state["ff_graph"] = graph
                st.session_state["ff_config"] = config
                st.session_state["ff_template_data"] = template
                st.session_state["ff_result"] = result
                st.session_state["ff_history"] = []

        graph = st.session_state["ff_graph"]
        config = st.session_state["ff_config"]
        template = st.session_state["ff_template_data"]
        result = st.session_state["ff_result"]
        total_questions = len(template["questions"])

        # --- Check completion ---
        interrupts = result.get("__interrupt__", [])
        if not interrupts:
            st.progress(1.0, text=f"{total_questions}/{total_questions} questions answered")
            st.success(f"{template['name']} — COMPLETE")
            for q in template["questions"]:
                ans = result.get("answers", {}).get(q["id"], "(unanswered)")
                st.markdown(f"**{q['question']}**")
                st.markdown(f"> {ans}")
            if st.button("Reset form"):
                _clear_ff_state()
                st.rerun()
            return

        # --- Show progress ---
        answered_count = len(result.get("answers", {}))
        st.progress(answered_count / total_questions, text=f"{answered_count}/{total_questions} questions answered")

        # --- Show history ---
        for entry in st.session_state.get("ff_history", []):
            st.markdown(f"**Q: {entry['question']}**")
            st.markdown(f"> {entry['answer']}")
            if entry.get("feedback"):
                st.warning(f"Feedback: {entry['feedback']}")
            if entry.get("accepted"):
                st.caption("Accepted")
            st.divider()

        # --- Current question ---
        prompt = interrupts[0].value
        question_text = prompt["question"]

        if "previous_feedback" in prompt:
            st.warning(f"**Feedback:** {prompt['previous_feedback']}")

        st.markdown(f"### {question_text}")

        with st.form("answer_form", clear_on_submit=True):
            answer = st.text_area("Your answer", key="ff_answer_input")
            submitted = st.form_submit_button("Submit", type="primary")

        if submitted and answer is not None:
            with st.spinner("Validating answer..."):
                result = graph.invoke(Command(resume=answer), config=config)
                st.session_state["ff_result"] = result

                new_interrupts = result.get("__interrupt__", [])
                if not new_interrupts:
                    st.session_state["ff_history"].append({
                        "question": question_text,
                        "answer": answer,
                        "accepted": True,
                    })
                else:
                    new_prompt = new_interrupts[0].value
                    if new_prompt["question_id"] != prompt["question_id"]:
                        st.session_state["ff_history"].append({
                            "question": question_text,
                            "answer": answer,
                            "accepted": True,
                        })
                    else:
                        st.session_state["ff_history"].append({
                            "question": question_text,
                            "answer": answer,
                            "feedback": new_prompt.get("previous_feedback", ""),
                            "accepted": False,
                        })

                st.rerun()


# ---------------------------------------------------------------------------
# Tab 2: Form Filler — Agent mode
# ---------------------------------------------------------------------------

def _form_filler_agent(selected: str):
    """Agent mode: conversational multi-turn chat."""

    # Initialize chat history
    if "ff_messages" not in st.session_state:
        st.session_state["ff_messages"] = []

    if st.button("Start Form", key="ff_agent_start") or st.session_state.get("ff_active"):
        if not st.session_state.get("ff_active") or st.session_state.get("ff_current_template") != selected:
            with st.spinner("Creating agent and starting conversation..."):
                agent, template = create_form_agent(str(TEMPLATES_DIR / selected))
                config = {"configurable": {"thread_id": f"streamlit-agent-{selected}"}}

                result = agent.invoke(
                    {"messages": [{"role": "user", "content": "I'm ready to fill out the form."}]},
                    config=config,
                )

                agent_reply = result["messages"][-1].content

                st.session_state["ff_active"] = True
                st.session_state["ff_current_template"] = selected
                st.session_state["ff_agent"] = agent
                st.session_state["ff_config"] = config
                st.session_state["ff_template_data"] = template
                st.session_state["ff_messages"] = [
                    {"role": "assistant", "content": agent_reply},
                ]

        # --- Render chat history ---
        for msg in st.session_state["ff_messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # --- Chat input ---
        if user_input := st.chat_input("Your answer"):
            st.session_state["ff_messages"].append({"role": "user", "content": user_input})

            with st.spinner("Thinking..."):
                agent = st.session_state["ff_agent"]
                config = st.session_state["ff_config"]

                result = agent.invoke(
                    {"messages": [{"role": "user", "content": user_input}]},
                    config=config,
                )

                agent_reply = result["messages"][-1].content

            st.session_state["ff_messages"].append({"role": "assistant", "content": agent_reply})
            st.rerun()

        # --- Reset ---
        if st.session_state.get("ff_active"):
            if st.button("Reset form"):
                _clear_ff_state()
                st.rerun()


# ---------------------------------------------------------------------------
# Tab 2: Form Filler — dispatcher
# ---------------------------------------------------------------------------

def form_filler_tab():
    st.header("Form Filler")

    templates = list_templates()
    if not templates:
        st.info("No templates found in `templates/`. Create one in the Template Builder tab first.")
        return

    # --- Mode toggle ---
    mode = st.toggle(
        "Use Agent (chat) mode",
        value=st.session_state.get("ff_mode", False),
        help="**Graph**: deterministic question-by-question with structured validation. "
             "**Agent**: free-form conversational chat powered by an LLM.",
    )

    # Reset state when mode changes
    if mode != st.session_state.get("ff_mode"):
        _clear_ff_state()
        st.session_state["ff_mode"] = mode

    if mode:
        st.caption("Agent mode — conversational chat with LLM-driven flow")
    else:
        st.caption("Graph mode — deterministic question-by-question with HITL interrupts")

    selected = st.selectbox("Select template", templates, key="ff_template")

    if mode:
        _form_filler_agent(selected)
    else:
        _form_filler_graph(selected)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Form Agent", layout="wide")
st.title("Form Agent")

tab1, tab2 = st.tabs(["Template Builder", "Form Filler"])

with tab1:
    template_builder_tab()

with tab2:
    form_filler_tab()
