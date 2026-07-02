from __future__ import annotations

from typing import Any

import streamlit as st

from returns_agent.graph import ReturnsAgent
from returns_agent.llm import LLMError
from returns_agent.models import FinalDecision

TERMINAL_STATUSES = {"COMPLETED", "FAILED", "HUMAN_REVIEW"}


# Reset only the demo session. The core agent code does not depend on Streamlit.
def _new_ticket() -> None:
    st.session_state.agent_state = None
    st.session_state.last_decision = None
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi, I can help with returns or order issues.",
            "decision": None,
        }
    ]


# Reuse the agent object inside the Streamlit session.
def _get_agent() -> ReturnsAgent:
    agent = st.session_state.get("agent")

    if agent is None:
        agent = ReturnsAgent()
        st.session_state.agent = agent

    return agent


# Streamlit renders the same four-key payload the CLI prints.
def _decision_payload(decision: FinalDecision) -> dict[str, Any]:
    return decision.model_dump(mode="json")


# Keep JSON rendering in one place so every assistant turn looks consistent.
def _render_decision(decision: FinalDecision) -> None:
    st.json(_decision_payload(decision))


# Run one customer turn and save the returned LangGraph state for resume.
def _run_turn(customer_message: str) -> None:
    agent = _get_agent()
    prior_state = st.session_state.get("agent_state")

    decision, state = agent.run(
        customer_message=customer_message,
        prior_state=prior_state,
        ticket_id="streamlit-demo",
    )

    st.session_state.agent_state = state
    st.session_state.last_decision = decision
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": decision.customer_reply,
            "decision": decision,
        }
    )


# Sidebar is just ticket status, not part of the agent.
def _render_sidebar() -> None:
    with st.sidebar:
        st.subheader("Ticket")

        if st.button("New ticket", use_container_width=True):
            _new_ticket()
            st.rerun()

        decision = st.session_state.get("last_decision")
        state = st.session_state.get("agent_state") or {}

        if decision is not None:
            st.metric("Decision", decision.decision.value)
            st.caption(f"Category: {decision.category.value}")
            st.caption(f"Status: {state.get('status', 'NEW')}")
        else:
            st.caption("No decision yet.")


# Render chat messages plus the strict JSON behind each agent response.
def _render_messages() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            decision = message.get("decision")
            if decision is not None:
                with st.expander("Strict JSON"):
                    _render_decision(decision)


# Terminal tickets do not re-enter the automated workflow in this demo.
def _ticket_is_closed() -> bool:
    state = st.session_state.get("agent_state") or {}
    return state.get("status") in TERMINAL_STATUSES


# Convert UI input into one agent turn and show provider/runtime errors clearly.
def _handle_customer_message(customer_message: str) -> None:
    st.session_state.messages.append(
        {"role": "user", "content": customer_message, "decision": None}
    )

    try:
        with st.spinner("Running returns workflow..."):
            _run_turn(customer_message)
    except (LLMError, RuntimeError, ValueError) as exc:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"Failed to run the agent: {exc}",
                "decision": None,
            }
        )


# Streamlit entry point. This stays thin on purpose.
def main() -> None:
    st.set_page_config(page_title="Returns Agent Demo", layout="wide")

    if "messages" not in st.session_state:
        _new_ticket()

    _render_sidebar()

    st.title("Autonomous Returns Agent")
    st.caption("Thin demo UI over the LangGraph returns workflow.")

    _render_messages()

    if _ticket_is_closed():
        st.info("This ticket has left the automated workflow. Start a new ticket to test another conversation.")
        return

    pending_message = st.session_state.pop("pending_message", None)
    customer_message = pending_message or st.chat_input("Type a customer message...")

    if customer_message:
        _handle_customer_message(customer_message)
        st.rerun()


if __name__ == "__main__":
    main()
