from enum import StrEnum
from typing import Annotated, TypedDict

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
)
from langgraph.graph.message import add_messages

from returns_agent.models import (
    AgentResponse,
    Category,
    FinalDecision,
    Order,
    PolicyOutcome,
)

DEFAULT_MAX_CLARIFICATION_ATTEMPTS = 5


# Internal ticket status. This helps the caller know if the workflow should resume.
class WorkflowStatus(StrEnum):
    RUNNING = "RUNNING"
    WAITING_FOR_CUSTOMER = "WAITING_FOR_CUSTOMER"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# This is the graph memory that moves from node to node.
class AgentState(TypedDict, total=False):
    ticket_id: str
    status: WorkflowStatus

    # LangGraph appends messages instead of replacing the whole conversation.
    messages: Annotated[list[AnyMessage], add_messages]

    # LLM-understood fields. These can improve across turns.
    agent_response: AgentResponse | None
    order_id: str | None
    category: Category
    issue_summary: str

    # Trusted data from the order tool, not from the customer message.
    order: Order | None

    # Policy fields are produced by deterministic Python.
    policy_outcome: PolicyOutcome | None
    human_review_required: bool

    # The four-key JSON we return for the current turn.
    final_decision: FinalDecision | None

    # Avoid endless clarification loops if the customer cannot provide info.
    clarification_attempts: int
    max_clarification_attempts: int


# First state for a new ticket. Later turns reuse prior_state instead.
def build_initial_state(customer_message: str, ticket_id: str = "local-demo") -> AgentState:
    return {
        "ticket_id": ticket_id,
        "status": WorkflowStatus.RUNNING,
        "messages": [HumanMessage(content=customer_message)],
        "order_id": None,
        "category": Category.UNKNOWN,
        "issue_summary": "",
        "human_review_required": False,
        "clarification_attempts": 0,
        "max_clarification_attempts": DEFAULT_MAX_CLARIFICATION_ATTEMPTS,
    }
