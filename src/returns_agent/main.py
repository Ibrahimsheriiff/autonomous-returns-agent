import argparse
import json
import sys
from typing import NoReturn

from returns_agent.graph import ReturnsAgent
from returns_agent.llm import LLMError
from returns_agent.models import Decision, FinalDecision
from returns_agent.state import AgentState
from returns_agent.trace import render_markdown_trace

DEFAULT_MAX_INTERACTIVE_TURNS = 4


# CLI entry point. It can run one message, interactive mode, or print the graph.
def main() -> None:
    args = _parse_args()

    if args.show_graph:
        _print_workflow_graph()
        return

    try:
        agent = ReturnsAgent()
        first_message = args.message or _read_user_message(
            "Customer message: ",
            required=True,
        )

        final_decision = _run_interactive_ticket(
            agent=agent,
            first_message=first_message,
            ticket_id=args.ticket_id,
            max_turns=args.max_turns,
            pretty=args.pretty,
            show_trace=args.show_trace,
        )

        if final_decision.decision != Decision.ASK_FOR_INFO:
            _print_json(final_decision, pretty=args.pretty)

    except (LLMError, RuntimeError, ValueError) as exc:
        _exit_with_error(str(exc))


# Interactive loop lives outside LangGraph because customers reply asynchronously.
def _run_interactive_ticket(
    agent: ReturnsAgent,
    first_message: str,
    ticket_id: str,
    max_turns: int,
    pretty: bool,
    show_trace: bool,
) -> FinalDecision:
    state: AgentState | None = None
    current_message = first_message

    # Each loop is a new graph run with the previous state passed back in.
    for _ in range(max_turns):
        decision, state = agent.run(
            customer_message=current_message,
            prior_state=state,
            ticket_id=ticket_id,
        )
        if show_trace:
            _print_trace(state)

        if decision.decision != Decision.ASK_FOR_INFO:
            return decision

        # Print the question before waiting for the next customer message.
        _print_json(decision, pretty=pretty)

        follow_up = _read_user_message(
            f"{decision.customer_reply}\nCustomer follow-up: ",
            required=False,
        )

        if not follow_up:
            return decision

        current_message = follow_up

    # Fail loudly rather than hiding a ticket that never resolved.
    if state is None:
        raise RuntimeError("Agent did not create state.")

    raise RuntimeError(
        "The ticket reached the interactive turn limit before a final decision."
    )


# Prints the actual compiled graph, not a hand-drawn copy.
def _print_workflow_graph() -> None:
    agent = ReturnsAgent()
    print(agent.draw_mermaid())


# Keep stdout as JSON because the assessment asks for programmatic output.
def _print_json(decision: FinalDecision, pretty: bool) -> None:
    payload = decision.model_dump(mode="json")

    if pretty:
        print(json.dumps(payload, indent=2))
        return

    print(json.dumps(payload, separators=(",", ":")))


# Trace goes to stderr so stdout can stay clean JSON.
def _print_trace(state: AgentState) -> None:
    print(render_markdown_trace(state), file=sys.stderr)
    print(file=sys.stderr)


# Read from stdin so the CLI works both interactively and in scripts.
def _read_user_message(prompt: str, required: bool) -> str:
    print(prompt, file=sys.stderr, end="")
    value = sys.stdin.readline().strip()

    if required and not value:
        raise ValueError("A customer message is required.")

    return value


# Command-line options for demo and assessment runs.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Autonomous Returns & Resolution Agent."
    )

    parser.add_argument(
        "-m",
        "--message",
        help="Raw customer message. If omitted, the CLI reads from stdin.",
    )
    parser.add_argument(
        "--ticket-id",
        default="local-demo",
        help="Ticket/session ID used for resumable state.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=DEFAULT_MAX_INTERACTIVE_TURNS,
        help="Maximum ASK_FOR_INFO follow-up turns before failing closed.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    parser.add_argument(
        "--show-graph",
        action="store_true",
        help="Print the compiled LangGraph workflow as Mermaid and exit.",
    )
    parser.add_argument(
        "--show-trace",
        action="store_true",
        help="Print a compact Markdown trace for each graph turn.",
    )

    return parser.parse_args()


# One clean error path for CLI failures.
def _exit_with_error(message: str) -> NoReturn:
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
