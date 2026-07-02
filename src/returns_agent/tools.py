from typing import Any

from langchain_core.tools import tool
from langsmith import traceable
from pydantic import BaseModel, Field

from returns_agent.models import Order


# Tool argument schema shown to the LLM.
class GetOrderInput(BaseModel):
    order_id: str = Field(
        description=(
            "Customer order number to look up. Accepts values like '101' or '#101'."
        )
    )


# The assessment asked for hardcoded order data, so this is our mock API store.
MOCK_ORDERS: dict[str, Order] = {
    "101": Order(
        order_id="101",
        item="Milan Boucle Sofa",
        price=899.00,
        status="Delivered 5 days ago",
        delivered_days_ago=5,
    ),
    "102": Order(
        order_id="102",
        item="Ceramic Vase",
        price=45.00,
        status="Delivered 45 days ago",
        delivered_days_ago=45,
    ),
    "103": Order(
        order_id="103",
        item="Outdoor Dining Set",
        price=1200.00,
        status="In Transit",
        delivered_days_ago=None,
    ),
}


# Customers may type "#101" or spaces, so clean that before lookup.
def normalize_order_id(order_id: str) -> str:
    return order_id.strip().removeprefix("#").strip()


# Plain Python lookup. In production this is where an order API call would live.
@traceable(run_type="tool", name="get_order")
def get_order(order_id: str) -> Order | None:
    normalized_order_id = normalize_order_id(order_id)
    return MOCK_ORDERS.get(normalized_order_id)


# LangChain tool wrapper. The model calls this, not the raw dictionary.
@tool(
    "get_order",
    args_schema=GetOrderInput,
    description=(
        "Look up trusted order data by order_id. Use this before applying return "
        "policy. Returns whether the order was found and, when found, the item, "
        "price, delivery status, and delivered_days_ago fields."
    ),
)
def get_order_tool(order_id: str) -> dict[str, Any]:
    order = get_order(order_id)
    normalized_order_id = normalize_order_id(order_id)

    # Not found is a normal customer-care case, not a system crash.
    if order is None:
        return {
            "found": False,
            "order_id": normalized_order_id,
            "error": "ORDER_NOT_FOUND",
        }

    return {
        "found": True,
        "order": order.model_dump(mode="json"),
    }


# Keep the allowed tool list explicit. The agent only gets what is in this list.
ORDER_TOOLS = [get_order_tool]
