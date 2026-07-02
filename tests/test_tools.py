from returns_agent.tools import (
    get_order,
    get_order_tool,
)


def test_get_order_returns_trusted_mock_order() -> None:
    order = get_order("#101")

    assert order is not None
    assert order.order_id == "101"
    assert order.item == "Milan Boucle Sofa"
    assert order.price == 899.00


def test_get_order_tool_returns_safe_found_payload() -> None:
    payload = get_order_tool.invoke({"order_id": " #103 "})

    assert payload == {
        "found": True,
        "order": {
            "order_id": "103",
            "item": "Outdoor Dining Set",
            "price": 1200.00,
            "status": "In Transit",
            "delivered_days_ago": None,
        },
    }


def test_get_order_tool_returns_safe_not_found_payload() -> None:
    payload = get_order_tool.invoke({"order_id": "999"})

    assert payload == {
        "found": False,
        "order_id": "999",
        "error": "ORDER_NOT_FOUND",
    }
