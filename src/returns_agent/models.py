from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


# These are the only decisions the outside system should ever receive.
class Decision(StrEnum):
    REFUND = "REFUND"
    REJECT = "REJECT"
    ESCALATE = "ESCALATE"
    ASK_FOR_INFO = "ASK_FOR_INFO"


# Ticket categories are controlled so the model cannot invent random labels.
class Category(StrEnum):
    DAMAGE_IN_TRANSIT = "Damage in transit"
    DAMAGE_OR_BROKEN_ITEM = "Damage or broken item"
    ASSEMBLY_ISSUE = "Assembly issue"
    MAJOR_DEFECT = "Major defect"
    CHANGE_OF_MIND = "Change of mind"
    DELIVERY_STATUS_ENQUIRY = "Delivery status enquiry"
    WRONG_COLOUR_OR_PRODUCT_MISMATCH = "Wrong colour or product mismatch"
    UNKNOWN = "Unknown"


# Internal graph routing. This is not the final business decision.
class AgentRoute(StrEnum):
    ASK_CUSTOMER = "ASK_CUSTOMER"
    RUN_POLICY = "RUN_POLICY"


# Trusted order shape returned by the mock API/tool.
class Order(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    order_id: str = Field(min_length=1)
    item: str = Field(min_length=1)
    price: float = Field(ge=0)
    status: str = Field(min_length=1)
    delivered_days_ago: int | None = None


# What the LLM returns after we ask it to choose the next graph step.
class AgentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: AgentRoute
    order_id: str | None = None
    category: Category = Category.UNKNOWN
    customer_reply: str | None = None
    issue_summary: str = ""


# Policy output stays separate so the LLM cannot overwrite the decision.
class PolicyOutcome(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    decision: Decision
    reason_code: str
    explanation: str


# Reply model only gets to write wording, not decide the outcome.
class CustomerReplyDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_reply: str = Field(min_length=1)

    # Empty customer replies are not useful, so fail validation early.
    @field_validator("customer_reply")
    @classmethod
    def customer_reply_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("customer_reply must not be blank")

        return value


# This is the exact four-key JSON contract from the assessment.
class FinalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str | None
    decision: Decision
    customer_reply: str = Field(min_length=1)
    category: Category

    # Same guard at the final boundary as well.
    @field_validator("customer_reply")
    @classmethod
    def customer_reply_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("customer_reply must not be blank")

        return value
