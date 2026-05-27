"""Pydantic schemas for Google Gemini Deep Research API request and response types.

This module defines all request and response schemas for the Google Gemini
Deep Research integration. Unlike Perplexity's synchronous model, Gemini uses
an async polling workflow where clients submit a job, receive an interaction ID,
and then poll for status updates and results.

Schema Organization:
    1. Enums - GeminiInteractionStatus, GeminiStreamEventType, GeminiDeltaType
    2. Nested Models - GeminiUsage, GeminiStep
    3. Request Models - GeminiDeepResearchRequest, GeminiDeepResearchPollRequest
    4. Response Models - GeminiDeepResearchJobResponse, GeminiDeepResearchResultResponse
"""

from datetime import datetime
from enum import StrEnum

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

# =============================================================================
# Enums
# =============================================================================


class GeminiInteractionStatus(StrEnum):
    """Status states for Gemini deep research interactions.

    Represents the lifecycle states of an async deep research job from
    submission through completion or failure.

    Attributes:
        PENDING: Job submitted but not yet started processing.
        IN_PROGRESS: Job is actively being processed.
        COMPLETED: Job finished successfully with results available.
        REQUIRES_ACTION: Job is waiting for an external action/tool result.
        FAILED: Job encountered an error and could not complete.
        CANCELLED: Job was cancelled before completion.
        INCOMPLETE: Job stopped before producing a complete result.
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REQUIRES_ACTION = "requires_action"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INCOMPLETE = "incomplete"


class GeminiStreamEventType(StrEnum):
    """Event types for Gemini streaming responses during polling.

    Identifies the type of event received when polling for job progress,
    enabling clients to handle different event types appropriately.

    Attributes:
        THINKING_UPDATE: Progress update on reasoning/thinking process.
        RESEARCH_UPDATE: Progress update on research gathering.
        FINAL_RESULT: Final result payload with complete response.
        INTERACTION_START: Native Interactions API stream start event.
        INTERACTION_COMPLETE: Native Interactions API stream completion event.
        INTERACTION_STATUS_UPDATE: Native Interactions API status update event.
        CONTENT_START: Native Interactions API content start event.
        CONTENT_DELTA: Native Interactions API content delta event.
        CONTENT_STOP: Native Interactions API content stop event.
        ERROR: Error event indicating job failure.
    """

    THINKING_UPDATE = "thinking_update"
    RESEARCH_UPDATE = "research_update"
    FINAL_RESULT = "final_result"
    INTERACTION_START = "interaction.start"
    INTERACTION_COMPLETE = "interaction.complete"
    INTERACTION_STATUS_UPDATE = "interaction.status_update"
    CONTENT_START = "content.start"
    CONTENT_DELTA = "content.delta"
    CONTENT_STOP = "content.stop"
    ERROR = "error"


class GeminiDeltaType(StrEnum):
    """Delta types for incremental response updates.

    Categorizes the type of content in incremental streaming updates.

    Attributes:
        TEXT: Text content delta.
        TOOL_CALL: Tool/function call delta.
        STATUS: Status change delta.
    """

    TEXT = "text"
    TOOL_CALL = "tool_call"
    STATUS = "status"


# =============================================================================
# Nested Models
# =============================================================================


class GeminiUsage(BaseModel):
    """Token usage information from Gemini API response.

    Contains token counts for monitoring API usage and costs.
    """

    model_config = ConfigDict(extra="allow")

    input_tokens: int = Field(
        default=0,
        validation_alias=AliasChoices(
            "input_tokens",
            "inputTokens",
            "prompt_tokens",
            "promptTokens",
            "promptTokenCount",
            "inputTokenCount",
        ),
        description="Number of tokens in the input prompt",
    )
    output_tokens: int = Field(
        default=0,
        validation_alias=AliasChoices(
            "output_tokens",
            "outputTokens",
            "completion_tokens",
            "completionTokens",
            "candidatesTokenCount",
            "outputTokenCount",
        ),
        description="Number of tokens in the generated output",
    )
    total_tokens: int = Field(
        default=0,
        validation_alias=AliasChoices(
            "total_tokens",
            "totalTokens",
            "total_token_count",
            "totalTokenCount",
        ),
        description="Total tokens used in the request",
    )


class GeminiStep(BaseModel):
    """Timeline step from Gemini deep research responses.

    The Interactions API revision 2026-05-20 returns a chronological `steps`
    timeline instead of the legacy flat `outputs` array. Each step may be a
    model message, thinking summary, tool call, tool result, or status update.
    The backend normalizes nested Gemini content blocks into `content` and
    `thinking_summary` so the frontend can render a stable shape while still
    preserving unknown step fields via `extra="allow"`.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    step_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("step_id", "id"),
        description="Unique identifier for this timeline step, if provided",
    )
    step_type: str | None = Field(
        default=None,
        validation_alias=AliasChoices("step_type", "type"),
        description="Type of timeline step, such as message, thought, or tool call",
    )
    status: str | None = Field(
        default=None,
        description="Step-level status, if provided by Gemini",
    )
    content: str = Field(
        default="",
        validation_alias=AliasChoices("content", "text"),
        description="Text content extracted from this timeline step",
    )
    thinking_summary: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "thinking_summary",
            "thinkingSummary",
            "summary",
        ),
        description="Summary of reasoning/thinking for this step",
    )
    delta_type: GeminiDeltaType | None = Field(
        default=None,
        description="Legacy delta type if returned by older Interactions revisions",
    )


# =============================================================================
# Request Schemas
# =============================================================================


class GeminiDeepResearchRequest(BaseModel):
    """Request schema for initiating a Gemini deep research job.

    Contains all parameters for submitting a new deep research query.
    The job runs asynchronously and returns an interaction ID for polling.

    Parameter Groups:
        Core: query (required)
        Options: enable_thinking_summaries, file_search_store_names
        Continuation: previous_interaction_id
    """

    model_config = ConfigDict(extra="forbid")

    # -------------------------------------------------------------------------
    # Core Parameters
    # -------------------------------------------------------------------------

    query: str = Field(
        min_length=1,
        max_length=32000,
        description="The research query to investigate",
    )

    # -------------------------------------------------------------------------
    # Optional Parameters
    # -------------------------------------------------------------------------

    enable_thinking_summaries: bool = Field(
        default=False,
        description="Include thinking/reasoning summaries in responses",
    )
    file_search_store_names: list[str] | None = Field(
        default=None,
        description="Names of file stores to search for context",
    )
    previous_interaction_id: str | None = Field(
        default=None,
        description="ID of previous interaction for continuation",
    )

    # -------------------------------------------------------------------------
    # Validators
    # -------------------------------------------------------------------------

    @field_validator("query", mode="before")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate and normalize query string."""
        if isinstance(v, str):
            v = v.strip()
        return v

    @field_validator("file_search_store_names", mode="before")
    @classmethod
    def validate_file_stores(cls, v: list[str] | None) -> list[str] | None:
        """Validate and normalize file store names list."""
        if v is None:
            return None
        if not isinstance(v, list):
            return None
        # Filter out empty strings and strip whitespace
        return [s.strip() for s in v if s and s.strip()]


class GeminiDeepResearchPollRequest(BaseModel):
    """Request schema for polling Gemini deep research job status.

    Contains parameters for checking job progress and retrieving results.
    The last_event_id enables reconnection after disconnection.
    """

    model_config = ConfigDict(extra="forbid")

    interaction_id: str = Field(
        min_length=1,
        description="The interaction ID returned from job creation",
    )
    last_event_id: str | None = Field(
        default=None,
        description="ID of last received event for reconnection",
    )


# =============================================================================
# Response Schemas
# =============================================================================


class GeminiDeepResearchJobResponse(BaseModel):
    """Response schema for Gemini deep research job creation.

    Returned immediately after submitting a new deep research request.
    Contains the interaction ID needed for subsequent polling.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    interaction_id: str = Field(
        alias="id",
        description="Unique identifier for this research interaction",
    )
    status: GeminiInteractionStatus = Field(
        default=GeminiInteractionStatus.PENDING,
        description="Initial status of the job (typically pending)",
    )
    created_at: datetime | None = Field(
        default=None,
        alias="createTime",
        description="Timestamp when the job was created",
    )


class GeminiDeepResearchResultResponse(BaseModel):
    """Response schema for Gemini deep research polling results.

    Contains the current status and any available timeline steps from polling.
    When status is COMPLETED, steps and usage will be populated.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    status: GeminiInteractionStatus = Field(
        description="Current status of the research job",
    )
    steps: list[GeminiStep] = Field(
        default_factory=list,
        description="Chronological timeline steps from the research interaction",
    )
    usage: GeminiUsage | None = Field(
        default=None,
        description="Token usage information (available on completion)",
    )
    completed_at: datetime | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "completed_at",
            "completedAt",
            "completed",
            "updated",
            "updateTime",
        ),
        description="Timestamp when the job completed or was last updated",
    )
    event_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("event_id", "eventId"),
        description="ID of this event for reconnection tracking",
    )
    event_type: GeminiStreamEventType | None = Field(
        default=None,
        validation_alias=AliasChoices("event_type", "eventType"),
        description="Type of streaming event (if applicable)",
    )
    error_message: str | None = Field(
        default=None,
        validation_alias=AliasChoices("error_message", "errorMessage"),
        description="Error message if status is FAILED",
    )
