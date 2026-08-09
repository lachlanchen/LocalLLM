from __future__ import annotations

import ast
import json
import re
import unicodedata
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

MAX_MODEL_PLAN_BYTES = 16 * 1024
MAX_PLAN_STEPS = 8
MAX_PLAN_NODES = 256
MAX_PLAN_DEPTH = 8

CapabilityName = Literal["respond", "web_search", "paper_search", "vision", "python"]
STEP_ID_PATTERN = re.compile(r"^step_[1-9][0-9]{0,2}$")
MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,199}$")
URL_PATTERN = re.compile(r"(?:\b(?:https?|ftp|file|data):/{0,2}|\bwww\.)", re.IGNORECASE)


def reject_dangerous_source_controls(value: str) -> str:
    """Reject invisible Unicode formatting that can disguise reviewed source."""

    if any(unicodedata.category(character) == "Cf" for character in value):
        raise ValueError("Python source cannot contain invisible formatting controls")
    return value


class StrictAgentModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class NoArguments(StrictAgentModel):
    pass


class SearchArguments(StrictAgentModel):
    query: str = Field(min_length=3, max_length=800)
    limit: int = Field(default=8, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def reject_urls(cls, value: str) -> str:
        if URL_PATTERN.search(value):
            raise ValueError("search queries cannot contain URLs")
        return value


class VisionArguments(StrictAgentModel):
    image_ids: list[str] = Field(min_length=1, max_length=4)
    question: str = Field(min_length=1, max_length=1200)

    @field_validator("image_ids")
    @classmethod
    def validate_image_ids(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("image_ids must be unique")
        for image_id in value:
            if not re.fullmatch(r"img_[0-9a-f]{16,64}", image_id):
                raise ValueError("image_ids must be opaque LocalLLM image identifiers")
        return value

    @field_validator("question")
    @classmethod
    def reject_question_urls(cls, value: str) -> str:
        if URL_PATTERN.search(value):
            raise ValueError("vision questions cannot contain URLs")
        return value


class PythonArguments(StrictAgentModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    code: str = Field(min_length=1, max_length=32_768)
    timeout_seconds: int = Field(default=10, ge=1, le=20)

    @field_validator("code")
    @classmethod
    def reject_nul(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("code must not contain NUL bytes")
        reject_dangerous_source_controls(value)
        if URL_PATTERN.search(value):
            raise ValueError("planned code cannot contain URLs")
        repaired = value.replace("\\r\\n", "\n").replace("\\n", "\n")
        try:
            original_tree = ast.parse(value, filename="<agent-plan>", mode="exec")
        except SyntaxError as original_error:
            # Small local models occasionally double-escape JSON newlines, yielding
            # `statement\\nstatement` after the outer JSON has been decoded. Repair
            # only that exact serialization mistake, and only when the repaired
            # source parses. The operator still sees and approves the final bytes.
            if repaired == value:
                raise ValueError("planned Python must be syntactically valid") from original_error
            try:
                ast.parse(repaired, filename="<agent-plan>", mode="exec")
            except SyntaxError as repaired_error:
                raise ValueError("planned Python must be syntactically valid") from repaired_error
            value = repaired
        else:
            # A leading comment can make double-escaped multi-line code parse as an
            # empty, harmless script. When there are no real line breaks, accept the
            # repaired form only if it is valid and reveals additional statements.
            if repaired != value and "\n" not in value:
                try:
                    repaired_tree = ast.parse(repaired, filename="<agent-plan>", mode="exec")
                except SyntaxError:
                    pass
                else:
                    if len(repaired_tree.body) > len(original_tree.body):
                        value = repaired
        return value


class StepBase(StrictAgentModel):
    id: str = Field(pattern=r"^step_[1-9][0-9]{0,2}$")
    objective: str = Field(min_length=1, max_length=600)
    depends_on: list[str] = Field(default_factory=list, max_length=MAX_PLAN_STEPS)

    @field_validator("depends_on")
    @classmethod
    def validate_dependencies(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("depends_on entries must be unique")
        if any(not STEP_ID_PATTERN.fullmatch(item) for item in value):
            raise ValueError("depends_on contains an invalid step identifier")
        return value

    @field_validator("objective")
    @classmethod
    def reject_objective_urls(cls, value: str) -> str:
        if URL_PATTERN.search(value):
            raise ValueError("step objectives cannot contain URLs")
        return value


class RespondStep(StepBase):
    capability: Literal["respond"]
    arguments: NoArguments = Field(default_factory=NoArguments)


class WebSearchStep(StepBase):
    capability: Literal["web_search"]
    arguments: SearchArguments


class PaperSearchStep(StepBase):
    capability: Literal["paper_search"]
    arguments: SearchArguments


class VisionStep(StepBase):
    capability: Literal["vision"]
    arguments: VisionArguments


class PythonStep(StepBase):
    capability: Literal["python"]
    arguments: PythonArguments


AgentStep = Annotated[
    RespondStep | WebSearchStep | PaperSearchStep | VisionStep | PythonStep,
    Field(discriminator="capability"),
]


class AgentPlan(StrictAgentModel):
    schema_version: Literal["1"]
    goal: str = Field(min_length=1, max_length=1200)
    steps: list[AgentStep] = Field(min_length=1, max_length=MAX_PLAN_STEPS)

    @field_validator("goal")
    @classmethod
    def reject_goal_urls(cls, value: str) -> str:
        if URL_PATTERN.search(value):
            raise ValueError("planned goals cannot contain URLs")
        return value

    @model_validator(mode="after")
    def validate_graph(self) -> AgentPlan:
        seen: set[str] = set()
        if self.steps[-1].capability != "respond":
            raise ValueError("the final step must use respond")
        if sum(step.capability == "respond" for step in self.steps) != 1:
            raise ValueError("a plan must contain exactly one final respond step")
        for step in self.steps:
            if step.id in seen:
                raise ValueError("step identifiers must be unique")
            missing_or_future = set(step.depends_on) - seen
            if missing_or_future:
                raise ValueError("dependencies must refer to earlier steps")
            if step.id in step.depends_on:
                raise ValueError("a step cannot depend on itself")
            seen.add(step.id)
        return self


class IncompleteAgentPlan(StrictAgentModel):
    """A fully valid plan graph whose sole permitted omission is final respond."""

    schema_version: Literal["1"]
    goal: str = Field(min_length=1, max_length=1200)
    steps: list[AgentStep] = Field(min_length=1, max_length=MAX_PLAN_STEPS - 1)

    @field_validator("goal")
    @classmethod
    def reject_goal_urls(cls, value: str) -> str:
        if URL_PATTERN.search(value):
            raise ValueError("planned goals cannot contain URLs")
        return value

    @model_validator(mode="after")
    def validate_incomplete_graph(self) -> IncompleteAgentPlan:
        seen: set[str] = set()
        for index, step in enumerate(self.steps, start=1):
            if step.capability == "respond":
                raise ValueError("safe completion applies only when respond is entirely absent")
            if step.id != f"step_{index}":
                raise ValueError("safe completion requires sequential step identifiers")
            if step.id in seen or set(step.depends_on) - seen:
                raise ValueError("dependencies must refer to unique earlier steps")
            seen.add(step.id)
        return self


class PlanValidationRequest(StrictAgentModel):
    model_output: str = Field(min_length=2, max_length=MAX_MODEL_PLAN_BYTES)
    enabled_capabilities: list[CapabilityName] = Field(
        default_factory=lambda: ["respond"], max_length=5
    )

    @field_validator("enabled_capabilities")
    @classmethod
    def validate_capabilities(cls, value: list[CapabilityName]) -> list[CapabilityName]:
        if len(set(value)) != len(value):
            raise ValueError("enabled_capabilities entries must be unique")
        return value


class PlanProposalRequest(StrictAgentModel):
    goal: str = Field(min_length=1, max_length=4000)
    model: str = Field(min_length=1, max_length=200)
    enabled_capabilities: list[CapabilityName] = Field(min_length=1, max_length=5)

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        if not MODEL_ID_PATTERN.fullmatch(value):
            raise ValueError("model contains unsupported characters")
        return value

    @field_validator("enabled_capabilities")
    @classmethod
    def validate_enabled_capabilities(cls, value: list[CapabilityName]) -> list[CapabilityName]:
        if len(set(value)) != len(value):
            raise ValueError("enabled_capabilities entries must be unique")
        if "respond" not in value:
            raise ValueError("respond must always be enabled")
        return value


class StagedStep(StrictAgentModel):
    id: str
    capability: CapabilityName
    state: Literal["ready", "awaiting_explicit_confirmation"]
    objective: str
    depends_on: list[str]


class PlanStagedEvent(StrictAgentModel):
    type: Literal["plan.staged"] = "plan.staged"
    schema_version: Literal["1"] = "1"
    step_count: int = Field(ge=1, le=MAX_PLAN_STEPS)
    capabilities: list[CapabilityName]


class PlanValidationResponse(StrictAgentModel):
    plan: AgentPlan
    steps: list[StagedStep]
    events: list[PlanStagedEvent]
    executable: Literal[False] = False


class PlanProposalResponse(PlanValidationResponse):
    planner: Literal["local-model", "deterministic-fallback"]
    warning: str | None = None


class UntrustedPlanError(ValueError):
    """Raised when a model-produced plan does not satisfy the fixed contract."""


def _bounded_integer(value: str) -> int:
    if len(value.lstrip("-")) > 10:
        raise ValueError("integer is too large")
    return int(value)


def _reject_float(_value: str) -> float:
    raise ValueError("floating-point values are not allowed in an agent plan")


def _reject_constant(_value: str) -> None:
    raise ValueError("non-finite values are not allowed in an agent plan")


def _validate_json_shape(
    value: object, *, depth: int = 0, counter: list[int] | None = None
) -> None:
    if counter is None:
        counter = [0]
    counter[0] += 1
    if counter[0] > MAX_PLAN_NODES or depth > MAX_PLAN_DEPTH:
        raise ValueError("agent plan JSON is too complex")
    if isinstance(value, dict):
        if any(not isinstance(key, str) or len(key) > 80 for key in value):
            raise ValueError("agent plan contains an invalid object key")
        for child in value.values():
            _validate_json_shape(child, depth=depth + 1, counter=counter)
    elif isinstance(value, list):
        for child in value:
            _validate_json_shape(child, depth=depth + 1, counter=counter)


def _decode_bounded_plan(raw_model_output: str) -> dict[str, Any]:
    encoded = raw_model_output.encode("utf-8")
    if len(encoded) > MAX_MODEL_PLAN_BYTES:
        raise ValueError("model plan exceeds the byte limit")
    decoded: Any = json.loads(
        raw_model_output,
        parse_int=_bounded_integer,
        parse_float=_reject_float,
        parse_constant=_reject_constant,
    )
    if not isinstance(decoded, dict):
        raise ValueError("agent plan must be a JSON object")
    _validate_json_shape(decoded)
    return decoded


class AgentPlanCoordinator:
    """Validate and stage an untrusted model plan without executing any tool.

    The coordinator is deliberately model-agnostic. A 4B model and a 70B model
    receive exactly the same JSON contract, validation limits, and capability
    policy. The caller must dispatch staged steps separately.
    """

    def stage(
        self, raw_model_output: str, enabled_capabilities: list[CapabilityName]
    ) -> PlanValidationResponse:
        try:
            decoded = _decode_bounded_plan(raw_model_output)
            plan = AgentPlan.model_validate(decoded)
        except (json.JSONDecodeError, UnicodeError, ValueError, ValidationError) as exc:
            raise UntrustedPlanError("model output is not a valid bounded agent plan") from exc

        allowed = set(enabled_capabilities)
        allowed.add("respond")
        requested = {step.capability for step in plan.steps}
        unavailable = requested - allowed
        if unavailable:
            names = ", ".join(sorted(unavailable))
            raise UntrustedPlanError(f"plan requested disabled capabilities: {names}")

        staged = [
            StagedStep(
                id=step.id,
                capability=step.capability,
                state=(
                    "awaiting_explicit_confirmation" if step.capability == "python" else "ready"
                ),
                objective=step.objective,
                depends_on=step.depends_on,
            )
            for step in plan.steps
        ]
        ordered_capabilities = list(dict.fromkeys(step.capability for step in plan.steps))
        return PlanValidationResponse(
            plan=plan,
            steps=staged,
            events=[PlanStagedEvent(step_count=len(staged), capabilities=ordered_capabilities)],
        )

    def complete_missing_respond(self, raw_model_output: str) -> str | None:
        """Append only a missing passive final response to an otherwise strict plan."""

        try:
            decoded = _decode_bounded_plan(raw_model_output)
            incomplete = IncompleteAgentPlan.model_validate(decoded)
        except (json.JSONDecodeError, UnicodeError, ValueError, ValidationError):
            return None

        completed = incomplete.model_dump(mode="json")
        last_step = incomplete.steps[-1]
        completed["steps"].append(
            {
                "id": f"step_{len(incomplete.steps) + 1}",
                "capability": "respond",
                "objective": "Answer using the completed prior steps",
                "depends_on": [last_step.id],
                "arguments": {},
            }
        )
        return json.dumps(completed, ensure_ascii=False, separators=(",", ":"))
