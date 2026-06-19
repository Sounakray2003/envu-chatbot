"""Pydantic models for API source ingestion requests."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import (
    BaseModel,
    Field,
    HttpUrl,
    SecretStr,
    field_validator,
    model_validator,
)


class HttpMethod(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


WRITE_METHODS = {HttpMethod.POST, HttpMethod.PUT, HttpMethod.PATCH}


class AuthType(str, Enum):
    NONE = "none"
    BASIC = "basic"
    BEARER = "bearer token"
    API_KEY = "api key"
    OAUTH2 = "oauth2"


class PaginationType(str, Enum):
    PAGE_NUMBER = "page_number"
    CURSOR = "cursor"
    OFFSET = "offset"
    NONE = "none"


class PostPaginationStrategy(str, Enum):
    QUERY = "query"
    BODY = "body"


class NoAuth(BaseModel):
    type: Literal[AuthType.NONE] = AuthType.NONE


class BasicAuth(BaseModel):
    type: Literal[AuthType.BASIC] = AuthType.BASIC
    username: str = Field(..., min_length=1, description="API username")
    password: SecretStr = Field(..., description="API password")


class BearerAuth(BaseModel):
    type: Literal[AuthType.BEARER] = AuthType.BEARER
    token: SecretStr = Field(..., description="Bearer token")


class ApiKeyAuth(BaseModel):
    type: Literal[AuthType.API_KEY] = AuthType.API_KEY
    header_name: str = Field(
        default="X-API-Key",
        description="Header name used for the API key",
    )
    api_key: SecretStr = Field(..., description="API key value")


class OAuth2Auth(BaseModel):
    type: Literal[AuthType.OAUTH2] = AuthType.OAUTH2
    token_url: HttpUrl = Field(..., description="OAuth2 token endpoint")
    client_id: str = Field(..., min_length=1)
    client_secret: SecretStr = Field(...)
    scope: str = Field(default="")


AuthConfig = Annotated[
    Union[NoAuth, BasicAuth, BearerAuth, ApiKeyAuth, OAuth2Auth],
    Field(discriminator="type"),
]


class PaginationConfig(BaseModel):
    """Pagination configuration for multi-page API responses."""

    pagination_type: PaginationType = Field(default=PaginationType.NONE)
    page_param_name: Optional[str] = Field(default=None)
    page_size_param_name: Optional[str] = Field(default=None)
    page_size_value: Optional[int] = Field(default=None, ge=1, le=1000)
    start_page: int = Field(default=1, ge=0)
    post_pagination_strategy: PostPaginationStrategy = Field(
        default=PostPaginationStrategy.QUERY
    )

    @model_validator(mode="after")
    def _require_params_when_paginating(self) -> "PaginationConfig":
        if self.pagination_type != PaginationType.NONE:
            if not self.page_param_name:
                raise ValueError(
                    "page_param_name is required when pagination_type is not 'none'"
                )
            if not self.page_size_param_name:
                raise ValueError(
                    "page_size_param_name is required when pagination_type is not 'none'"
                )
            if self.page_size_value is None:
                raise ValueError(
                    "page_size_value is required when pagination_type is not 'none'"
                )
        return self


class APISourceInput(BaseModel):
    """Validated API source configuration used by ingestion."""

    source_type: Literal["api"] = "api"

    source_name: str = Field(..., min_length=1, max_length=255)
    knowledge_base_id: Optional[str] = Field(default=None)
    knowledge_base_name: Optional[str] = Field(default=None, max_length=255)

    http_method: HttpMethod = Field(default=HttpMethod.GET)
    url: HttpUrl = Field(..., description="Full API endpoint URL")
    headers: dict[str, str] = Field(default_factory=dict)
    query_params: dict[str, str] = Field(default_factory=dict)
    request_body: Optional[Any] = Field(default=None)
    verify_ssl: bool = Field(default=True)
    ca_bundle_path: Optional[str] = Field(default=None)

    auth: AuthConfig = Field(default_factory=NoAuth)
    pagination: PaginationConfig = Field(default_factory=PaginationConfig)

    json_path: Optional[str] = Field(default=None)
    field_paths: list[str] = Field(default_factory=list)

    @field_validator("url", mode="before")
    @classmethod
    def _normalise_url(cls, value: str) -> str:
        normalized = str(value).strip()
        if normalized and not normalized.startswith(("http://", "https://")):
            normalized = "https://" + normalized
        return normalized

    @model_validator(mode="after")
    def _body_only_for_write_methods(self) -> "APISourceInput":
        if self.request_body is not None and self.http_method not in WRITE_METHODS:
            raise ValueError(
                f"request_body should not be set for {self.http_method.value} requests. "
                f"request_body is only valid for: {', '.join(m.value for m in WRITE_METHODS)}."
            )
        return self

    @field_validator("field_paths", mode="before")
    @classmethod
    def _normalise_field_paths(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [part.strip() for part in value.split(",")]
        if not isinstance(value, list):
            raise ValueError("field_paths must be a list of strings")
        return [str(item).strip() for item in value if str(item).strip()]

    @model_validator(mode="after")
    def _validate_knowledge_base_target(self) -> "APISourceInput":
        if self.knowledge_base_id and self.knowledge_base_name:
            raise ValueError(
                "Provide either knowledge_base_id or knowledge_base_name, not both"
            )
        if self.knowledge_base_name is not None:
            self.knowledge_base_name = self.knowledge_base_name.strip() or None
        if self.ca_bundle_path is not None:
            self.ca_bundle_path = self.ca_bundle_path.strip() or None
        return self
