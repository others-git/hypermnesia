from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Memory(BaseModel):
    id: str
    owner_id: str
    scope: str
    type: str = "fact"
    content: str
    description: str
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    importance: float = 1.0
    created_at: datetime | None = None
    updated_at: datetime | None = None
    last_accessed_at: datetime | None = None


class SearchHit(Memory):
    similarity: float
