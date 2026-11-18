"""Model artifact inventory response models."""

from pydantic import BaseModel


class ModelInfo(BaseModel):
	model_id: str
	filename: str
	artifact_type: str
	size_bytes: int
	modified_at: str


class ModelsResponse(BaseModel):
	models: list[ModelInfo]
	message: str