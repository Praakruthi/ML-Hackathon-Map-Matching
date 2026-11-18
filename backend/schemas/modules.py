"""Module readiness response models."""

from pydantic import BaseModel


class ModuleStatus(BaseModel):
	name: str
	package: str
	detected: bool
	adapter_ready: bool
	ready: bool
	message: str


class ModulesStatusResponse(BaseModel):
	pipeline_ready: bool
	modules: list[ModuleStatus]