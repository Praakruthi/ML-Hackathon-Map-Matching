"""Readiness registry for the three independently developed modules."""

from dataclasses import asdict

from backend.integrations.base import BaseAdapter
from backend.integrations.member1_adapter import Member1Adapter
from backend.integrations.member2_adapter import Member2Adapter
from backend.integrations.member3_adapter import Member3Adapter


class ModuleRegistry:
	def __init__(self, adapters: tuple[BaseAdapter, ...] | None = None) -> None:
		self.adapters = adapters or (
			Member1Adapter(),
			Member2Adapter(),
			Member3Adapter(),
		)

	def get_status(self) -> dict[str, object]:
		statuses = [asdict(adapter.readiness()) for adapter in self.adapters]
		return {
			"pipeline_ready": bool(statuses) and all(item["ready"] for item in statuses),
			"modules": statuses,
		}


module_registry = ModuleRegistry()