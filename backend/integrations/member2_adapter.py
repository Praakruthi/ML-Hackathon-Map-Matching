"""Replaceable adapter boundary for Member 2 model inference."""

from typing import Any

from backend.integrations.base import BaseAdapter
from backend.integrations.contracts import (
	FeatureSet,
	Member2Implementation,
	ProbabilitySet,
)


class Member2Adapter(BaseAdapter):
	name = "Member 2 - Machine Learning"
	package = "src.member2"

	def __init__(self, implementation: Member2Implementation | None = None) -> None:
		super().__init__(implementation)

	def load_best_model(self, model_id: str | None = None) -> Any:
		return self.implementation().load_best_model(model_id)

	def predict_candidates(self, model: Any, features: FeatureSet) -> ProbabilitySet:
		return self.implementation().predict_candidates(model, features)