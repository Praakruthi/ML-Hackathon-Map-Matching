"""Common readiness reporting and errors for member adapters."""

from dataclasses import dataclass
from importlib.util import find_spec


class ModuleUnavailableError(RuntimeError):
	"""Raised when a requested member implementation is not integrated."""


@dataclass(frozen=True)
class ModuleReadiness:
	name: str
	package: str
	detected: bool
	adapter_ready: bool
	ready: bool
	message: str


class BaseAdapter:
	name = "member"
	package = ""

	def __init__(self, implementation: object | None = None) -> None:
		self._implementation = implementation

	@property
	def ready(self) -> bool:
		return self._implementation is not None

	def readiness(self) -> ModuleReadiness:
		try:
			detected = bool(self.package and find_spec(self.package))
		except (ImportError, ModuleNotFoundError, ValueError):
			detected = False
		if self.ready:
			message = "Standardized adapter implementation is connected."
		elif detected:
			message = "Module package detected, but the standardized adapter is not connected."
		else:
			message = "Module implementation has not been detected."
		return ModuleReadiness(
			name=self.name,
			package=self.package,
			detected=detected,
			adapter_ready=self.ready,
			ready=self.ready,
			message=message,
		)

	def implementation(self):
		if self._implementation is None:
			raise ModuleUnavailableError(
				f"{self.name} is unavailable: {self.readiness().message}"
			)
		return self._implementation