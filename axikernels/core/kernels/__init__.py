from .kernel import Kernel

try:
	from .objective_function import L2ObjectiveFunction
except ModuleNotFoundError as exc:
	if exc.name not in {"ruamel", "ruamel.yaml"}:
		raise
	L2ObjectiveFunction = None