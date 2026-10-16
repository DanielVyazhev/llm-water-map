"""drybench: measuring filler in LLM outputs.

The score is the task-relevant information a response delivers to a fixed reader model, per output
token. Filler is whatever can be removed without losing that information. Entropy signals of the
generating model attribute filler to its source (autopilot, template hedging, over-elaboration,
compensation for uncertainty).
"""

from drybench.types import Item, Probe, Span

__version__ = "1.2.4"
# Bump when scoring (or item answer forms) changes: stored scores are recomputed (all of them, or
# only those drybench.rescore.PARTIAL names for that version).
SCORING_VERSION = "11"
# Bump when segmentation or the entropy signals change: stored signals are recomputed.
SIGNALS_VERSION = "3"
__all__ = ["Item", "Probe", "Span", "SCORING_VERSION", "SIGNALS_VERSION", "__version__"]
