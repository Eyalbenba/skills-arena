"""Reporter module.

Provides output formatting for results:
- TextReporter: Human-readable text output
"""

from .text import TextReporter, print_results

__all__ = [
    "TextReporter",
    "print_results",
]
