"""AI Trading Bot source package.

Importing this package verifies that Python resolved it from the checkout you are actually
working in (GH #1070). The shared venv's editable install pins ``src``/``cli`` to the checkout
``pip install -e .`` was run from, so without this guard a command run from a git worktree
silently executes a *different* branch's code. Placing the check here — rather than in each
entry point — covers ``atb``, ``pytest``, ``python experiments/*.py`` and any ad-hoc script,
because all of them must import ``src`` before they can do anything.
"""

from src._source_root import verify_source_root

verify_source_root()

del verify_source_root
