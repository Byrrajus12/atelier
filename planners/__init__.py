"""Model-backed planners — integrations, deliberately OUTSIDE ``core/``.

A planner that reaches a hosted model needs network I/O and an API key, which is the
same category of concern as an Easel (synthetic input, screen capture): environment
integration, not domain logic. ``core/`` stays pure (CLAUDE.md Principle 2) and depends
only on the ``Planner`` ABC in ``core/planner.py``; implementations live here and are
injected at the edge (``scripts/live_run.py``).
"""
