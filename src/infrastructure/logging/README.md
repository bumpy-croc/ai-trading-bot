# Platform Logging

Centralized logging facilities:
- `config.py` – configures structured logging handlers/formatters.
- `context.py` – contextvars-based request metadata that gets attached to every
  log event.
- `events.py` – convenience emitters (engine, risk, data, db events).
- `decision_logger.py` – persists strategy decisions to the database.

Import helpers from here instead of scattering ad-hoc logging utilities across
`src/utils` or `src/trading`.
