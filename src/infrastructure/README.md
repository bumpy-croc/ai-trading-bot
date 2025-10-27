# Platform Infrastructure

Shared infrastructure that supports every subsystem (logging, runtime environment
helpers, path/secret discovery). Modules inside this package must remain
framework-agnostic and safe to import from any context (CLI, backtests, live
trading).

Subpackages:
- `logging/` – structured logging configuration, context propagation, and
  decision logging helpers.
- `runtime/` – process bootstrap utilities (project paths, geo detection,
  cache TTL guards, secret management).

All modules here should avoid depending on strategy/trading domains to keep the
layer reusable.
