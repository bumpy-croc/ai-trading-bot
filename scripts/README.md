# Scripts Directory

All operational Python utilities have been migrated into the unified `atb` CLI.
Use `atb --help` to discover the available commands, for example:

- `atb migration baseline` – component strategy regression benchmarking (archives legacy baselines)
- `atb docs validate` – documentation health checks
- `atb strategies version` – strategy manifest updates
- `atb tests parse-junit` – parse JUnit XML failure reports
- `atb regime visualize` – generate regime analysis charts

This folder now only stores ancillary assets referenced by deployment tools,
including:

- `postgres-init.sql` – database bootstrap SQL used by Docker Compose services.

If you previously relied on a script in this directory and cannot find the
corresponding CLI command, check the updated documentation under `docs/` or run
`atb --help`.
