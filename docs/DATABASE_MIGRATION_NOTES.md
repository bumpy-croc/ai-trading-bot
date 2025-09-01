# Database Migration Notes

We use Alembic with SQLAlchemy 2.x. Avoid calling `Base.metadata.create_all()` inside migrations as it bypasses Alembic's change tracking and may cause drift.

Recommended flow:

1) Ensure env.py defines `target_metadata`:
   - `target_metadata = Base.metadata`

2) Create migrations using autogenerate:

```
alembic revision --autogenerate -m "<message>"
```

3) Review generated operations, especially types and constraints.

4) Apply:

```
alembic upgrade head
```

5) For model changes, repeat steps 2-4.

If the initial migration previously used `create_all`, regenerate it via autogenerate and replace the stubbed `0001_initial_schema.py` with explicit `op.create_table(...)` operations.
