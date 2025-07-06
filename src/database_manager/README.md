# Database Manager Web UI

A lightweight Flask-Admin panel for viewing and managing the **ai-trading-bot** PostgreSQL database.

---

## Features

* **Auto-generated CRUD UI** – every SQLAlchemy model (`trades`, `positions`, etc.) is exposed automatically via Flask-Admin.
* **Search & Filters** – text columns are searchable out-of-the-box.
* **Detail & Edit Views** – inspect or modify individual records.
* **Create / Delete Records** – useful for quick data corrections during development.
* **Schema Sync (`/migrate`)** – ensures tables exist without a full Alembic setup.
* **Health & Diagnostics**
  * `/health` – simple OK response for load-balancers.
  * `/db_info` – live connection-pool statistics.
* **Graceful Error Handling** – if the database cannot be reached the service still starts (exposes `/db_error`).

---

## Running Locally

```bash
# 1. Install dependencies (virtualenv recommended)
pip install -r requirements.txt  # or requirements-server.txt

# 2. Ensure DATABASE_URL is exported (PostgreSQL)
export DATABASE_URL=postgresql://user:password@localhost:5432/ai_trading

# 3. Launch the web UI
python src/database_manager/app.py

# 4. Open your browser
open http://localhost:8000/admin  # or visit manually
```

If you need to (re)create any missing tables:

```bash
curl http://localhost:8000/migrate  # will respond with JSON status
```

---

## Deployment on Railway

1. Add **DATABASE_URL** & **DB_MANAGER_SECRET_KEY** environment variables.
2. Point the *Start Command* to:

```bash
python src/database_manager/app.py
```

3. Railway automatically injects `PORT`; the app factory binds to it.

---

## Extending / Customising

* To hide sensitive columns or make them read-only, subclass `CustomModelView` in `app.py` and override properties, then register your own view.
* Additional routes can be added below the existing ones – keep using the shared `db_manager` and `db_session` objects.

---

© 2025 – AI Trading Bot project