# Database Manager Web UI

A lightweight Flask-Admin panel for viewing and managing the trading bot PostgreSQL database.

## Features
- Auto-generated CRUD for SQLAlchemy models (trades, positions, etc.)
- Search, filters, detail/edit views
- Health endpoints: `/health`, connection info `/db_info`
- Simple schema sync endpoint: `/migrate`

## Run locally
```bash
pip install -r requirements-server.txt
export DATABASE_URL=postgresql://trading_bot:dev_password_123@localhost:5432/ai_trading_bot
python src/database_manager/app.py
# Open http://localhost:8000/admin
```

Schema sync:
```bash
curl http://localhost:8000/migrate
```

## Railway
- Set `DATABASE_URL` and `DB_MANAGER_SECRET_KEY`
- Start command: `python src/database_manager/app.py`

## Extend
- Subclass `CustomModelView` in `app.py` to hide/make fields read-only
- Add routes using shared `db_manager` and `db_session`

© 2025 – AI Trading Bot
