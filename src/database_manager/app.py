import os
import logging
from typing import List

from flask import Flask, jsonify
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView
from sqlalchemy.orm import scoped_session

# Re-use existing database layer
from database.manager import DatabaseManager  # type: ignore
from database.models import Base  # type: ignore

logger = logging.getLogger(__name__)


class CustomModelView(ModelView):
    """Generic model view with sensible defaults and automatic search columns."""

    can_create = True
    can_edit = True
    can_delete = True
    can_view_details = True
    page_size = 50

    def __init__(self, model, session, **kwargs):
        # Dynamically determine searchable string columns
        searchable: List[str] = []
        for column in model.__table__.columns:  # type: ignore[attr-defined]
            # Some column types (e.g. JSON) do not expose python_type
            try:
                if hasattr(column.type, "python_type") and column.type.python_type is str:
                    searchable.append(column.name)
            except NotImplementedError:
                # Skip columns that don't implement python_type
                continue
        kwargs.setdefault("column_searchable_list", searchable)
        super().__init__(model, session, **kwargs)


def create_app() -> Flask:
    """Factory to create Flask application for the database manager."""

    # Initialise database manager (re-uses existing connection settings)
    db_manager = DatabaseManager()
    engine = db_manager.engine
    if engine is None or db_manager.session_factory is None:
        raise RuntimeError("DatabaseManager failed to initialise engine/session factory.")

    # Scoped session ensures each request gets its own session
    db_session = scoped_session(db_manager.session_factory)

    # Create Flask app
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("DB_MANAGER_SECRET_KEY", "dev-key-change-in-production")

    # Set up Flask-Admin
    admin = Admin(app, name="Database Manager", template_mode="bootstrap4")

    # Dynamically register all models declared in Base metadata
    for mapper in Base.registry.mappers:
        model_class = mapper.class_
        admin.add_view(CustomModelView(model_class, db_session))

    # Basic health route
    @app.route("/health")
    def health_check():
        return {"status": "ok"}

    # Database info route (helpful for debugging)
    @app.route("/db_info")
    def db_info():
        return jsonify(db_manager.get_database_info())

    # Simple schema "migration" route to ensure new tables are created
    @app.route("/migrate", methods=["POST", "GET"])
    def migrate():
        """Synchronise database schema (creates any missing tables)."""
        try:
            Base.metadata.create_all(engine)
            return {"status": "success", "message": "Schema synchronised."}
        except Exception as exc:  # pragma: no cover
            logger.exception("Schema migration failed: %s", exc)
            return {"status": "error", "message": str(exc)}, 500

    # Clean up DB sessions after each request
    @app.teardown_appcontext
    def shutdown_session(exception=None):  # noqa: D401, pylint: disable=unused-argument
        db_session.remove()

    return app


if __name__ == "__main__":
    application = create_app()
    port = int(os.environ.get("PORT", 8000))
    application.run(host="0.0.0.0", port=port)