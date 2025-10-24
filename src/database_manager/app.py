import logging
import os
import time
from typing import Optional

import sqlalchemy.exc
from flask import (  # type: ignore
    Flask,
    Response,
    jsonify,
    redirect,
    render_template_string,
    request,
    session,
    url_for,
)
from flask_admin import Admin  # type: ignore
from flask_admin.contrib.sqla import ModelView  # type: ignore
from flask_admin.form import SecureForm  # type: ignore
from flask_limiter import Limiter  # type: ignore
from flask_limiter.util import get_remote_address  # type: ignore
from flask_login import (  # type: ignore
    LoginManager,
    UserMixin,
    login_required,
    login_user,
    logout_user,
)
from flask_wtf.csrf import CSRFProtect  # type: ignore
from sqlalchemy.orm import scoped_session  # type: ignore

# Re-use existing database layer
from src.database.manager import DatabaseManager  # type: ignore
from src.database.models import Base  # type: ignore

logger = logging.getLogger(__name__)


class CustomModelView(ModelView):
    """Generic model view with sensible defaults and automatic search columns."""

    can_create = True
    can_edit = True
    can_delete = True
    can_view_details = True
    page_size = 50
    form_base_class = SecureForm

    def __init__(self, model, session, **kwargs):
        # Dynamically determine searchable string columns
        searchable: list[str] = []
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


class AdminUser(UserMixin):
    def __init__(self, id):
        self.id = id


# SEC-008: CSRF Protection
csrf = CSRFProtect()

# SEC-009: Rate Limiting
limiter = Limiter(key_func=get_remote_address)


def create_app() -> Flask:
    """Factory to create and configure the Flask application.

    The function attempts to establish a database connection via
    :class:`database.manager.DatabaseManager`. If the connection fails, the
    application will still start but only expose a generic error endpoint so
    that deployment health checks can continue to function while signalling a
    500 status to callers.
    """

    # Enforce SECRET_KEY from env
    secret_key = os.environ.get("DB_MANAGER_SECRET_KEY")
    if not secret_key:
        logger.error("❌ Environment variable 'DB_MANAGER_SECRET_KEY' is not set. Exiting.")
        raise SystemExit(1)

    # Initialise database manager (re-uses existing connection settings)
    try:
        db_manager = DatabaseManager()
    except (
        sqlalchemy.exc.OperationalError
    ) as exc:  # pragma: no cover – we want to surface the error in logs
        logger.exception(
            "❌ Failed to initialise DatabaseManager due to a database connection error: %s", exc
        )
        app = Flask(__name__)
        from src.utils.secrets import get_secret_key

        app.config["SECRET_KEY"] = get_secret_key(env_var="DB_MANAGER_SECRET_KEY")

        @app.route("/db_error")
        def db_error() -> tuple[dict[str, str], int]:
            """Endpoint returned when the database is unavailable."""
            return {"status": "error", "message": "Database connection failed."}, 500

        return app

    engine = db_manager.engine
    if engine is None or db_manager.session_factory is None:
        raise RuntimeError("DatabaseManager failed to initialise engine/session factory.")

    # Scoped session ensures each request gets its own session
    db_session = scoped_session(db_manager.session_factory)

    # Create Flask app
    app = Flask(__name__)
    from src.utils.secrets import get_secret_key

    app.config["SECRET_KEY"] = get_secret_key(env_var="DB_MANAGER_SECRET_KEY")

    # Register global security extensions
    csrf.init_app(app)
    limiter.init_app(app)

    # --- Flask-Login setup for admin authentication ---
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = "login"

    ADMIN_USERNAME = os.environ.get("DB_MANAGER_ADMIN_USER", "admin")
    ADMIN_PASSWORD = os.environ.get("DB_MANAGER_ADMIN_PASS", "password")

    @login_manager.user_loader
    def load_user(user_id):
        if user_id == ADMIN_USERNAME:
            return AdminUser(user_id)
        return None

    @app.route("/login", methods=["GET", "POST"])
    @csrf.exempt
    def login():
        if request.method == "POST":
            username = request.form.get("username")
            password = request.form.get("password")
            if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                user = AdminUser(username)
                login_user(user)
                return redirect(request.args.get("next") or url_for("admin.index"))
            return render_template_string(
                """
                <h3>Login Failed</h3>
                <form method='post'>
                    <input name='username' placeholder='Username'><br>
                    <input name='password' type='password' placeholder='Password'><br>
                    <input type='submit' value='Login'>
                </form>
                <p>Invalid credentials.</p>
            """
            )
        return render_template_string(
            """
            <form method='post'>
                <input name='username' placeholder='Username'><br>
                <input name='password' type='password' placeholder='Password'><br>
                <input type='submit' value='Login'>
            </form>
        """
        )

    @app.route("/logout")
    @login_required
    def logout():
        logout_user()
        return redirect(url_for("login"))

    # Set up Flask-Admin
    admin = Admin(app, name="Database Manager", template_mode="bootstrap4")

    # Dynamically register all models declared in Base metadata
    for mapper in Base.registry.mappers:
        model_class = mapper.class_
        admin.add_view(CustomModelView(model_class, db_session))

    # Basic health route
    @app.route("/health")
    def health_check() -> dict[str, str]:
        """Simple health-check endpoint used by load-balancers and uptime monitors."""
        return {"status": "ok"}

    # Database info route (helpful for debugging)
    @app.route("/db_info")
    def db_info() -> Response:
        """Return live connection-pool statistics and configuration details."""
        return jsonify(db_manager.get_database_info())

    # Simple schema "migration" route to ensure new tables are created
    @app.route("/migrate", methods=["POST", "GET"])
    @csrf.exempt
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
    def shutdown_session(
        exception: Optional[Exception] = None,
    ) -> None:  # noqa: D401, pylint: disable=unused-argument
        """Remove the scoped SQLAlchemy session to avoid connection leaks."""
        db_session.remove()

    return app


if __name__ == "__main__":
    application = create_app()
    port = int(os.environ.get("PORT", 8000))
    application.run(host="0.0.0.0", port=port)  # nosec B104: intended for containerized deployment
