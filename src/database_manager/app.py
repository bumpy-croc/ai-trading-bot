import logging
import os
from typing import Optional

import sqlalchemy.exc
from flask import (  # type: ignore
    Flask,
    Response,
    jsonify,
    redirect,
    render_template_string,
    request,
    url_for,
)
from flask_admin import Admin  # type: ignore
from flask_admin.contrib.sqla import ModelView  # type: ignore
from flask_login import (  # type: ignore
    LoginManager,
    UserMixin,
    login_required,
    login_user,
    logout_user,
)
from werkzeug.security import check_password_hash, generate_password_hash  # type: ignore
from sqlalchemy.orm import scoped_session  # type: ignore

# Re-use existing database layer
from src.database.manager import DatabaseManager  # type: ignore
from src.database.models import Base  # type: ignore

logger = logging.getLogger(__name__)


def _ensure_secret_key() -> str:
    """Ensure SECRET_KEY is set, exit if not in production.
    
    SEC-003 Fix: Consolidate SECRET_KEY handling for consistent security logic.
    """
    secret_key = os.environ.get("DB_MANAGER_SECRET_KEY")
    if secret_key:
        return secret_key
    
    # Allow fallback only in development
    env = os.getenv("ENV", "development").lower()
    if env in ("development", "test", "testing"):
        logger.warning(
            "⚠️  Using default SECRET_KEY - set DB_MANAGER_SECRET_KEY in production"
        )
        return "dev-key-change-in-production"
    
    logger.error("❌ DB_MANAGER_SECRET_KEY required in production environment")
    raise SystemExit(1)


def _get_admin_credentials() -> tuple[str, str]:
    """Get and validate admin credentials from environment.
    
    SEC-001 Fix: Require DB_MANAGER_ADMIN_PASS environment variable (no fallback).
    """
    admin_username = os.environ.get("DB_MANAGER_ADMIN_USER")
    admin_password = os.environ.get("DB_MANAGER_ADMIN_PASS")
    
    if not admin_username:
        admin_username = "admin"
        logger.info("Using default admin username: 'admin'")
    
    if not admin_password:
        logger.error(
            "❌ DB_MANAGER_ADMIN_PASS environment variable must be set. "
            "Please set a strong password to secure the database manager."
        )
        raise SystemExit(1)
    
    return admin_username, admin_password


class CustomModelView(ModelView):
    """Generic model view with sensible defaults and automatic search columns."""

    can_create = True
    can_edit = True
    can_delete = True
    can_view_details = True
    page_size = 50

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


def create_app() -> Flask:
    """Factory to create and configure the Flask application.

    The function attempts to establish a database connection via
    :class:`database.manager.DatabaseManager`. If the connection fails, the
    application will still start but only expose a generic error endpoint so
    that deployment health checks can continue to function while signalling a
    500 status to callers.
    """

    # Consolidated SECRET_KEY handling (SEC-003 Fix)
    app_secret_key = _ensure_secret_key()

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
        app.config["SECRET_KEY"] = app_secret_key

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
    app.config["SECRET_KEY"] = app_secret_key

    # --- Flask-Login setup for admin authentication ---
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = "login"

    # Get and validate admin credentials (SEC-001 Fix)
    ADMIN_USERNAME, ADMIN_PASSWORD = _get_admin_credentials()
    
    # Hash password for secure storage (SEC-002 Fix)
    ADMIN_PASSWORD_HASH = generate_password_hash(ADMIN_PASSWORD, method="pbkdf2:sha256")
    logger.info(f"✅ Database manager admin authentication configured for user: {ADMIN_USERNAME}")

    @login_manager.user_loader
    def load_user(user_id):
        if user_id == ADMIN_USERNAME:
            return AdminUser(user_id)
        return None

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            username = request.form.get("username")
            password = request.form.get("password")
            # SEC-002 Fix: Use secure password hash comparison (timing-attack resistant)
            if username == ADMIN_USERNAME and check_password_hash(ADMIN_PASSWORD_HASH, password):
                user = AdminUser(username)
                login_user(user)
                logger.info(f"✅ Admin login successful for user: {username}")
                return redirect(request.args.get("next") or url_for("admin.index"))
            
            logger.warning(f"⚠️  Failed admin login attempt for user: {username}")
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
