"""Open-redirect protection for the admin login `next` parameter."""

import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
from flask import Blueprint

from src.database.admin_ui.app import is_safe_redirect_url

pytestmark = pytest.mark.fast


@pytest.mark.parametrize(
    "target",
    [
        "/",
        "/admin",
        "/admin/",
        "/admin/user/?page=2&sort=1",
        "/dashboard#section",
        "/path/with%20space",
        "/a//b",  # double slash inside the path is harmless; only a leading one is not
    ],
)
def test_same_origin_relative_paths_are_allowed(target):
    assert is_safe_redirect_url(target) is True


@pytest.mark.parametrize(
    "target",
    [
        None,
        "",
        "admin",  # relative without a leading slash
        " /admin",
        "https://evil.com",
        "http://evil.com/admin",
        "ftp://evil.com",
        "javascript:alert(1)",
        "data:text/html,x",
        "//evil.com",
        "///evil.com",
        "////evil.com/x",
        "/\\evil.com",  # backslash bypass: browsers turn "\" into "/"
        "/\\/evil.com",
        "\\\\evil.com",
        "\\/evil.com",
        "/admin\\..\\evil",
        "/\t/evil.com",  # browsers strip tab/newline from Location values
        "/\n/evil.com",
        "/\r/evil.com",
        "/\x00/evil.com",
        "/%5Cevil.com",  # encoded backslash
        "/%5cevil.com",
        "/%2F/evil.com",  # encoded second slash
        "/%2f/evil.com",
        "/%255Cevil.com",  # double-encoded backslash
        "/%25255Cevil.com",
        "/%09/evil.com",  # encoded tab
        "/%0a/evil.com",
        "/%252F/evil.com",
        "/%25252F%25252Fevil.com",
        "/%2525252525252Fevil.com",  # encoding depth beyond the decode limit
    ],
)
def test_off_origin_or_ambiguous_targets_are_rejected(target):
    assert is_safe_redirect_url(target) is False


@pytest.mark.parametrize("target", [123, b"/admin", ["/admin"], object()])
def test_non_string_targets_are_rejected(target):
    assert is_safe_redirect_url(target) is False


@pytest.fixture
def admin_client(monkeypatch):
    """Flask test client for the real login route, with DB and flask-admin stubbed out."""

    class DummyDatabaseManager:
        def __init__(self):
            self.engine = object()
            self.session_factory = lambda: None

        def get_database_info(self):
            return {"status": "ok"}

    class DummyScopedSession:
        def remove(self):
            return None

    class DummyAdmin:
        def __init__(self, app, *args, **kwargs):
            blueprint = Blueprint("admin", __name__)
            blueprint.add_url_rule("/admin/", "index", lambda: "admin index")
            app.register_blueprint(blueprint)

        def add_view(self, view):
            return None

    class DummyModelView:
        def __init__(self, *args, **kwargs):
            pass

    flask_admin_module = ModuleType("flask_admin")
    flask_admin_contrib = ModuleType("flask_admin.contrib")
    flask_admin_sqla = ModuleType("flask_admin.contrib.sqla")
    flask_admin_form = ModuleType("flask_admin.form")
    flask_admin_module.Admin = DummyAdmin
    flask_admin_contrib.sqla = flask_admin_sqla
    flask_admin_sqla.ModelView = DummyModelView
    flask_admin_form.SecureForm = type("DummySecureForm", (), {})

    class DummyLoginManager:
        login_view = None

        def init_app(self, app):
            return None

        def user_loader(self, callback):
            return callback

    flask_login_module = ModuleType("flask_login")
    flask_login_module.LoginManager = DummyLoginManager
    flask_login_module.UserMixin = type("DummyUserMixin", (), {})
    flask_login_module.current_user = SimpleNamespace(is_authenticated=False)
    flask_login_module.login_required = lambda func: func
    flask_login_module.login_user = lambda user: None
    flask_login_module.logout_user = lambda: None

    dummy_base = SimpleNamespace(
        registry=SimpleNamespace(mappers=[]),
        metadata=SimpleNamespace(create_all=lambda engine: None),
    )

    env = {
        "DB_MANAGER_ADMIN_USER": "admin",
        "DB_MANAGER_ADMIN_PASS": "super-secure",
        "DB_MANAGER_SECRET_KEY": "secret",
        "ENV": "development",
    }
    with patch.dict(os.environ, env, clear=False):
        monkeypatch.setitem(sys.modules, "flask_admin", flask_admin_module)
        monkeypatch.setitem(sys.modules, "flask_admin.contrib", flask_admin_contrib)
        monkeypatch.setitem(sys.modules, "flask_admin.contrib.sqla", flask_admin_sqla)
        monkeypatch.setitem(sys.modules, "flask_admin.form", flask_admin_form)
        monkeypatch.setitem(sys.modules, "flask_login", flask_login_module)

        import src.database.admin_ui.app as app_module

        monkeypatch.setattr(app_module, "DatabaseManager", DummyDatabaseManager)
        monkeypatch.setattr(app_module, "scoped_session", lambda factory: DummyScopedSession())
        monkeypatch.setattr(app_module, "Base", dummy_base)

        app = app_module.create_app()
        with app.test_client() as client:
            yield client


def _login(client, next_url):
    return client.post(
        "/login",
        query_string={"next": next_url},
        data={"username": "admin", "password": "super-secure"},
    )


def test_login_follows_safe_next_target(admin_client):
    response = _login(admin_client, "/admin/user/?page=2")

    assert response.status_code == 302
    assert response.headers["Location"] == "/admin/user/?page=2"


@pytest.mark.parametrize(
    "next_url",
    ["//evil.com", "/\\evil.com", "https://evil.com", "/%5Cevil.com", "/\t/evil.com"],
)
def test_login_ignores_unsafe_next_target(admin_client, next_url):
    response = _login(admin_client, next_url)

    assert response.status_code == 302
    location = response.headers["Location"]
    assert location == "/admin/"
