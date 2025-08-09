import os


def get_secret_key(env_var: str = "FLASK_SECRET_KEY", *, allow_default_in_dev: bool = True) -> str:
    value = os.getenv(env_var)
    env = os.getenv("ENV", os.getenv("FLASK_ENV", "development")).lower()
    if value:
        return value
    if allow_default_in_dev and env in {"dev", "development", "test", "testing"}:
        return "dev-key-change-in-production"
    raise RuntimeError(f"Missing required secret: {env_var}. Set it in environment for non-development environments.")