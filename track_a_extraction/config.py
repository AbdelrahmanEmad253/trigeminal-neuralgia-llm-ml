"""Settings read from environment variables (stdlib only). No keys are stored in the repo."""
import os

PROVIDERS = os.environ.get("LLM_PROVIDERS", "openrouter,groq")
SLEEP_SECONDS = float(os.environ.get("LLM_SLEEP_SECONDS", "3"))
COOLDOWN_SECONDS = int(os.environ.get("LLM_COOLDOWN_SECONDS", "900"))
NCBI_EMAIL = os.environ.get("NCBI_EMAIL", "your_email@example.com")


def key_and_model(provider: str) -> tuple[str | None, str | None]:
    """Return (api key, model id) from <PROVIDER>_API_KEY / <PROVIDER>_MODEL; None if unset."""
    p = provider.upper()
    return os.environ.get(f"{p}_API_KEY"), os.environ.get(f"{p}_MODEL")
