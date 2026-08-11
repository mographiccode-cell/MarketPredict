from __future__ import annotations

"""Vercel-safe MarketPredict entrypoint.

The main application historically imports the ML runtime at module import time. On a
serverless cold start, a native ML loader error would therefore crash the entire
function before /health or /login could respond. This entrypoint injects a lazy
CampaignPredictor proxy and executes the same app.py without changing website
routes, templates, authentication, or prediction behavior.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_FILE = ROOT / "app.py"


class CampaignPredictor:
    def __new__(cls, *args, **kwargs):
        from predictor import CampaignPredictor as RealCampaignPredictor
        return RealCampaignPredictor(*args, **kwargs)


class InputValidationError(ValueError):
    pass


source = APP_FILE.read_text(encoding="utf-8")
source = source.replace(
    "from predictor import CampaignPredictor, InputValidationError\n",
    "",
    1,
)
namespace = {
    "__file__": str(APP_FILE),
    "__name__": "marketpredict_vercel_app",
    "__package__": None,
    "CampaignPredictor": CampaignPredictor,
    "InputValidationError": InputValidationError,
}
exec(compile(source, str(APP_FILE), "exec"), namespace)
app = namespace["app"]
