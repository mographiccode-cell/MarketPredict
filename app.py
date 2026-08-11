from __future__ import annotations

"""Vercel-safe MarketPredict FastAPI entrypoint.

The full website lives in marketpredict_app.py. This small entrypoint prevents the
native ML runtime from being imported during the serverless cold start. The real
V5 predictor is imported only when a prediction (or model health check) actually
needs it.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP_FILE = ROOT / "marketpredict_app.py"


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
    "__name__": "marketpredict_serverless_app",
    "__package__": None,
    "CampaignPredictor": CampaignPredictor,
    "InputValidationError": InputValidationError,
}
exec(compile(source, str(APP_FILE), "exec"), namespace)
app = namespace["app"]
