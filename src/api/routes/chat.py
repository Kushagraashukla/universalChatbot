"""Chat endpoint and public app info."""

from fastapi import APIRouter, Header, Request

from src.api import service
from src.api.schemas import AppInfo, ChatRequest, ChatResponse, ErrorResponse
from src.apps import get_app

router = APIRouter(prefix="/api/v1", tags=["chat"])

ERRORS = {code: {"model": ErrorResponse} for code in (400, 401, 403, 404, 422, 500, 503)}


# A plain `def` (not async): FastAPI runs it in a worker thread, so a slow LLM call doesn't block other requests.
@router.post("/chat", response_model=ChatResponse, responses=ERRORS)
def chat(body: ChatRequest, request: Request, x_api_key: str | None = Header(None)):
    return service.answer(body, request.headers.get("origin"), x_api_key, request.state.request_id)


@router.get("/apps/{app_id}", response_model=AppInfo, responses=ERRORS)
def app_info(app_id: str):
    """Public, browser-safe details (name, welcome text, example questions). No paths or settings."""
    app = get_app(app_id)
    if app is None:
        raise service.ApiError(404, "unknown_app", "Unknown or disabled app_id.")
    return app.public_info()
