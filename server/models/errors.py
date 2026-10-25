"""Turning provider failures into short messages for the glasses. Shared by every model adapter."""

INVALID_KEY = "Server API key is invalid."
BUSY = "Busy right now. Try again."
QUOTA = "AI quota used up. Check API billing."
SERVICE = "AI service error. Try again."
SLOW = "AI is slow right now. Try again."
OFFLINE = "Can't reach the AI. Check WiFi."


def for_status(code: int) -> str:
    if code in (401, 403):
        return INVALID_KEY
    if code == 429:
        return BUSY
    return SERVICE
