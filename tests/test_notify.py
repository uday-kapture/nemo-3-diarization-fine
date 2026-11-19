import io
import json
import urllib.error
from urllib.parse import parse_qs, urlsplit

import pytest

from nemo_3_diarization import notify, tracking

URL = "https://chat.googleapis.com/v1/spaces/AAA/messages?key=secret-key&token=secret-token"


class FakeOpener:
    def __init__(self, *outcomes):
        self.outcomes, self.requests = list(outcomes), []

    def __call__(self, request, timeout):
        self.requests.append(request)
        outcome = self.outcomes.pop(0) if self.outcomes else 200
        if isinstance(outcome, BaseException):
            raise outcome
        if outcome != 200:
            body = json.dumps({"error": {"code": outcome, "message": "Invalid space"}}).encode()
            raise urllib.error.HTTPError(request.full_url, outcome, "error", {}, io.BytesIO(body))
        return io.BytesIO(b"{}")


def _chat(*outcomes, thread="run-1"):
    opener = FakeOpener(*outcomes)
    return notify.GChat(URL, thread, opener=opener, sleep=lambda s: None), opener


def test_posts_text_into_the_run_thread():
    chat, opener = _chat()
    assert chat.send("hello")
    (req,) = opener.requests
    query = parse_qs(urlsplit(req.full_url).query)
    assert query["key"] == ["secret-key"] and query["threadKey"] == ["run-1"]
    assert query["messageReplyOption"] == ["REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD"]
    assert json.loads(req.data) == {"text": "hello"}


def test_retries_rate_limits_and_server_errors_then_gives_up_quietly(capsys):
    chat, opener = _chat(429, 503)
    assert chat.send("x") and len(opener.requests) == 3
    chat, opener = _chat(500, 500, 500)
    assert not chat.send("x") and len(opener.requests) == 3
    chat, opener = _chat(400)  # a bad request is not retried
    assert not chat.send("x") and len(opener.requests) == 1
    assert "HTTP 400: Invalid space" in capsys.readouterr().err
    chat, opener = _chat(TimeoutError(), TimeoutError(), TimeoutError())
    assert not chat.send("x")
    assert "secret" not in capsys.readouterr().err  # the URL holds the key


def test_without_a_webhook_nothing_is_sent(monkeypatch):
    monkeypatch.delenv(notify.WEBHOOK_ENV, raising=False)
    chat = notify.GChat.from_env("run")
    assert not chat.enabled and not chat.send("x")


def test_long_messages_are_cut():
    chat, opener = _chat()
    chat.send("x" * 10_000)
    assert len(json.loads(opener.requests[0].data)["text"]) <= notify.MAX_CHARS


def test_val_text_tracks_best_and_eta():
    first = notify.val_text("r", 250, 3000, 0.40, None, 0.5, 600.0)
    assert "val_loss 0.4000 (new best)" in first and "8%" in first and "ETA 1h 50m" in first
    assert "(best 0.3000)" in notify.val_text("r", 500, 3000, 0.40, 0.30, 0.5, 1200.0)
    assert "ETA" not in notify.val_text("r", 3000, 3000, 0.40, 0.30, None, 7200.0)


def test_failed_text_says_stopped_for_ctrl_c_and_shows_traceback_otherwise():
    assert "stopped at step 7" in notify.failed_text("r", SystemExit(1), 7)
    try:
        raise RuntimeError("CUDA out of memory")
    except RuntimeError as e:
        text = notify.failed_text("r", e, 7)
    assert text.startswith("🔴 *r* failed at step 7: RuntimeError") and "CUDA out of memory" in text


def test_tracking_uri_is_required(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(tracking.TrackingError, match="MLFLOW_TRACKING_URI is not set"):
        tracking.tracking_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.com")
    assert tracking.tracking_uri() == "https://mlflow.example.com"


def test_numeric_drops_nan_none_and_bools():
    assert tracking.numeric({"der": 0.1, "nan": float("nan"), "none": None, "b": True, "n": 3}, "1.04s/") == {
        "1.04s/der": 0.1, "1.04s/n": 3.0}
