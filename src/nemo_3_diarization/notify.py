"""Google Chat messages about training runs, through a space's incoming webhook (as in nemotron-3.5-asr).

The webhook URL is a secret (anyone holding it can post into the space), so it comes from the GCHAT_WEBHOOK_URL
environment variable, never from a flag or file; without it nothing is sent. Each run posts into its own thread
(threadKey = run name). Messages carry run names, step counts and metrics, never audio or labels. Sending never
raises: a failed post is reported on stderr and training goes on.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
import urllib.error
import urllib.request
from urllib.parse import urlencode

WEBHOOK_ENV = "GCHAT_WEBHOOK_URL"
MAX_CHARS = 4000  # Chat takes far more; longer is unreadable in a thread
TRACEBACK_LINES = 12


class GChat:
    def __init__(self, url: str | None, thread: str | None = None, *, timeout: float = 10.0, retries: int = 2,
                 opener=urllib.request.urlopen, sleep=time.sleep):
        self.url, self.thread, self.timeout, self.retries = url, thread, timeout, retries
        self.opener, self.sleep = opener, sleep

    @classmethod
    def from_env(cls, thread: str | None = None, **kw) -> GChat:
        return cls(os.environ.get(WEBHOOK_ENV) or None, thread, **kw)

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    def endpoint(self) -> str:
        if not self.thread:
            return self.url
        reply = {"threadKey": self.thread, "messageReplyOption": "REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD"}
        return f"{self.url}{'&' if '?' in self.url else '?'}{urlencode(reply)}"

    def send(self, text: str) -> bool:
        if not self.url:
            return False
        if len(text) > MAX_CHARS:
            text = text[: MAX_CHARS - 20] + "\n… (cut)"
        request = urllib.request.Request(
            self.endpoint(), data=json.dumps({"text": text}).encode(),
            headers={"Content-Type": "application/json; charset=UTF-8"}, method="POST",
        )
        for attempt in range(self.retries + 1):
            try:
                with self.opener(request, timeout=self.timeout) as response:
                    response.read()
                return True
            except urllib.error.HTTPError as e:  # the URL holds the key: report the status and Google's reason only
                error = f"HTTP {e.code}{_reason(e)}"
                if e.code != 429 and e.code < 500:
                    break
            except OSError as e:  # URLError, timeouts, resets
                error = type(e).__name__
            if attempt < self.retries:
                self.sleep(2 ** (attempt + 1))
        print(f"gchat: message not sent ({error})", file=sys.stderr)
        return False


def _reason(e: urllib.error.HTTPError) -> str:
    try:
        return ": " + json.loads(e.read())["error"]["message"][:200]
    except Exception:
        return ""


# ---------- message text ----------


def duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    h, m = divmod(seconds // 60, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m"


def started_text(name: str, max_steps: int, batch_size: int, trainable: int, gpu: str) -> str:
    return f"*{name}* started: {max_steps:,} steps, batch {batch_size}, {trainable:,} LoRA params, {gpu}"


def val_text(name: str, step: int, max_steps: int, val_loss: float, best_before: float | None,
             train_loss: float | None, elapsed_s: float) -> str:
    parts = [f"*{name}* step {step:,} / {max_steps:,} ({step / max_steps:.0%})"]
    if best_before is None or val_loss < best_before:
        parts.append(f"val_loss {val_loss:.4f} (new best)")
    else:
        parts.append(f"val_loss {val_loss:.4f} (best {best_before:.4f})")
    if train_loss is not None:
        parts.append(f"train loss {train_loss:.4f}")
    if 0 < step < max_steps:
        parts.append(f"ETA {duration(elapsed_s / step * (max_steps - step))}")
    return " · ".join(parts)


def bad_loss_text(name: str, step: int, loss: float) -> str:
    return f"⚠️ *{name}* loss is {loss} at step {step:,}"


def finished_text(name: str, step: int, best_val_loss: float | None, elapsed_s: float, merged: str) -> str:
    best = f", best val_loss {best_val_loss:.4f}" if best_val_loss is not None else ", no validation ran"
    return f"✅ *{name}* finished at step {step:,}{best} in {duration(elapsed_s)}; merged model: {merged}"


def failed_text(name: str, exc: BaseException, step: int) -> str:
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        return f"⏹️ *{name}* stopped at step {step:,} ({type(exc).__name__}); weights so far are exported"
    tail = "".join(traceback.format_exception(exc)).strip().splitlines()[-TRACEBACK_LINES:]
    return f"🔴 *{name}* failed at step {step:,}: {type(exc).__name__}\n```\n" + "\n".join(tail) + "\n```"
