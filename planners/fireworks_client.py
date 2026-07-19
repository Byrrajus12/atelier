"""The Fireworks chat-completions client — the network edge, behind a tiny interface.

``FireworksClient`` is an ABC with one method so ``VLMPlanner`` can be constructed with a
fake in tests: no API key, no sockets, no recorded-cassette machinery, and no chance a
test run bills the account. ``HTTPFireworksClient`` is the real implementation and is the
ONLY thing in the planner path that touches the network.

Every transport failure surfaces as ``FireworksError`` so the planner has exactly one
exception type to catch when deciding whether to retry — an HTTP 500, a timeout, and a
non-JSON body are all "the call didn't produce a usable response", and the planner treats
them identically.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict

DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_TIMEOUT = 60.0
API_KEY_ENV = "FIREWORKS_API_KEY"


class FireworksError(RuntimeError):
    """A chat-completion call failed to produce a usable response: transport error,
    timeout, non-2xx status, or an undecodable body."""


class FireworksClient(ABC):
    """The seam between the planner and the network. One method, so a test double is
    trivial to write."""

    @abstractmethod
    def create_chat_completion(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST ``body`` to chat/completions and return the decoded JSON response.
        Raises ``FireworksError`` if no usable response came back."""


class HTTPFireworksClient(FireworksClient):
    """Real client over Fireworks' OpenAI-compatible endpoint.

    The API key is read from the ``FIREWORKS_API_KEY`` environment variable by default
    and is never written to disk or logged — the planner logs request/response content,
    so the key deliberately does not travel in anything it logs.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get(API_KEY_ENV)
        if not key:
            raise ValueError(
                f"no Fireworks API key: pass api_key= or set ${API_KEY_ENV}"
            )
        self._api_key = key
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._timeout = timeout

    def create_chat_completion(self, body: Dict[str, Any]) -> Dict[str, Any]:
        import requests  # imported here so the module is importable without requests

        try:
            resp = requests.post(
                self._url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=self._timeout,
            )
        except requests.RequestException as ex:
            raise FireworksError(f"request to Fireworks failed: {ex}") from ex

        if resp.status_code >= 400:
            # Body text is included because Fireworks puts the actionable detail there
            # (bad model id, rejected tool_choice); truncated so a huge error page can't
            # flood the planner's logs.
            raise FireworksError(
                f"Fireworks returned {resp.status_code}: {resp.text[:500]}"
            )
        try:
            return resp.json()
        except ValueError as ex:
            raise FireworksError(f"Fireworks response was not JSON: {ex}") from ex
