"""The only part of the gateway that talks to the outside world.

``OfflineStub`` is the default and sends nothing anywhere, so the whole flow runs
on a laptop with no network. ``AnthropicConnector`` sends cleared text to one
fixed endpoint. It needs ``pip install dragonscope[gateway-api]`` and an API key.
"""

from __future__ import annotations

import re
from typing import Protocol

from dragonscope.gateway.placeholders import CODE

SYSTEM_PROMPT = (
    "You are drafting for a licensed professional, who reviews every draft before use. "
    "Identifiers in the case were replaced with codes such as PERSON_ABCD. Keep each code "
    "exactly as written, do not guess who anyone is, and do not add names, numbers or "
    "dates that are not in the case."
)


class ConnectorError(Exception):
    """The outside model gave no usable answer. Messages never include the prompt."""


class Connector(Protocol):
    name: str

    def complete(self, system: str, prompt: str) -> str: ...


class OfflineStub:
    """Writes a predictable draft from the cleared text, without any network.

    ``invent`` makes it add identifiers that were never in the case, to show the
    re-linker flagging them.
    """

    name = "offline-stub"

    def __init__(self, invent: bool = False):
        self.invent = invent

    def complete(self, system: str, prompt: str) -> str:
        case = prompt.split("Case:", 1)[-1].strip()
        sentences = re.split(r"(?<=[.!?])\s+", case)
        codes = list(dict.fromkeys(CODE.findall(case)))
        subject = codes[0] if codes else "the client"
        lines = [
            f"Draft for {subject}.",
            "Summary: " + " ".join(sentences[:2]),
            "Plan: review the record above and confirm the next steps with the client.",
        ]
        if self.invent:
            lines.append("Contact PERSON_ZZZZ on 0771234567 for follow-up.")
        return "\n".join(lines)


class AnthropicConnector:
    """Sends cleared text to the Claude API, at one fixed endpoint.

    Nothing identifying goes in metadata: the request has a model, a fixed system
    prompt and the cleared text. Use an organisation set up for zero data
    retention where available. That is defence in depth, not the legal basis.
    ``fallbacks="default"`` lets the API retry a declined request on another
    Claude model, which receives the same cleared text.
    """

    name = "anthropic"
    ENDPOINT = "https://api.anthropic.com"
    MODEL = "claude-opus-5-5"

    def __init__(self, client=None, model: str = MODEL, effort: str = "medium"):
        if client is None:
            try:
                import anthropic
            except ImportError:
                raise ConnectorError(
                    "Install the API client: pip install dragonscope[gateway-api]"
                ) from None
            client = anthropic.Anthropic(base_url=self.ENDPOINT)
        self.client = client
        self.model = model
        self.effort = effort

    def complete(self, system: str, prompt: str) -> str:
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                output_config={"effort": self.effort},
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as error:
            # Only the type: an error message could repeat part of the request.
            raise ConnectorError(f"The model call failed ({type(error).__name__}).") from None
        if response.stop_reason == "refusal":
            raise ConnectorError("The model declined this request.")
        text = "".join(block.text for block in response.content if block.type == "text")
        if not text.strip():
            raise ConnectorError("The model returned no text.")
        return text


def make_connector(name: str) -> Connector:
    if name == "stub":
        return OfflineStub()
    if name == "anthropic":
        return AnthropicConnector()
    raise ValueError(f"unknown connector {name!r}: use stub or anthropic")
