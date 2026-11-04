"""A split-brain gateway: BDH keeps identifiers on-premises, an outside model reasons.

    from dragonscope.gateway import Gateway
    gateway = Gateway(model)
    gateway.open_case("case-1", record_text)
    draft = gateway.submit("case-1", "Write a discharge summary.", "Dr A, attending")

A prototype. See docs/GATEWAY.md for what is built, what is measured and what
is still a gap. It is not legal advice and has not been reviewed by lawyers.
"""

from dragonscope.gateway.audit import AuditLog
from dragonscope.gateway.connector import AnthropicConnector, ConnectorError, OfflineStub
from dragonscope.gateway.gate import LeakGate
from dragonscope.gateway.identifiers import Identifier, identify
from dragonscope.gateway.pipeline import Draft, Gateway, GatewayError
from dragonscope.gateway.placeholders import PlaceholderMap, deidentify

__all__ = [
    "AnthropicConnector",
    "AuditLog",
    "ConnectorError",
    "Draft",
    "Gateway",
    "GatewayError",
    "Identifier",
    "LeakGate",
    "OfflineStub",
    "PlaceholderMap",
    "deidentify",
    "identify",
]
