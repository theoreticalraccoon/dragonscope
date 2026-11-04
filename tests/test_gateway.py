"""The split-brain gateway: de-identify, gate, send, re-link, review, erase."""

import json
from random import Random

import pytest
import torch
from fastapi.testclient import TestClient

from dragonscope.gateway import (
    AnthropicConnector,
    AuditLog,
    ConnectorError,
    Gateway,
    GatewayError,
    LeakGate,
    OfflineStub,
    PlaceholderMap,
    deidentify,
    identify,
)
from dragonscope.gateway import samples
from dragonscope.gateway.gate import format_layer, lexicon_layer, protected_items
from dragonscope.gateway.identifiers import generalise_quasi
from dragonscope.gateway.server import create_app
from dragonscope.model_config import demo_config
from dragonscope.vendor.bdh.bdh import BDH

NAME = "Dilani Wickramasinghe"
NIC = "199012345678"
CARD = "4111 1111 1111 1111"
PHONE = "077 123 4567"
CASE = (
    f"Patient: {NAME}, NIC {NIC}, a 47-year-old Minister of Fisheries from Galle, "
    f"was admitted on 12 March 2026 with chest pain. Phone {PHONE}. Card {CARD}. "
    "Treated with aspirin and discharged on 15 March 2026."
)
SECRETS = [NAME, "Wickramasinghe", NIC, CARD, PHONE, "Galle", "12 March"]


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return BDH(demo_config()).eval()


def kinds(text):
    return [found.kind for found in identify(text)]


# De-identifier


def test_each_identifier_type_is_found():
    text = (
        f"Name: {NAME}. NIC {NIC}. Card {CARD}. Phone +94 77 123 4567. "
        "Email d.w@mail.lk. MRN: BHT-2026-0042. Lives in Kandy. Born 3/4/1950. "
        "Mother aged 93. Account no 7781234500."
    )
    found = set(kinds(text))
    assert {"name", "nic", "card", "phone", "email", "record", "place", "date", "age_90"} <= found


def test_dates_become_years_and_old_ages_90_plus():
    cleared, _ = deidentify("Admitted on 12 March 2026, aged 93.", PlaceholderMap())
    assert cleared == "Admitted in 2026, aged 90+."


def test_cards_that_fail_luhn_are_not_cards():
    assert "card" not in kinds("Card 4111 1111 1111 1112.")


def test_a_name_part_gets_the_full_names_code():
    placeholders = PlaceholderMap()
    cleared, _ = deidentify(f"Patient: {NAME}. Mr Wickramasinghe is stable.", placeholders)
    codes = placeholders.codes()
    assert len(codes) == 1
    assert cleared.count(codes[0]) == 2


def test_quasi_identifiers_generalise_in_two_levels():
    text = "A 47-year-old Minister of Fisheries with Wilson's disease."
    first, changed = generalise_quasi(text, 1)
    assert first == "A 47-year-old public official with a rare metabolic condition."
    assert sorted(f.kind for f in changed) == ["condition", "occupation"]
    second, _ = generalise_quasi(first, 2)
    assert second == "A 40-49-year-old public official with a rare metabolic condition."


# Placeholders


def test_codes_are_random_letters_and_differ_between_requests():
    first, second = PlaceholderMap(), PlaceholderMap()
    a = first.code_for("card", CARD)
    b = second.code_for("card", CARD)
    assert a != b or first.code_for("card", "x") != second.code_for("card", "x")
    assert a.startswith("CARD_") and a[5:].isalpha() and a[5:].isupper()
    assert first.code_for("card", CARD) == a  # Stable within one request.


def test_relink_restores_values_and_flags_invented_ones():
    placeholders = PlaceholderMap()
    code = placeholders.code_for("name", NAME)
    relinked = placeholders.relink(f"Summary for {code}. Call 0771234567 or PERSON_ZZZZ.")
    assert NAME in relinked.text
    assert relinked.invented == ["phone"]
    assert relinked.unknown_codes == 1


def test_destroy_empties_the_map():
    placeholders = PlaceholderMap()
    code = placeholders.code_for("nic", NIC)
    placeholders.destroy()
    assert len(placeholders) == 0
    assert placeholders.relink(code).text == code


# Gate


def test_gate_blocks_text_that_still_holds_an_identifier():
    assert not format_layer(f"Card {CARD}").passed
    assert not format_layer("Ref 1234567").passed
    assert not format_layer("Seen on 3 May 2026").passed
    assert format_layer("Seen in 2026, aged 90+, dose 500 mg.").passed
    assert not lexicon_layer("Dr Fonseka saw PERSON_ABCD.", []).passed
    assert not lexicon_layer("the patient Wijesekara", ["Wijesekara"]).passed
    assert lexicon_layer("PERSON_ABCD was seen in PLACE_WXYZ.", []).passed
    result = LeakGate().clear(f"Card {CARD} was used.")
    assert not result.allowed


def test_gate_generalises_at_most_twice_then_needs_a_person():
    gate = LeakGate(weights={"year": 1.0, "occupation": 2.0}, risk_threshold=1.0)
    result = gate.clear("A judge was admitted in 2025 and seen again in 2026.")
    assert result.rounds == 2
    assert result.needs_review
    assert not result.allowed
    assert "legal professional" in result.text


def test_memory_layer_reports_without_blocking_unless_asked(model):
    text = "Patient: Nimal Perera, the Minister of Fisheries, was admitted."
    items, values = protected_items(text)
    pairs = list(enumerate(text.encode()))
    outgoing = "The patient leads the fisheries ministry."
    reporting = LeakGate(model, memory_threshold=0.0).clear(outgoing, values, pairs, items)
    memory = reporting.layers[3]
    assert memory.name == "memory" and memory.score is not None and memory.score > 0
    assert not memory.passed and reporting.allowed
    blocking = LeakGate(model, block_on_memory=True, memory_threshold=0.0)
    assert not blocking.clear(outgoing, values, pairs, items).allowed


# Audit


def test_audit_entries_hold_no_values_and_detect_tampering(model):
    gateway = Gateway(model)
    gateway.open_case("c1", CASE)
    gateway.submit("c1", "Write a discharge summary.", "Dr A")
    log = gateway.audit
    unchained = json.dumps([{k: v for k, v in e.items() if k != "sent_text"} for e in log.entries])
    for secret in SECRETS:
        assert secret not in unchained
    assert log.verify()
    log.entries[0]["bytes"] += 1
    assert not log.verify()


# Pipeline


def test_end_to_end_only_placeholders_leave(model):
    gateway = Gateway(model)
    gateway.open_case("c1", CASE)
    draft = gateway.submit("c1", f"Write a discharge summary for {NAME}.", "Dr A, attending")
    assert draft.status == "pending"
    for secret in SECRETS:
        assert secret not in draft.sent_text
    assert "public official" in draft.sent_text
    assert NAME in draft.answer
    withheld = {item["kind"] for item in draft.withheld}
    assert {"name", "nic", "card", "phone", "place", "date", "occupation"} <= withheld
    assert {"kind": "card", "last4": "1111"} in draft.withheld
    decided = gateway.decide(draft.id, "Dr B", approve=True)
    assert decided.status == "approved" and decided.reviewer == "Dr B"
    with pytest.raises(GatewayError):
        gateway.decide(draft.id, "Dr B", approve=False)


def test_a_blocked_request_sends_nothing(model):
    class Recorder(OfflineStub):
        calls = 0

        def complete(self, system, prompt):
            Recorder.calls += 1
            return super().complete(system, prompt)

    gate = LeakGate(model, block_on_memory=True, memory_threshold=0.0)
    gateway = Gateway(model, connector=Recorder(), gate=gate)
    gateway.open_case("c1", CASE)
    draft = gateway.submit("c1", "Summarise.", "Dr A")
    assert draft.status == "blocked"
    assert draft.sent_text is None and Recorder.calls == 0


def test_invented_identifiers_are_flagged(model):
    gateway = Gateway(model, connector=OfflineStub(invent=True))
    gateway.open_case("c1", CASE)
    draft = gateway.submit("c1", "Summarise.", "Dr A")
    assert draft.invented == ["phone"]
    assert draft.unknown_codes == 1


def test_forget_one_fact_rebuilds_memory_and_scrubs_drafts(model):
    gateway = Gateway(model)
    gateway.open_case("c1", CASE)
    draft = gateway.submit("c1", f"Summarise for {NAME}.", "Dr A")
    result = gateway.forget_fact("c1", NAME)
    assert result["copies"] == 1 and result["residual"] < 1e-5
    assert NAME not in gateway.cases["c1"].text
    assert NAME not in draft.answer and NAME not in draft.task


def test_erasing_a_case_leaves_no_copy(model, tmp_path):
    audit_file = tmp_path / "audit.jsonl"
    gateway = Gateway(model, audit=AuditLog(audit_file))
    gateway.open_case("c1", CASE)
    gateway.submit("c1", f"Summarise for {NAME}.", "Dr A")
    gateway.erase_case("c1")
    dumped = json.dumps(gateway.state()) + audit_file.read_text(encoding="utf-8")
    for secret in SECRETS + ["4111", "PERSON_", "public official"]:
        assert secret not in dumped
    assert gateway.cases == {} and gateway.drafts == {}
    assert gateway.audit.verify()
    assert gateway.audit.entries[-1]["event"] == "erase"


def test_case_references_and_sizes_are_checked(model):
    gateway = Gateway(model)
    for bad in ("", "has space", "199012345678", "x" * 40):
        with pytest.raises(GatewayError):
            gateway.open_case(bad, CASE)
    with pytest.raises(GatewayError):
        gateway.open_case("c1", "x" * 3000)
    with pytest.raises(GatewayError):
        gateway.submit("missing", "Summarise.", "Dr A")


# Connector


def test_anthropic_connector_sends_only_the_cleared_text():
    class Block:
        type = "text"
        text = "Draft for PERSON_ABCD."

    class Response:
        stop_reason = "end_turn"
        content = [Block()]

    class Messages:
        def create(self, **request):
            self.request = request
            return Response()

    class Client:
        class beta:
            messages = Messages()

    client = Client()
    connector = AnthropicConnector(client=client)
    assert connector.complete("system", "Task: x\n\nCase:\nPERSON_ABCD") == "Draft for PERSON_ABCD."
    request = client.beta.messages.request
    assert request["model"] == "claude-opus-5-5"
    assert request["messages"] == [{"role": "user", "content": "Task: x\n\nCase:\nPERSON_ABCD"}]
    assert "metadata" not in request

    Response.stop_reason = "refusal"
    with pytest.raises(ConnectorError):
        connector.complete("system", "prompt")


# Samples


def test_samples_are_labelled_and_mostly_cleared():
    for sample in samples.build(12, seed=1):
        cleared, _ = deidentify(sample.text, PlaceholderMap())
        for kind, value in sample.labels:
            if kind != "name":
                assert value not in cleared, kind


# Console API


def test_console_flow_and_origin_check(model):
    gateway = Gateway(model)
    client = TestClient(create_app(gateway))
    assert "DragonScope Gateway" in client.get("/").text
    payload = {"case_ref": "c1", "case_text": CASE, "task": "Summarise.", "requester": "Dr A"}
    draft = client.post("/api/submit", json=payload).json()["draft"]
    assert draft["status"] == "pending"

    refused = client.post("/api/submit", json=payload, headers={"origin": "http://evil.example"})
    assert refused.status_code == 403

    bad = client.post("/api/submit", json={**payload, "case_ref": NIC})
    assert bad.status_code == 400 and NIC not in bad.text

    reviewed = client.post("/api/review", json={"draft": draft["id"], "decision": "approve", "reviewer": "Dr B"})
    assert reviewed.json()["draft"]["status"] == "approved"
    assert client.post("/api/erase", json={"case_ref": "c1"}).json() == {"erased": True}
    assert NAME not in client.get("/api/state").text


def test_red_team_set_is_well_formed():
    case, paraphrases, controls = samples.red_team(Random(0))
    assert "Minister of Fisheries" in case
    for text in paraphrases + controls:
        assert format_layer(text).passed and lexicon_layer(text, []).passed
