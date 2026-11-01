from dragonscope.detect import digits_of, find_in_reply, find_stated

VALID_CARD = "4111 1111 1111 1111"


def test_stated_card_needs_luhn():
    assert [d.kind for d in find_stated(f"My card number is {VALID_CARD}.")] == ["card"]
    assert find_stated("My card number is 4111 1111 1111 1112.") == []


def test_stated_card_formats():
    for card in ("4111-1111-1111-1111", "4111111111111111"):
        found = find_stated(f"card {card}.")
        assert [(d.kind, d.start, d.end) for d in found] == [("card", 5, 5 + len(card))]


def test_stated_nic_formats():
    for nic in ("123456789V", "123456789X", "200012345678"):
        assert [d.kind for d in find_stated(f"My NIC is {nic}.")] == ["nic"]
    for wrong in ("12345678V", "1234567890V", "12345678901", "123456789A"):
        assert find_stated(f"My NIC is {wrong}.") == []


def test_card_is_not_also_a_nic():
    assert [d.kind for d in find_stated(VALID_CARD.replace(" ", ""))] == ["card"]


def test_reply_catches_a_garbled_card_that_fails_luhn():
    reply = "Your card number is 4711 1111 1111 5111."
    found = find_in_reply(reply)
    assert found.kind == "card"
    assert found.luhn_valid is False
    assert reply[found.start : found.end] == "4711 1111 1111 5111"


def test_reply_catches_wrong_lengths_and_stray_characters():
    reply = "Your card number is 4711 11A1 111."
    found = find_in_reply(reply)
    assert found.kind == "card"
    assert reply[found.start : found.end] == "4711 11A1 111"


def test_reply_includes_nic_suffix():
    reply = "Your NIC is 123456789V."
    found = find_in_reply(reply)
    assert found.kind == "nic"
    assert reply[found.start : found.end] == "123456789V"


def test_reply_ignores_ordinary_text():
    assert find_in_reply("We are open 9 to 5 on weekdays.") is None
    assert find_in_reply("I don't have that on record.") is None


def test_unattributed_number_in_a_reply():
    assert find_in_reply("Reference 5551234.").kind == "number"


def test_digits_of():
    assert digits_of("4111-1111 x") == "41111111"
