from random import Random

import numpy as np

from dragonscope.data import (
    KIND_RECALL,
    KIND_REFUSAL,
    MAX_BYTES,
    REFUSAL,
    STATES,
    build_conversation,
    build_dataset,
    luhn_valid,
    random_card,
    split_names,
)


def conversation_at(dataset, index):
    begin, end = dataset["offsets"][index], dataset["offsets"][index + 1]
    return dataset["bytes"][begin:end].tobytes().decode("ascii")


def test_cards_pass_luhn():
    rng = Random(1)
    assert all(luhn_valid(random_card(rng)) for _ in range(500))


def test_luhn_rejects_a_changed_digit():
    assert luhn_valid("4111 1111 1111 1111")
    assert not luhn_valid("4111 1111 1111 1112")


def test_generation_is_repeatable():
    first = build_dataset(50, 7, "train")
    second = build_dataset(50, 7, "train")
    assert np.array_equal(first["bytes"], second["bytes"])
    assert np.array_equal(first["answers"], second["answers"])


def test_answer_spans_point_at_replies():
    dataset = build_dataset(500, 3, "train")
    for conversation, start, end, kind, _field, state in dataset["answers"]:
        text = conversation_at(dataset, conversation)
        reply = text[start:end]
        assert text[start - 3 : start] == "A: "
        assert reply.endswith("\n")
        if kind == KIND_REFUSAL:
            assert reply == REFUSAL + "\n"
            assert STATES[state] != "given"
        else:
            assert kind == KIND_RECALL
            assert STATES[state] == "given"
            value = reply.removesuffix(".\n").split(" is ", 1)[1]
            assert text.index(value) < start


def test_refusals_only_follow_missing_values_and_length_is_bounded():
    rng = Random(11)
    for _ in range(300):
        text, answers = build_conversation(rng, "train")
        for answer in answers:
            if answer.kind == KIND_REFUSAL:
                assert STATES[answer.state] in ("absent", "blank", "masked")
        assert len(text.encode("ascii")) <= MAX_BYTES


def test_every_state_and_kind_appears():
    answers = build_dataset(1000, 5, "train")["answers"]
    assert set(answers[:, 3]) == {KIND_RECALL, KIND_REFUSAL}
    assert set(answers[:, 5]) == set(range(len(STATES)))


def test_validation_names_are_held_out():
    train = split_names("train")
    val = split_names("val")
    for train_values, val_values in zip(train, val):
        assert val_values
        assert set(train_values).isdisjoint(val_values)


def test_padding_makes_longer_conversations_with_correct_answers():
    dataset = build_dataset(200, 3, "train", max_bytes=2048, pad_to=2048)
    lengths = np.diff(dataset["offsets"])
    assert lengths.max() <= 2048
    assert lengths.mean() > 600
    for conversation, start, end, kind, *_ in dataset["answers"]:
        reply = conversation_at(dataset, conversation)[start:end]
        assert reply.endswith("\n")
        assert (reply == REFUSAL + "\n") == (kind == KIND_REFUSAL)
