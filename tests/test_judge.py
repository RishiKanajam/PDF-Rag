from types import SimpleNamespace

import eval.judge as judge_mod
from eval.judge import Judge, load_judge_prompt, make_judge, parse_verdict


class _FakeModel:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(
            content=self.reply,
            usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
            response_metadata={"model_name": "fake", "system_fingerprint": "fp_test"},
        )


def _register_fake(monkeypatch, reply):
    model = _FakeModel(reply)

    class FakeJudge(Judge):
        provider = "fake"
        api_key_env = "FAKE_KEY"

        def _make_client(self):
            return model

    monkeypatch.setenv("FAKE_KEY", "x")
    monkeypatch.setitem(judge_mod.JUDGES, "fake", FakeJudge)
    return model


def test_prompt_file_has_all_placeholders():
    prompt = load_judge_prompt()
    for placeholder in ("{{question}}", "{{gold_answer}}", "{{context}}", "{{answer}}"):
        assert placeholder in prompt


def test_parse_verdict_tolerates_wrapping_text():
    raw = 'Here you go: {"faithfulness": 1, "answer_relevance": 0.5, "context_relevance": 1, "correctness": 0, "abstained": false, "reason": "ok"}'
    assert parse_verdict(raw)["answer_relevance"] == 0.5
    assert parse_verdict("not json") is None
    assert parse_verdict('{"faithfulness": 1}') is None  # missing fields


def test_new_provider_plugs_in_and_scores_are_snapped(monkeypatch, tmp_path):
    reply = '{"faithfulness": 0.9, "answer_relevance": 1, "context_relevance": 0.4, "correctness": "0", "abstained": "false", "reason": "fine"}'
    model = _register_fake(monkeypatch, reply)

    judge = make_judge("fake", model="fake-1", cache_dir=tmp_path)
    verdict = judge.score("Q {braces}?", "gold", "context", "answer")

    assert (verdict.faithfulness, verdict.context_relevance, verdict.correctness) == (1.0, 0.5, 0.0)
    assert verdict.abstained is False and not verdict.parse_error
    assert "Q {braces}?" in model.prompts[0] and "{{" not in model.prompts[0]

    # Second identical call is served from the cache.
    again = judge.score("Q {braces}?", "gold", "context", "answer")
    assert again.cached and len(model.prompts) == 1


def test_unparseable_reply_is_flagged(monkeypatch, tmp_path):
    _register_fake(monkeypatch, "I think it's pretty good")
    verdict = make_judge("fake", model="fake-1", cache_dir=tmp_path).score("q", "g", "c", "a")
    assert verdict.parse_error and verdict.faithfulness is None


def test_missing_key_fails_loudly(monkeypatch, tmp_path):
    _register_fake(monkeypatch, "{}")
    monkeypatch.delenv("FAKE_KEY")
    try:
        make_judge("fake", model="fake-1", cache_dir=tmp_path)
    except EnvironmentError as e:
        assert "FAKE_KEY" in str(e)
    else:
        raise AssertionError("expected EnvironmentError")


def test_offline_cache_never_calls_the_model(tmp_path):
    import pytest
    from eval.llm_cache import CacheMiss, CachedChatModel

    online = CachedChatModel(_FakeModel("hello"), "fake", {}, tmp_path / "c.sqlite")
    online.invoke("cached prompt")

    offline = CachedChatModel(None, "fake", {}, tmp_path / "c.sqlite", offline=True)
    assert offline.invoke("cached prompt").content == "hello"
    with pytest.raises(CacheMiss):
        offline.invoke("never seen")
