"""
judge.py
--------
LLM-as-judge behind a single interface.

    judge = make_judge("groq", model=config.JUDGE_MODEL, cache_dir=...)
    verdict = judge.score(question, gold_answer, context, answer)

Every provider returns the same JudgeVerdict, built from the same prompt
(eval/judge_prompt.md), so the rest of the evaluation never needs to know
which provider is behind it. To add a provider (e.g. Claude), subclass Judge,
implement _make_client(), and register it in JUDGES — then select it with
`python eval/run_eval.py --judge-provider <name>`.
"""

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

from eval.llm_cache import CachedChatModel

PROMPT_PATH = Path(__file__).resolve().parent / "judge_prompt.md"
SCORE_FIELDS = ("faithfulness", "answer_relevance", "context_relevance", "correctness")


def load_judge_prompt(path: Path = PROMPT_PATH) -> str:
    text = path.read_text()
    match = re.search(r"<!-- PROMPT START -->\n(.*)\n<!-- PROMPT END -->", text, re.S)
    if not match:
        raise ValueError(f"No PROMPT START/END markers in {path}")
    return match.group(1)


@dataclass
class JudgeVerdict:
    faithfulness: float | None
    answer_relevance: float | None
    context_relevance: float | None
    correctness: float | None
    abstained: bool | None
    reason: str
    parse_error: bool
    raw: str
    usage: Dict[str, int] | None
    cached: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _snap(value: Any) -> float | None:
    """Coerce a judge score onto the 0 / 0.5 / 1 scale."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return min((0.0, 0.5, 1.0), key=lambda s: abs(s - v))


def parse_verdict(raw: str) -> Dict[str, Any] | None:
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not all(f in data for f in SCORE_FIELDS):
        return None
    return data


class Judge:
    provider: str = ""
    api_key_env: str = ""

    def __init__(self, model: str, cache_dir: Path):
        if not os.getenv(self.api_key_env):
            raise EnvironmentError(
                f"{self.api_key_env} is not set — the {self.provider} judge needs it."
            )
        self.model = model
        self.prompt_template = load_judge_prompt()
        self.llm = CachedChatModel(
            self._make_client(),
            model_name=f"{self.provider}/{model}",
            params={"temperature": 0, "json": True},
            cache_path=cache_dir / "llm_cache.sqlite",
        )

    def _make_client(self):
        raise NotImplementedError

    def score(self, question: str, gold_answer: str, context: str, answer: str) -> JudgeVerdict:
        prompt = (
            self.prompt_template
            .replace("{{question}}", question)
            .replace("{{gold_answer}}", gold_answer)
            .replace("{{context}}", context or "(no context retrieved)")
            .replace("{{answer}}", answer)
        )
        response = self.llm.invoke(prompt)
        data = parse_verdict(response.content)
        if data is None:
            return JudgeVerdict(None, None, None, None, None, "", True,
                                response.content, response.usage_metadata, response.cached)
        abstained = data.get("abstained")
        return JudgeVerdict(
            faithfulness=_snap(data["faithfulness"]),
            answer_relevance=_snap(data["answer_relevance"]),
            context_relevance=_snap(data["context_relevance"]),
            correctness=_snap(data["correctness"]),
            abstained=abstained if isinstance(abstained, bool) else str(abstained).lower() == "true",
            reason=str(data.get("reason", "")),
            parse_error=False,
            raw=response.content,
            usage=response.usage_metadata,
            cached=response.cached,
        )


class GroqJudge(Judge):
    provider = "groq"
    api_key_env = "GROQ_API_KEY"

    def _make_client(self):
        from langchain_groq import ChatGroq

        return ChatGroq(
            api_key=os.environ[self.api_key_env],
            model_name=self.model,
            temperature=0,
            max_tokens=300,
            max_retries=6,
            model_kwargs={"response_format": {"type": "json_object"}},
        )


JUDGES = {
    "groq": GroqJudge,
}


def make_judge(provider: str, model: str, cache_dir: Path) -> Judge:
    if provider not in JUDGES:
        raise ValueError(f"Unknown judge provider '{provider}'. Available: {sorted(JUDGES)}")
    return JUDGES[provider](model=model, cache_dir=cache_dir)
