# Judge prompt

`eval/judge.py` reads this file at runtime: the text between the two PROMPT markers below is the exact prompt sent to the judge. It runs at temperature 0, in JSON mode, with one call per (question, answer) pair. `{{question}}`, `{{gold_answer}}`, `{{context}}` and `{{answer}}` are filled in by plain string replacement.

Scores use a three-point scale (0, 0.5, 1) rather than a continuous 0–1 range. A coarse scale is easier for an LLM to apply consistently, and easier for a human to check in `human_check.csv`.

<!-- PROMPT START -->
You are a strict evaluator of a retrieval-augmented question-answering system.
You are given a QUESTION, a GOLD ANSWER written by a human, the CONTEXT passages
the system retrieved, and the system's ANSWER. Score the ANSWER on the criteria
below. Use only the values 0, 0.5 or 1 for each score.

faithfulness — Is every factual claim in the ANSWER supported by the CONTEXT?
  1   = every claim is supported by the CONTEXT (an answer that only says it
        does not know, or makes no factual claims, scores 1)
  0.5 = mostly supported, but at least one minor claim is not in the CONTEXT
  0   = a significant claim is unsupported by or contradicts the CONTEXT
  Judge this against the CONTEXT only, not against the GOLD ANSWER or your own knowledge.

answer_relevance — Does the ANSWER address the QUESTION that was asked?
  1   = directly addresses the question
  0.5 = partially addresses it, or buries the answer in irrelevant material
  0   = does not address the question (an answer that declines to answer scores 0,
        unless the GOLD ANSWER says the question is not answerable, in which case it scores 1)

context_relevance — Does the CONTEXT contain the information needed to answer the QUESTION?
  1   = the CONTEXT contains the information in the GOLD ANSWER
  0.5 = the CONTEXT is on the right topic but only partly contains it
  0   = the CONTEXT does not contain it
  If the GOLD ANSWER says the question is not answerable, score 1 if the CONTEXT
  indeed does not answer the question, else 0.

correctness — Does the ANSWER agree with the GOLD ANSWER?
  1   = contains the key facts of the GOLD ANSWER with no contradictions
  0.5 = partly correct or incomplete
  0   = wrong, contradicts the GOLD ANSWER, or declines to answer an answerable question
  If the GOLD ANSWER says the question is not answerable: 1 if the ANSWER says the
  information is not available, 0 if it gives a substantive answer.

abstained — true if the ANSWER says it does not have enough information / cannot
  answer from the documents, false otherwise.

QUESTION:
{{question}}

GOLD ANSWER:
{{gold_answer}}

CONTEXT:
{{context}}

ANSWER:
{{answer}}

Respond with a single JSON object and nothing else, in exactly this form:
{"faithfulness": <0|0.5|1>, "answer_relevance": <0|0.5|1>, "context_relevance": <0|0.5|1>, "correctness": <0|0.5|1>, "abstained": <true|false>, "reason": "<one sentence>"}
<!-- PROMPT END -->
