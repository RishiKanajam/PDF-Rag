import csv

import eval.agreement as agreement
import eval.run_eval as run_eval


def _rows(n):
    return [
        {
            "id": f"q{i:02d}", "mode": "hybrid_rerank", "k": run_eval.GENERATION_K,
            "type": "single_fact", "question": f"Question {i}?", "gold_answer": "gold",
            "context": "ctx, with \"quotes\"\nand newlines", "answer": f"answer {i}",
            "judge_faithfulness": 1.0, "judge_answer_relevance": 0.5, "judge_correctness": 0.0,
            "judge_parse_error": False,
        }
        for i in range(n)
    ]


def test_export_writes_15_items_and_never_overwrites(tmp_path, monkeypatch):
    path = tmp_path / "human_check.csv"
    monkeypatch.setattr(run_eval, "HUMAN_CHECK", path)
    monkeypatch.setattr(run_eval, "ROOT", tmp_path)

    rows = _rows(40) + [{**_rows(1)[0], "k": 5, "id": "ablation"}]
    assert run_eval.export_human_check(rows) is True

    exported = list(csv.DictReader(open(path)))
    assert len(exported) == 15
    assert all(r["id"] != "ablation" for r in exported)          # shipped k only
    assert all(r["human_correctness"] == "" for r in exported)   # blank for the human
    assert exported[0]["context"] == "ctx, with \"quotes\"\nand newlines"

    # Same seed → same sample; existing file (with human scores) is left alone.
    path.write_text("filled in by a human")
    assert run_eval.export_human_check(rows) is False
    assert path.read_text() == "filled in by a human"


def test_export_is_readable_by_agreement(tmp_path, monkeypatch, capsys):
    path = tmp_path / "human_check.csv"
    monkeypatch.setattr(run_eval, "HUMAN_CHECK", path)
    monkeypatch.setattr(run_eval, "ROOT", tmp_path)
    run_eval.export_human_check(_rows(20))

    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r.update(human_faithfulness="1", human_answer_relevance="0.5", human_correctness="0")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    assert agreement.main(path) == 0
    out = capsys.readouterr().out
    assert "faithfulness" in out and "100%" in out
