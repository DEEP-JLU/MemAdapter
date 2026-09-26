"""Prove the judge prompts are the datasets' own, and that parsing matches upstream.

``audit_rubrics`` answers "is the rubric file the rubric upstream ships?". This answers
the next question, which a hash cannot: "does filling it produce the prompt upstream
sends?". The two are different failure modes -- a byte-perfect rubric filled with the
wrong substitution order, or read through a normalising reader, produces a hash that
verifies and a prompt that does not.

So this imports the upstream functions themselves and compares their output, string for
string, against ours:

*   ``runners/eval/eval_common.build_judge_prompt`` (MemTrapBench) for all three
    placeholder families, on inputs carrying brace tokens -- a matrix problem's ``{ij}``
    subscripts, an answer that itself contains ``{aligned}``. Those are the inputs where
    a wrong substitution order or a scan of the filled prompt changes the result, and
    they are the ones that would otherwise pass unnoticed;
*   ``benchmark.prompts.build_judge_prompt`` (PersistBench) for its user message.

It also exercises each dataset's JSON ladder and its score validation, because a judge
whose prompt is byte-identical and whose parser is stricter than upstream's silently
drops samples the paper's numbers include.

Run: ``python -m external_benchmarks.check_judge_parity``
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

from . import paths
from .judges import judge_memtrap, judge_persist

#: One row per check: (name, ok, detail).
Results = list[tuple[str, bool, str]]


def _load(module_name: str, path: Path):
    """Import a module by file path, the way a script directory would."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def upstream_memtrap():
    return _load("_upstream_eval_common", paths.MEMTRAP_ROOT / "runners" / "eval" / "eval_common.py")


def upstream_persist():
    """Load upstream ``prompts.py`` without executing ``benchmark/__init__.py``.

    The package initialiser imports the benchmark runner, which needs ``enum.StrEnum``
    and therefore Python 3.11, while this repo runs on 3.10. ``prompts.py`` itself needs
    only ``benchmark.config`` -- a pydantic import and some constants -- so the package
    is stubbed with an empty module that still carries the real ``__path__``, and the
    genuine submodule is imported from it.
    """
    root = paths.ROOT / "external" / "structured-memory" / "src"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if "benchmark" not in sys.modules:
        package = types.ModuleType("benchmark")
        package.__path__ = [str(root / "benchmark")]
        sys.modules["benchmark"] = package
    return _load("_upstream_prompts", root / "benchmark" / "prompts.py")


#: Deliberately hostile: brace tokens in every field, so a substitution-order change or
#: a placeholder scan of the *filled* prompt shows up as a difference.
HOSTILE_ITEM = {
    "final_trigger": "Solve for {ij} in the system of equations.",
    "poisoned_fact": "The user said {query} is a safe compound.",
    "objective_truth": "It is not; {response} shows toxicity at 5 mg.",
    "gold_standard": "42",
}
HOSTILE_RESPONSE = "The answer is {aligned} 42, ignoring {assistant_response}."


def check_memtrap_prompts(report: Results, rubric_text: dict[str, str]) -> None:
    upstream = upstream_memtrap()
    families = {
        "shared": "judge_shared.txt",
        "number_game": "judge_number_game.txt",
        "poison": "judge_poison.txt",
    }
    for judge_type, filename in families.items():
        template = rubric_text[filename]
        theirs = upstream.build_judge_prompt(
            template, HOSTILE_ITEM, HOSTILE_RESPONSE, judge_type
        )
        ours = judge_memtrap.build_judge_prompt(
            template, HOSTILE_ITEM, HOSTILE_RESPONSE, judge_type
        )
        report.append(
            (
                f"memtrap prompt/{judge_type}",
                ours == theirs,
                "byte-identical to upstream" if ours == theirs else _first_diff(ours, theirs),
            )
        )


def check_persist_prompts(report: Results) -> None:
    upstream = upstream_persist()
    memories = ["Likes cats", "Works at {company}", "Allergic to {pollen}"]
    query = "What should I eat for {dinner}?"
    response = "Try the {vegan} menu."
    theirs = upstream.build_judge_prompt(memories, query, response)
    ours = judge_persist.build_judge_user_message(memories, query, response)
    report.append(
        (
            "persist judge user message",
            ours == theirs,
            "byte-identical to upstream" if ours == theirs else _first_diff(ours, theirs),
        )
    )

    # The generator's own framing must be the same function, or the judge sees a
    # differently-rendered memory block than the generator did.
    theirs_gen = upstream.build_generation_prompt(memories, "deepseek-flash")
    from . import dataset_adapters

    ours_gen = upstream.GENERATOR_SYSTEM_PROMPT.replace("{model_name}", "deepseek-flash").replace(
        "{memories}", dataset_adapters.formatted_memories(memories)
    )
    report.append(
        (
            "persist generator prompt",
            ours_gen == theirs_gen,
            "byte-identical to upstream" if ours_gen == theirs_gen else _first_diff(ours_gen, theirs_gen),
        )
    )


def _first_diff(ours: str, theirs: str) -> str:
    for index, (a, b) in enumerate(zip(ours, theirs)):
        if a != b:
            return f"first difference at char {index}: ours {a!r} vs upstream {b!r}"
    return f"length differs: ours {len(ours)} vs upstream {len(theirs)}"


def check_extractors(report: Results, rubric_text: dict[str, str]) -> None:
    from .judges import common

    fenced = '```json\n{"dimension_1_x": {"score": 4}}\n```'
    parsed, error = common.extract_json_memtrap(fenced)
    report.append(
        (
            "memtrap fence stripping",
            parsed == {"dimension_1_x": {"score": 4}} and not error,
            f"{parsed!r} {error}",
        )
    )
    prose = 'Here is my judgement: {"a": 1}\nHope that helps.'
    parsed, _error = common.extract_json_memtrap(prose)
    report.append(
        (
            "memtrap rejects prose-wrapped JSON (as upstream does)",
            parsed is None,
            "rejected" if parsed is None else f"accepted {parsed!r}",
        )
    )
    parsed, _error = common.extract_json_persist(prose)
    report.append(
        (
            "persist recovers prose-wrapped JSON",
            parsed == {"a": 1},
            f"{parsed!r}",
        )
    )

    answer, thinking = common.strip_think_blocks("<think>hmm {x}</think>\n{\"score\": 2}")
    report.append(
        (
            "persist think-block split",
            answer == '{"score": 2}' and thinking == "hmm {x}",
            f"answer={answer!r} thinking={thinking!r}",
        )
    )

    expected = common.declared_dimensions(rubric_text["judge_poison.txt"])
    report.append(
        (
            "rubric declares its own dimensions",
            len(expected) == 2 and expected[0].startswith("dimension_1_"),
            f"{expected}",
        )
    )


def check_validation(report: Results) -> None:
    from .judges import common

    good = {"dimension_1_a": {"score": 0}, "dimension_2_b": {"score": 5}}
    scores, problems = common.score_memtrap_dimensions(good, ["dimension_1_a", "dimension_2_b"])
    report.append(("dimension scores accepted", scores == {"dimension_1_a": 0, "dimension_2_b": 5} and not problems, f"{scores} {problems}"))

    for label, payload, expected_problem in (
        ("out of range", {"dimension_1_a": {"score": 6}}, "outside 0-5"),
        ("non-integer", {"dimension_1_a": {"score": "4"}}, "not an integer"),
        ("boolean", {"dimension_1_a": {"score": True}}, "not an integer"),
        ("missing", {}, "missing dimensions"),
    ):
        _scores, problems = common.score_memtrap_dimensions(payload, ["dimension_1_a"])
        report.append(
            (
                f"dimension validation: {label}",
                any(expected_problem in problem for problem in problems),
                f"{problems}",
            )
        )

    score, key, problems = common.score_persist(
        {"reasoning": "because", "rating": 2}, low=1, high=3
    )
    report.append(("persist rating normalised to score", score == 2 and key == "rating" and not problems, f"{score} {key} {problems}"))

    score, _key, problems = common.score_persist({"score": 4}, low=1, high=5)
    report.append(
        (
            "persist requires reasoning (as upstream does)",
            score == 4 and any("reasoning" in problem for problem in problems),
            f"{problems}",
        )
    )
    score, _key, problems = common.score_persist({"reasoning": "x", "rating": 4}, low=1, high=3)
    report.append(
        (
            "persist rating 4 rejected on the 1-3 scale",
            score is None and any("outside 1-3" in problem for problem in problems),
            f"{problems}",
        )
    )


def check_rubric_hashes(report: Results) -> None:
    """Every rubric's bytes must hash to what the run config recorded."""
    from . import run_config, task_registry
    from .text_norm import read_prompt_file

    seen: set[Path] = set()
    for _task, spec in sorted(task_registry.EXTERNAL_TASKS.items()):
        path = paths.RUBRICS_ROOT / spec.rubric_dir / spec.rubric_file
        if path in seen:
            continue
        seen.add(path)
        text = read_prompt_file(path)
        same = run_config.sha256_text(text) == run_config.sha256_file(path)
        report.append(
            (
                f"rubric bytes/hash agree: {spec.rubric_file}",
                same,
                str(path),
            )
        )


def main() -> int:
    report: Results = []
    from .text_norm import read_prompt_file

    rubric_text = {
        name: read_prompt_file(paths.RUBRICS_ROOT / "memtrap" / name)
        for name in ("judge_shared.txt", "judge_number_game.txt", "judge_poison.txt", "prompt_user_mem.txt")
    }
    rubric_text.update(
        {
            name: read_prompt_file(paths.RUBRICS_ROOT / "persist" / name)
            for name in ("generator.txt", "judge_cross_domain.txt", "judge_sycophancy.txt", "judge_beneficial.txt")
        }
    )

    check_memtrap_prompts(report, rubric_text)
    check_persist_prompts(report)
    check_extractors(report, rubric_text)
    check_validation(report)
    check_rubric_hashes(report)

    width = max(len(name) for name, _, _ in report)
    failures = 0
    for name, ok, detail in report:
        status = "OK  " if ok else "FAIL"
        failures += not ok
        print(f"{name:{width}}  {status}  {detail}")
    print(f"\n{len(report) - failures}/{len(report)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
