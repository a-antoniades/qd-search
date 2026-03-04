"""Score and tournament predictor interfaces and implementations."""

from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from evaluator.llm import LLMBackend
from evaluator.models import (
    ScorePrediction,
    Solution,
    TaskContext,
    TournamentPrediction,
)

# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------


class ScorePredictor(ABC):
    """Predicts the numeric metric score of a solution."""

    @abstractmethod
    def predict_score(
        self, solution: Solution, task: TaskContext
    ) -> ScorePrediction:
        """Predict the metric score for a single solution."""

    def predict_scores(
        self, solutions: list[Solution], task: TaskContext
    ) -> list[ScorePrediction]:
        """Predict scores for a batch of solutions (default: sequential)."""
        return [self.predict_score(s, task) for s in solutions]


class TournamentPredictor(ABC):
    """Predicts which of two solutions performs better."""

    @abstractmethod
    def predict_winner(
        self, a: Solution, b: Solution, task: TaskContext
    ) -> TournamentPrediction:
        """Predict which solution performs better."""

    def predict_tournament(
        self, pairs: list[tuple[Solution, Solution]], task: TaskContext
    ) -> list[TournamentPrediction]:
        """Predict winners for a batch of pairs (default: sequential)."""
        return [self.predict_winner(a, b, task) for a, b in pairs]


# ---------------------------------------------------------------------------
# Random baselines
# ---------------------------------------------------------------------------


class RandomScorePredictor(ScorePredictor):
    """Baseline: predicts a random score between median and gold thresholds."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def predict_score(
        self, solution: Solution, task: TaskContext
    ) -> ScorePrediction:
        lo = task.median_threshold if task.median_threshold is not None else 0.0
        hi = task.gold_threshold if task.gold_threshold is not None else 1.0
        if lo > hi:
            lo, hi = hi, lo
        score = self._rng.uniform(lo, hi)
        return ScorePrediction(
            solution_id=solution.id,
            predicted_score=score,
            confidence=0.0,
            reasoning="random baseline",
        )


class RandomTournamentPredictor(TournamentPredictor):
    """Baseline: coin flip (expected ~50% accuracy)."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def predict_winner(
        self, a: Solution, b: Solution, task: TaskContext
    ) -> TournamentPrediction:
        winner = self._rng.choice([a, b])
        return TournamentPrediction(
            solution_a_id=a.id,
            solution_b_id=b.id,
            winner_id=winner.id,
            confidence=0.5,
            reasoning="random baseline (coin flip)",
        )


# ---------------------------------------------------------------------------
# LLM-based predictors
# ---------------------------------------------------------------------------

_SCORE_SYSTEM_PROMPT = (
    "You are a critical assessor. Your job is to predict how a solution "
    "ACTUALLY performs on unseen test data — not how well it could perform "
    "in theory.\n\n"
    "Before predicting a score, mentally simulate the solution's execution "
    "step by step:\n"
    "1. Will the code run without errors? Check for missing imports, shape "
    "mismatches, OOM risks, timeouts.\n"
    "2. Does the data pipeline handle edge cases? Check for leakage between "
    "train/test, missing value handling, encoding issues.\n"
    "3. Will the model generalize? Look for overfitting signals: no "
    "regularization, too many parameters, no cross-validation.\n"
    "4. What is the realistic gap between validation and test performance?\n\n"
    "Most solutions have subtle flaws that degrade test performance "
    "significantly relative to what the plan promises. Your predictions "
    "should span the full range from catastrophic failure to excellent — "
    "do not cluster near a single value.\n\n"
    "Respond with a JSON object with these keys in this order:\n"
    '"reasoning" (string — simulate execution and identify specific risks '
    "BEFORE predicting the score),\n"
    '"predicted_score" (float),\n'
    '"confidence" (float 0-1).'
)

_TOURNAMENT_SYSTEM_PROMPT = (
    "You are a critical assessor comparing two solutions. Predict which one "
    "ACTUALLY achieves a better test score — not which looks more "
    "impressive on paper.\n\n"
    "For each solution, mentally simulate its execution:\n"
    "1. Execution risk — will it crash, timeout, or produce invalid output?\n"
    "2. Generalization — does it guard against overfitting (regularization, "
    "validation strategy, data augmentation)?\n"
    "3. Approach fit — is the method appropriate for this specific task and "
    "data characteristics?\n"
    "4. Robustness — does it handle edge cases, missing data, and distribution "
    "shifts?\n\n"
    "A simpler, robust solution often beats a complex, fragile one on "
    "unseen test data.\n\n"
    "Respond with a JSON object with these keys in this order:\n"
    '"reasoning" (string — simulate each solution, then compare),\n'
    '"winner" (string, either "A" or "B"),\n'
    '"confidence" (float 0-1).'
)


def _build_task_section(task: TaskContext) -> str:
    """Build the task context section for LLM prompts."""
    parts = [f"## Task: {task.name}"]
    if task.description:
        parts.append(f"\n{task.description}\n")
    direction = "lower is better" if task.is_lower_better else "higher is better"
    if task.metric_name:
        parts.append(f"**Metric:** {task.metric_name} ({direction})")
    else:
        parts.append(f"**Metric direction:** {direction}")

    # Score range anchors — give the LLM the full performance spectrum
    if task.gold_threshold is not None and task.median_threshold is not None:
        gold, median = task.gold_threshold, task.median_threshold
        spread = abs(gold - median)
        if task.is_lower_better:
            poor = median + spread * 2
            fail = median + spread * 4
            parts.append(
                f"**Score scale:** Gold ≤ {gold} | Median ~ {median} "
                f"| Poor ~ {poor:.5g} | Failed ~ {fail:.5g} ({direction})"
            )
        else:
            poor = median - spread * 2
            fail = max(0.0, median - spread * 4)
            parts.append(
                f"**Score scale:** Gold ≥ {gold} | Median ~ {median} "
                f"| Poor ~ {poor:.5g} | Failed ~ {fail:.5g} ({direction})"
            )
    else:
        if task.gold_threshold is not None:
            parts.append(f"**Gold medal threshold:** {task.gold_threshold}")
        if task.median_threshold is not None:
            parts.append(f"**Median threshold:** {task.median_threshold}")

    return "\n".join(parts)


def _build_solution_section(
    solution: Solution, label: str = "", include_code: bool = True
) -> str:
    """Build the solution section for LLM prompts."""
    header = f"## Solution {label}" if label else "## Solution"
    parts = [header]

    # Execution metadata (ground truth signals)
    meta_items = []
    if solution.operators_used:
        meta_items.append(f"**Operators applied:** {', '.join(solution.operators_used)}")
    if solution.is_buggy:
        meta_items.append("**Execution status:** FAILED")
    elif solution.exit_code != 0:
        meta_items.append(f"**Execution status:** Non-zero exit code ({solution.exit_code})")
    else:
        meta_items.append("**Execution status:** Completed successfully")
    if meta_items:
        parts.append("\n".join(meta_items))

    if solution.plan:
        parts.append(f"\n### Plan\n{solution.plan}")
    if include_code and solution.code:
        parts.append(f"\n### Code\n```python\n{solution.code}\n```")
    return "\n".join(parts)


def _parse_json_response(text: str) -> dict:
    """Extract a JSON object from LLM response text.

    Handles common LLM quirks: markdown fences, unescaped control characters,
    and structurally malformed JSON (falls back to regex extraction).
    """
    import re

    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Clean control characters and retry
    cleaned = re.sub(r'[\x00-\x1f]', lambda m: ' ' if m.group() in ('\n', '\r') else '', text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Last resort: extract key fields via regex
    result = {}

    # Extract numeric fields (quoted or unquoted keys)
    for key in ("predicted_score", "confidence"):
        m = re.search(rf'(?:"{key}"|{key})\s*[:=]\s*([0-9.eE+-]+)', text)
        if m:
            result[key] = float(m.group(1))

    # Extract winner field (quoted or unquoted)
    m = re.search(r'(?:"winner"|winner)\s*[:=]\s*"?([ABab])"?', text)
    if m:
        result["winner"] = m.group(1).upper()

    # Extract reasoning (JSON quoted or bare "Reasoning:" block)
    m = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    if m:
        result["reasoning"] = m.group(1)
    else:
        # Bare "Reasoning:" header followed by text (Gemini non-JSON format)
        m = re.search(r'[Rr]easoning\s*:\s*\n?(.*?)(?:\n\s*(?:predicted_score|winner|confidence)\s*[:=]|\Z)', text, re.DOTALL)
        if m:
            result["reasoning"] = m.group(1).strip()
        else:
            result["reasoning"] = ""

    if "predicted_score" in result or "winner" in result:
        return result

    raise ValueError(f"Could not parse LLM response: {text[:200]}...")


class LLMScorePredictor(ScorePredictor):
    """Predicts metric scores using an LLM.

    Args:
        backend: The LLM backend to use for queries.
        include_code: Whether to include solution code in prompts.
    """

    def __init__(
        self, backend: LLMBackend, include_code: bool = True
    ) -> None:
        self.backend = backend
        self.include_code = include_code

    def predict_score(
        self, solution: Solution, task: TaskContext, _retries: int = 10
    ) -> ScorePrediction:
        import logging as _logging
        import re as _re
        import time as _time

        prompt = (
            f"{_build_task_section(task)}\n\n"
            f"{_build_solution_section(solution, include_code=self.include_code)}\n\n"
            "Predict the test metric score this solution will achieve."
        )
        last_error: Exception | None = None
        last_raw: str = ""
        for attempt in range(_retries):
            try:
                raw = self.backend.query(prompt, system_prompt=_SCORE_SYSTEM_PROMPT)
                last_raw = raw
                parsed = _parse_json_response(raw)
                # Accept common key variants from different models
                score_val = parsed.get("predicted_score") or parsed.get("score") or parsed.get("prediction")
                if score_val is None:
                    # Fallback: extract a plausible score from reasoning text
                    reasoning = parsed.get("reasoning", "")
                    m = _re.search(
                        r'(?:predict(?:ed)?[\s_]?score|estimated?\s+score|expect(?:ed)?\s+score|score\s*(?:of|:|\=|around|approximately|~))\s*[:=~]?\s*([0-9]+\.?[0-9]*)',
                        reasoning, _re.IGNORECASE,
                    )
                    if m:
                        score_val = m.group(1)
                        _logging.getLogger(__name__).info(
                            "Extracted score %.4f from reasoning for %s",
                            float(score_val), solution.id,
                        )
                    else:
                        raise KeyError(f"No score key found in {list(parsed.keys())}")
                return ScorePrediction(
                    solution_id=solution.id,
                    predicted_score=float(score_val),
                    confidence=float(parsed.get("confidence", 0.0)),
                    reasoning=str(parsed.get("reasoning", "")),
                    prompt=prompt,
                    raw_response=raw,
                )
            except (ValueError, KeyError) as e:
                last_error = e
                _logging.getLogger(__name__).warning(
                    "Score parse failed (attempt %d/%d) for %s: %s",
                    attempt + 1, _retries, solution.id, e,
                )
            except Exception as e:
                last_error = e
                wait = 2 ** attempt * 10 + _time.time() % 5  # jitter
                _logging.getLogger(__name__).warning(
                    "Score API error (attempt %d/%d) for %s: %s. Retrying in %.0fs",
                    attempt + 1, _retries, solution.id, type(e).__name__, wait,
                )
                _time.sleep(wait)
        raise last_error  # type: ignore[misc]


class LLMTournamentPredictor(TournamentPredictor):
    """Predicts pairwise winners using an LLM.

    Randomizes A/B ordering to avoid position bias.

    Args:
        backend: The LLM backend to use for queries.
        include_code: Whether to include solution code in prompts.
        seed: Random seed for A/B ordering.
    """

    def __init__(
        self,
        backend: LLMBackend,
        include_code: bool = True,
        seed: int | None = None,
    ) -> None:
        self.backend = backend
        self.include_code = include_code
        self._rng = random.Random(seed)

    def predict_winner(
        self, a: Solution, b: Solution, task: TaskContext, _retries: int = 5
    ) -> TournamentPrediction:
        import logging as _logging
        import time as _time

        # Randomize order to avoid position bias
        if self._rng.random() < 0.5:
            first, second = a, b
            label_first, label_second = "A", "B"
        else:
            first, second = b, a
            label_first, label_second = "B", "A"

        prompt = (
            f"{_build_task_section(task)}\n\n"
            f"{_build_solution_section(first, label='A', include_code=self.include_code)}\n\n"
            f"{_build_solution_section(second, label='B', include_code=self.include_code)}\n\n"
            "Which solution achieves a better test metric score?"
        )
        last_error: Exception | None = None
        for attempt in range(_retries):
            try:
                raw = self.backend.query(prompt, system_prompt=_TOURNAMENT_SYSTEM_PROMPT)
                parsed = _parse_json_response(raw)
                winner_label = parsed["winner"].strip().upper()
                # Map back from randomized labels to actual solutions
                if winner_label == label_first:
                    winner_id = first.id
                else:
                    winner_id = second.id

                return TournamentPrediction(
                    solution_a_id=a.id,
                    solution_b_id=b.id,
                    winner_id=winner_id,
                    confidence=float(parsed.get("confidence", 0.0)),
                    reasoning=str(parsed.get("reasoning", "")),
                    prompt=prompt,
                    raw_response=raw,
                )
            except (ValueError, KeyError) as e:
                last_error = e
                _logging.getLogger(__name__).warning(
                    "Tournament parse failed (attempt %d/%d) for %s vs %s: %s",
                    attempt + 1, _retries, a.id, b.id, e,
                )
            except Exception as e:
                last_error = e
                wait = 2 ** attempt * 10 + _time.time() % 5  # jitter
                _logging.getLogger(__name__).warning(
                    "Tournament API error (attempt %d/%d) for %s vs %s: %s. Retrying in %.0fs",
                    attempt + 1, _retries, a.id, b.id, type(e).__name__, wait,
                )
                _time.sleep(wait)
        # All retries exhausted — fall back to random choice instead of crashing
        _logging.getLogger(__name__).error(
            "Tournament FALLBACK (random) for %s vs %s after %d retries: %s",
            a.id, b.id, _retries, last_error,
        )
        winner = self._rng.choice([a, b])
        return TournamentPrediction(
            solution_a_id=a.id, solution_b_id=b.id, winner_id=winner.id,
            confidence=0.0, reasoning=f"FALLBACK (random): {last_error}",
        )


# ---------------------------------------------------------------------------
# Ensemble wrappers (majority voting / score averaging)
# ---------------------------------------------------------------------------


class EnsembleScorePredictor(ScorePredictor):
    """Averages N independent score predictions to reduce variance.

    When num_votes=1, delegates directly to the inner predictor (zero overhead).
    """

    def __init__(
        self,
        inner: ScorePredictor,
        num_votes: int = 3,
        max_workers: int = 8,
    ) -> None:
        self.inner = inner
        self.num_votes = num_votes
        self.max_workers = max_workers

    def predict_score(
        self, solution: Solution, task: TaskContext
    ) -> ScorePrediction:
        if self.num_votes <= 1:
            return self.inner.predict_score(solution, task)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [
                pool.submit(self.inner.predict_score, solution, task)
                for _ in range(self.num_votes)
            ]
            results = [f.result() for f in futures]

        scores = tuple(r.predicted_score for r in results)
        confidences = tuple(r.confidence for r in results)
        mean_score = sum(scores) / len(scores)
        mean_conf = sum(confidences) / len(confidences)

        # Pick reasoning from the prediction closest to the mean
        closest = min(results, key=lambda r: abs(r.predicted_score - mean_score))

        return ScorePrediction(
            solution_id=solution.id,
            predicted_score=mean_score,
            confidence=mean_conf,
            reasoning=closest.reasoning,
            prompt=closest.prompt,
            raw_response=closest.raw_response,
            individual_scores=scores,
            individual_confidences=confidences,
            num_votes=self.num_votes,
        )

    def predict_scores(
        self, solutions: list[Solution], task: TaskContext
    ) -> list[ScorePrediction]:
        if self.num_votes <= 1:
            return self.inner.predict_scores(solutions, task)

        # Flatten all (solution, vote) into one pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_map: dict[int, list] = {i: [] for i in range(len(solutions))}
            for i, sol in enumerate(solutions):
                for _ in range(self.num_votes):
                    fut = pool.submit(self.inner.predict_score, sol, task)
                    future_map[i].append(fut)

            out = []
            for i, sol in enumerate(solutions):
                results = [f.result() for f in future_map[i]]
                scores = tuple(r.predicted_score for r in results)
                confidences = tuple(r.confidence for r in results)
                mean_score = sum(scores) / len(scores)
                mean_conf = sum(confidences) / len(confidences)
                closest = min(results, key=lambda r: abs(r.predicted_score - mean_score))
                out.append(ScorePrediction(
                    solution_id=sol.id,
                    predicted_score=mean_score,
                    confidence=mean_conf,
                    reasoning=closest.reasoning,
                    prompt=closest.prompt,
                    raw_response=closest.raw_response,
                    individual_scores=scores,
                    individual_confidences=confidences,
                    num_votes=self.num_votes,
                ))
            return out


class EnsembleTournamentPredictor(TournamentPredictor):
    """Majority-votes N independent pairwise predictions to reduce variance.

    Creates N inner LLMTournamentPredictor instances with different seeds
    for independent A/B randomization. When num_votes=1, delegates directly.
    """

    def __init__(
        self,
        backend: LLMBackend,
        num_votes: int = 3,
        max_workers: int = 8,
        include_code: bool = True,
        base_seed: int = 42,
    ) -> None:
        self.num_votes = num_votes
        self.max_workers = max_workers
        self._inners = [
            LLMTournamentPredictor(
                backend=backend, include_code=include_code, seed=base_seed + i
            )
            for i in range(num_votes)
        ]

    def predict_winner(
        self, a: Solution, b: Solution, task: TaskContext
    ) -> TournamentPrediction:
        if self.num_votes <= 1:
            return self._inners[0].predict_winner(a, b, task)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [
                pool.submit(inner.predict_winner, a, b, task)
                for inner in self._inners
            ]
            results = [f.result() for f in futures]

        winners = tuple(r.winner_id for r in results)
        counts = Counter(winners)
        winner_id = counts.most_common(1)[0][0]
        vote_counts = tuple(counts.most_common())
        confidence = counts[winner_id] / self.num_votes

        # Pick reasoning from a vote that chose the majority winner
        rep = next(r for r in results if r.winner_id == winner_id)

        return TournamentPrediction(
            solution_a_id=a.id,
            solution_b_id=b.id,
            winner_id=winner_id,
            confidence=confidence,
            reasoning=rep.reasoning,
            prompt=rep.prompt,
            raw_response=rep.raw_response,
            individual_winners=winners,
            vote_counts=vote_counts,
            num_votes=self.num_votes,
        )

    def predict_tournament(
        self, pairs: list[tuple[Solution, Solution]], task: TaskContext
    ) -> list[TournamentPrediction]:
        if self.num_votes <= 1:
            return self._inners[0].predict_tournament(pairs, task)

        # Flat parallelism: submit all (pair_idx, vote_idx) to one pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_map: dict[int, list] = {i: [] for i in range(len(pairs))}
            for i, (a, b) in enumerate(pairs):
                for inner in self._inners:
                    fut = pool.submit(inner.predict_winner, a, b, task)
                    future_map[i].append(fut)

            out = []
            for i, (a, b) in enumerate(pairs):
                results = [f.result() for f in future_map[i]]
                winners = tuple(r.winner_id for r in results)
                counts = Counter(winners)
                winner_id = counts.most_common(1)[0][0]
                vote_counts = tuple(counts.most_common())
                confidence = counts[winner_id] / self.num_votes
                rep = next(r for r in results if r.winner_id == winner_id)
                out.append(TournamentPrediction(
                    solution_a_id=a.id,
                    solution_b_id=b.id,
                    winner_id=winner_id,
                    confidence=confidence,
                    reasoning=rep.reasoning,
                    prompt=rep.prompt,
                    raw_response=rep.raw_response,
                    individual_winners=winners,
                    vote_counts=vote_counts,
                    num_votes=self.num_votes,
                ))
            return out
