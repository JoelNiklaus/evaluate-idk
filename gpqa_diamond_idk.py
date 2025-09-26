from __future__ import annotations

import random
import numpy as np
import os
import logging
import re

from dotenv import load_dotenv
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.requests import SamplingMethod
from lighteval.metrics.utils.extractive_match_utils import (
    IndicesExtractionConfig,
    get_extraction_regexes,
    extract_target_from_pred,
)
from lighteval.utils.language import Language


logging.basicConfig(level=os.getenv("EVALUATE_IDK_LOG_LEVEL", "INFO"))

load_dotenv()

# -----------------------------
# Prompt function
# -----------------------------


def gpqa_diamond_idk_prompt(line: dict, task_name: str) -> Doc:
    """Convert a GPQA Diamond row to a Doc with 4 shuffled choices + E: I don't know.

    Expected input fields:
    - Question
    - Correct Answer
    - Incorrect Answer 1
    - Incorrect Answer 2
    - Incorrect Answer 3

    Behavior:
    - Shuffle the four substantive choices (A-D)
    - Always keep "E) I don't know" as the last option
    - Add the incentive/penalty instruction suffix
    """

    question_text = line["Question"].strip()
    correct = line["Correct Answer"].strip()
    incorrects = [
        line["Incorrect Answer 1"].strip(),
        line["Incorrect Answer 2"].strip(),
        line["Incorrect Answer 3"].strip(),
    ]

    # Shuffle A–D while keeping track of the correct answer index
    choices_ad = incorrects + [correct]
    random.shuffle(choices_ad)
    gold_index = choices_ad.index(correct)

    # Build the query with A–D from shuffled list and E fixed as IDK
    query = (
        f"Answer the following multiple choice question.\n\n"
        f"{question_text}\n\n"
        f"A) {choices_ad[0]}\n"
        f"B) {choices_ad[1]}\n"
        f"C) {choices_ad[2]}\n"
        f"D) {choices_ad[3]}\n"
        f"E) I don't know\n\n"
    )

    # Model must output a letter among A–E
    return Doc(
        task_name=task_name,
        query=query,
        choices=LETTER_INDICES[:5],  # ["A","B","C","D","E"]
        gold_index=gold_index,  # only among A–D
        instruction=(
            f"Before answering, think about the question step by step. "
            f"Answer only if you are confident, since mistakes are penalized with -1 points, "
            f"while correct answers receive 1 point, and the answer E) \"I don't know\" always receives 0 points. "
            f"The answer should be 'Answer: ' followed by the letter of the correct answer.")
    )


# -----------------------------
# Custom metrics
# -----------------------------

def extract_letter_fallback(pred: str) -> str | None:
    """Fallback extractor for a single-letter choice when regex extraction fails.

    Heuristics (in order):
      1) LaTeX boxed forms anywhere: $\boxed{X}$, \boxed{X}, boxed(X)
      2) "Answer: X" (strict)
      3) "Final answer: X"
      4) "Option X" or "Choice X"
    """
    if not pred:
        return None

    # 1) LaTeX boxed forms anywhere in the output
    #    Accept variants: $\boxed{E}$, \boxed{E}, boxed(E), boxed{E}, and $\boxed{\text{E}}$
    boxed_patterns = [
        # \boxed{E} with optional surrounding $ ... $
        r"\$?\\boxed\s*\{\s*([A-E])\s*\}\$?",
        # \boxed{\text{E}} with optional spaces and optional surrounding $ ... $
        r"\$?\\boxed\s*\{\s*\\text\s*\{\s*([A-E])\s*\}\s*\}\$?",
        # \boxed{\mathrm{E}} and \boxed{\mathbf{E}} common variants
        r"\$?\\boxed\s*\{\s*\\mathrm\s*\{\s*([A-E])\s*\}\s*\}\$?",
        r"\$?\\boxed\s*\{\s*\\mathbf\s*\{\s*([A-E])\s*\}\s*\}\$?",
        # plain 'boxed(E)' or 'boxed{E}' without backslash
        r"\bboxed\s*\(\s*([A-E])\s*\)",
        r"\bboxed\s*\{\s*([A-E])\s*\}",
    ]
    for pat in boxed_patterns:
        m = re.search(pat, pred, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # 2) Strict format if present anywhere
    strict = re.search(r"\banswer\s*[:\-]?\s*([A-E])\b", pred, re.IGNORECASE)
    if strict:
        return strict.group(1).upper()

    tail = pred[-200:]

    # 3) "final answer: X"
    m = re.search(r"\bfinal\s+answer\s*[:\-]?\s*([A-E])\b", tail, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 4) "option X" or "choice X"
    m = re.search(r"\b(?:option|choice)\s*([A-E])\b", tail, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return None


class ExtractiveLetterIdkGrouped(SampleLevelComputation):
    """Compute trad_score, idk_score and idk_freq together using the same extraction.

    - trad_score: 1 if correct (A-D), else 0 (including E)
    - idk_score: +1 correct (A-D), 0 if E, -1 otherwise
    - idk_freq: 1 if E, else 0
    - extract_fail: 1 if no extraction succeeded for any prediction, else 0
    """

    def __init__(
        self,
        language: Language = Language.ENGLISH,
        aggregation_function=max,
        fallback_mode: str = "first_match",
        extraction_mode: str = "any_match",
        precision: int = 6,
        timeout_seconds: int = 5,
    ):
        self.language = language
        self.gold_extraction_target = [IndicesExtractionConfig(prefix_for_extraction="NativeLetters")]
        self.pred_extraction_target = [IndicesExtractionConfig(prefix_for_extraction="NativeLetters")]
        self.aggregation_function = aggregation_function
        self.fallback_mode = fallback_mode
        self.extraction_mode = extraction_mode
        self.precision = precision
        self.timeout_seconds = timeout_seconds

    def _score_letter(self, letter: str, correct_letter: str, choices: list[str]) -> int:
        if not isinstance(letter, str) or letter not in choices:
            return -1
        if letter == "E":
            return 0
        if letter == correct_letter:
            return 1
        return -1

    def compute(self, doc: Doc, model_response, **kwargs) -> dict:
        golds = doc.get_golds()
        predictions = model_response.final_text

        gold_extraction_regexes = get_extraction_regexes(doc, self.gold_extraction_target, self.language)
        pred_extraction_regexes = get_extraction_regexes(doc, self.pred_extraction_target, self.language)

        extracted_predictions = []
        for pred in predictions:
            preds = extract_target_from_pred(
                pred,
                pred_extraction_regexes,
                self.fallback_mode,
                self.extraction_mode,
                self.timeout_seconds,
            )
            if len(preds) == 0:
                fallback_letter = extract_letter_fallback(pred)
                if fallback_letter is not None:
                    preds = [fallback_letter]
            if len(preds) > 1:
                seen = set()
                preds = [x for x in preds if not (x in seen or seen.add(x))]
            extracted_predictions.append(preds)

        extracted_golds = [
            extract_target_from_pred(
                gold,
                gold_extraction_regexes,
                "no_fallback",
                self.extraction_mode,
                self.timeout_seconds,
            )
            for gold in golds
        ]

        if any(len(g) == 0 for g in extracted_golds):
            extracted_golds = [[gold] for gold in golds]
        else:
            extracted_golds = [
                (list(dict.fromkeys(g)) if len(g) > 1 else g) for g in extracted_golds
            ]

        if doc.specific is None:
            doc.specific = {}
        doc.specific["extracted_predictions"] = [str(pred) for preds in extracted_predictions for pred in preds]
        doc.specific["extracted_golds"] = [str(gold) for golds_ in extracted_golds for gold in golds_]

        correct_letter = doc.choices[doc.gold_index] if 0 <= doc.gold_index < len(doc.choices) else None

        def score_for_one_prediction_group(preds_for_one_text: list[str]) -> tuple[int, int, int, int]:
            if len(preds_for_one_text) == 0:
                return 0, -1, 0, 1
            scores = [self._score_letter(p, correct_letter, doc.choices) for p in preds_for_one_text]
            idk_flags = [1 if p == "E" else 0 for p in preds_for_one_text if isinstance(p, str)]
            trad_scores = [1 if (isinstance(p, str) and p == correct_letter) else 0 for p in preds_for_one_text]
            return max(trad_scores), max(scores), max(idk_flags) if idk_flags else 0, 0

        per_prediction = [score_for_one_prediction_group(preds) for preds in extracted_predictions]
        if len(per_prediction) == 0:
            return {"trad_score": 0.0, "idk_score": -1.0, "idk_freq": 0.0, "extract_fail": 1.0}

        trad_scores = [t for t, _, _, _ in per_prediction]
        idk_scores = [s for _, s, _, _ in per_prediction]
        idk_flags = [f for _, _, f, _ in per_prediction]
        extract_fail_flags = [ef for _, _, _, ef in per_prediction]

        return {
            "trad_score": float(self.aggregation_function(trad_scores)),
            "idk_score": float(self.aggregation_function(idk_scores)),
            "idk_freq": float(max(idk_flags)),
            "extract_fail": float(1 if all(flag == 1 for flag in extract_fail_flags) else 0),
        }


idk_grouped_metrics = SampleLevelMetricGrouping(
    metric_name=["trad_score", "idk_score", "idk_freq", "extract_fail"],
    sample_level_fn=ExtractiveLetterIdkGrouped(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn={"trad_score": np.mean, "idk_score": np.mean, "idk_freq": np.mean, "extract_fail": np.mean},
    higher_is_better={"trad_score": True, "idk_score": True, "idk_freq": False, "extract_fail": False},
)


# -----------------------------
# Task config
# -----------------------------

task = LightevalTaskConfig(
    name="gpqa-diamond-idk",
    prompt_function=gpqa_diamond_idk_prompt,
    suite=["community"],
    hf_repo="Idavidrein/gpqa",
    hf_subset="gpqa_diamond",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=[idk_grouped_metrics],
    generation_size=32768,
    stop_sequence=["\n"],
)

# Export table for discovery
TASKS_TABLE = [task]


