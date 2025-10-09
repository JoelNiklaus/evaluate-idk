from __future__ import annotations

import random
import logging
import ast
import re
import numpy as np
import os
import litellm

from dotenv import load_dotenv

from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.metrics.utils.extractive_match_utils import (
    IndicesExtractionConfig,
    get_extraction_regexes,
    extract_target_from_pred,
)
from lighteval.utils.language import Language


logging.basicConfig(level=os.getenv("EVALUATE_IDK_LOG_LEVEL", "INFO"))

load_dotenv()

litellm.suppress_debug_info = True

random.seed(42)


logger = logging.getLogger(__name__)


NUM_CHOICES = 5  # A, B, C, D, E (where E is "I don't know")

# Task configuration constants
GENERATION_SIZE = 32768
STOP_SEQUENCES = ["\n", "\n\n"]

PROMPT_TEMPLATE = """Question:
{question_text}

Choices:
{choices_str}

Before answering, think about the question step by step.
Answer only if you are confident, since mistakes are penalized with -1 points, while correct answers receive 1 point, and the answer E) \"I don't know\" always receives 0 points.

Format your final answer as follows:
Final Answer: ###C###

Answer:
"""

# -----------------------------
# Prompt functions
# -----------------------------


def shuffle_choices(choice_list: list[str], gold_index: int) -> tuple[list[str], int]:
    """Shuffle choices while tracking the correct answer's new position.
    
    Args:
        choice_list: List of 4 answer choices
        gold_index: Current index of the correct answer (0-3)
        
    Returns:
        Tuple of (shuffled_choices, new_gold_index)
    """
    correct_answer = choice_list[gold_index]
    shuffled = choice_list.copy()
    random.shuffle(shuffled)
    new_gold_index = shuffled.index(correct_answer)
    return shuffled, new_gold_index


def build_choices_string(choice_list: list[str]) -> str:
    """Build a formatted string of choices A-D plus E) I don't know.
    
    Args:
        choice_list: List of 4 answer choices
        
    Returns:
        Formatted string with A-D choices and E) I don't know
    """
    choices_str = ""
    for letter, choice in zip(LETTER_INDICES[:NUM_CHOICES - 1], choice_list):
        choices_str += f"{letter}) {choice}\n"
    choices_str += f"{LETTER_INDICES[NUM_CHOICES - 1]}) I don't know"
    return choices_str


def lexam_idk_prompt(sample, task_name: str = None):
    """Convert a LEXam row to a Doc with 4 shuffled choices + E: I don't know.
    
    Expected input fields:
    - question: the question text
    - choices: list of 4 answer choices
    - course: the course name
    - gold: index of correct answer (0-3)
    """
    course_name = sample["course"]
    question_text = sample["question"].strip()
    
    if isinstance(sample["choices"], list):
        choice_list = sample["choices"]
    else:
        choice_list = ast.literal_eval(sample["choices"])
    
    # Shuffle choices while tracking the correct answer
    shuffled_choices, gold_index = shuffle_choices(choice_list, sample["gold"])
    
    choices_str = build_choices_string(shuffled_choices)
    
    # Build the query with A–D from the choice list and E fixed as IDK
    instruction = f"""You are an expert in {course_name} and address legal issues in a structured, exam-style manner.
You are given a multiple-choice question, where only one choice (e.g., A, B, C, etc.) is correct.
Assume Swiss law applies unless specifically stated otherwise. If the context of the course justifies it, consider legal frameworks beyond Swiss law as well.

Please reason through the question step by step, using a chain-of-thought approach:
- Clarify the facts: Briefly restate or highlight the key facts in the question to anchor your reasoning.
- Issue Identification: What legal issue(s) arise from the facts?
- Rule Explanation: What legal rules or principles are relevant, and what are their sources (e.g., statutes, case law, doctrine)?
- Application and Reasoning: Apply the relevant rules to the facts, carefully weighing any ambiguities, exceptions, or competing interpretations.
- Eliminate Incorrect Answers: Briefly explain why each incorrect answer is wrong or less convincing.
- Conclusion: Clearly state the correct answer choice (e.g., A, B, C, etc.) with a brief justification for why it best fits the legal analysis.
"""
    
    return Doc(
        task_name=task_name,
        query=PROMPT_TEMPLATE.format(question_text=question_text, choices_str=choices_str),
        choices=LETTER_INDICES[:NUM_CHOICES],  # ["A","B","C","D","E"]
        gold_index=gold_index,  # gold is among A–D (0-3)
        instruction=instruction,
    )


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

    # Shuffle A–D while tracking the correct answer index
    choices_ad = incorrects + [correct]
    shuffled_choices, gold_index = shuffle_choices(choices_ad, gold_index=3)

    choices_str = build_choices_string(shuffled_choices)

    instruction = f"""Answer the following multiple choice question."""

    return Doc(
        task_name=task_name,
        query=PROMPT_TEMPLATE.format(question_text=question_text, choices_str=choices_str),
        choices=LETTER_INDICES[:NUM_CHOICES],  # ["A","B","C","D","E"]
        gold_index=gold_index,  # only among A–D
        instruction=instruction,
    )


# -----------------------------
# Custom metrics
# -----------------------------

def extract_letter_fallback(pred: str) -> str | None:
    """Fallback extractor for a single-letter choice when regex extraction fails.

    Heuristics (in order):
      1) LEXam format: ###X###
      2) LaTeX boxed forms anywhere: $\boxed{X}$, \boxed{X}, boxed(X)
      3) "Answer: X" (strict)
      4) "Final answer: X"
      5) "Option X" or "Choice X"
    """
    if not pred:
        return None

    # 1) LEXam format: ###X### (extract last occurrence)
    matches = re.findall(r"###([ABCDE])###", pred)
    if matches:
        return matches[-1].upper()

    # 2) LaTeX boxed forms anywhere in the output
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

    # 3) Strict format if present anywhere
    strict = re.search(r"\banswer\s*[:\-]?\s*([A-E])\b", pred, re.IGNORECASE)
    if strict:
        return strict.group(1).upper()

    tail = pred[-200:]

    # 4) "final answer: X"
    m = re.search(r"\bfinal\s+answer\s*[:\-]?\s*([A-E])\b", tail, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 5) "option X" or "choice X"
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
# Task configurations
# -----------------------------

lexam_idk_task = LightevalTaskConfig(
    name='lexam-en-idk',
    prompt_function=lexam_idk_prompt,
    hf_repo="LEXam-Benchmark/LEXam",
    hf_subset="mcq_4_choices",
    hf_filter=lambda x: x["language"] == "en",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=[idk_grouped_metrics],
    suite=["community"],
    generation_size=GENERATION_SIZE,
    stop_sequence=STOP_SEQUENCES,
)

gpqa_diamond_idk_task = LightevalTaskConfig(
    name="gpqa-diamond-idk",
    prompt_function=gpqa_diamond_idk_prompt,
    suite=["community"],
    hf_repo="Idavidrein/gpqa",
    hf_subset="gpqa_diamond",
    hf_filter=None,
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=[idk_grouped_metrics],
    generation_size=GENERATION_SIZE,
    stop_sequence=STOP_SEQUENCES,
)

# Export table for discovery
TASKS_TABLE = [lexam_idk_task, gpqa_diamond_idk_task]

if __name__ == "__main__":
    print([t.name for t in TASKS_TABLE])
    print(len(TASKS_TABLE))






