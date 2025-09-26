from __future__ import annotations

import random
import numpy as np
from dotenv import load_dotenv
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.requests import SamplingMethod

import os
import logging
import re

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

def get_answer_letter(pred: str) -> str:
    if not pred:
        return None
    # Strictly parse "Answer: {letter}"
    match = re.search(r"answer\s*:\s*([A-E])", pred, re.IGNORECASE)
    if not match:
        logging.warning("Could not parse 'Answer: {letter}' from prediction: %r", pred)
        return None
    return match.group(1).upper()


class GPQAScore(SampleLevelComputation):
    """Score per sample: +1 correct (A-D), -1 wrong (A-D), 0 for E (I don't know)."""

    def __init__(self, idk: bool):
        self.idk = idk

    def is_idk(self, pred_letter: str) -> bool:
        return pred_letter == "E" if self.idk else False

    def compute(self, doc: Doc, model_response, **kwargs) -> int:
        # Model is expected to answer with one of the letters present in doc.choices, e.g., "A".."E"
        pred_letter = get_answer_letter((model_response.final_text[0] or "").strip())
        if not pred_letter:
            return 0

        if pred_letter not in doc.choices:
            logging.warning("Invalid prediction. pred_raw=%r choices=%s", pred_letter, doc.choices)
            return 0

        correct_letter = (
            doc.choices[doc.gold_index] if 0 <= doc.gold_index < len(doc.choices) else None
        )

        # Map letters A–D to indices and compare to gold
        pred_ix = doc.choices.index(pred_letter)
        # If the model answered correctly, score is 1
        if pred_ix == doc.gold_index:
            score = 1
        else:
            if self.idk:
                # If we have the IDK score, it's -1 for wrong
                score = -1
                # Unless the model answered IDK, score is 0
                if self.is_idk(pred_letter):
                    score = 0
            else:
                # For the traditional score, it's 0 for wrong
                score = 0

        logging.debug("Correct: %s", correct_letter)
        logging.debug("Predicted: %s", pred_letter)
        logging.debug("Score: %s", score)
        logging.debug("Model response: %r", getattr(model_response, "final_text", None))
        return score


class GPQAIdkFlag(SampleLevelComputation):
    """Return 1 if the model answered E (I don't know), 0 otherwise."""

    def compute(self, doc: Doc, model_response, **kwargs) -> int:
        pred_letter = get_answer_letter((model_response.final_text[0] or "").strip())
        if not pred_letter:
            return 0
        return int(pred_letter == "E")


def corpus_mean_or_zero(flags):
    """Return the mean of flags as a float, or 0.0 if empty.

    Defined at module top-level to ensure picklability under multiprocessing.
    """
    if len(flags) == 0:
        return 0.0
    return float(np.mean(flags))

traditional_score_metric = SampleLevelMetric(
    metric_name="traditional_score",
    sample_level_fn=GPQAScore(idk=False),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)

idk_score_metric = SampleLevelMetric(
    metric_name="idk_score",
    sample_level_fn=GPQAScore(idk=True),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)

idk_percent_metric = SampleLevelMetric(
    metric_name="idk_percent",
    sample_level_fn=GPQAIdkFlag(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=corpus_mean_or_zero,
    higher_is_better=False,
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
    metrics=[traditional_score_metric, idk_score_metric, idk_percent_metric],
    generation_size=8192,
    stop_sequence=["\n"],
)


# Export table for discovery
TASKS_TABLE = [task]


