from datasets import load_dataset
from typing import Any, List, Tuple, Optional

import pandas as pd
import numpy as np
import random
import ast

# Set a random seed for reproducibility
RANDOM_STATE = 42
N_SAMPLES = 128

lexam = load_dataset("LEXam-Benchmark/LEXam", "mcq_16_choices")
lexam = pd.DataFrame(lexam["test"])

lexam['polarity'] = lexam['negative_question'].apply(lambda x: 'neg' if x else 'pos')

# Sample up to N rows per language
lexam = (
    lexam
    .groupby(["language", 'polarity'], group_keys=False)
    .apply(
        lambda x: x.sample(
            n=min(len(x), N_SAMPLES),
            random_state=RANDOM_STATE,
        )
    )
    .reset_index(drop=True)
)

def create_category(row_0, row_1):

  return 'lexam_' + row_0 + '_' + row_1

lexam['category'] = lexam.apply(lambda row: create_category(row['language'], row['polarity']), axis=1)

rename_mappings = {
    'choices': 'options',
    'gold': 'answer_index',
    'id': 'question_id',
}

lexam = lexam.rename(columns=rename_mappings)

mmlu = load_dataset("TIGER-Lab/MMLU-Pro")
mmlu = pd.DataFrame(mmlu["test"])

mmlu = mmlu[mmlu['options'].apply(lambda x: len(x) == 10)]
mmlu = mmlu[mmlu["category"] != "other"]

# Sample up to N rows per category
mmlu = (
    mmlu
    .groupby("category", group_keys=False)
    .apply(
        lambda x: x.sample(
            n=min(len(x), N_SAMPLES),
            random_state=RANDOM_STATE,
        )
    )
    .reset_index(drop=True)
)

cols = ['question_id', 'question', 'options', 'answer_index', 'category']
data = pd.concat([mmlu[cols], lexam[cols]], ignore_index=True)

def ensure_list(x):
    """
    Ensure options are a Python list.
    Handles:
    - list -> returned as-is
    - string representation of list -> safely parsed
    """
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception as e:
            raise ValueError(f"Failed to parse options string: {x}") from e
    raise TypeError(f"options must be a list or str, got {type(x)}")

data["options"] =  data["options"].apply(ensure_list)

def reduce_options_keep_answer(
    options,
    answer_index: int,
    k: int = 4,
    rng: Optional[random.Random] = None,
) -> Tuple[List[str], int]:

    if rng is None:
        rng = random.Random()

    options = ensure_list(options)
    n = len(options)

    if n < k:
        raise ValueError(f"options length {n} < k={k}")
    if not (0 <= answer_index < n):
        raise ValueError(f"answer_index {answer_index} out of range")

    correct = options[answer_index]
    wrongs = [opt for i, opt in enumerate(options) if i != answer_index]

    sampled_wrongs = rng.sample(wrongs, k - 1)

    new_options = [correct] + sampled_wrongs
    rng.shuffle(new_options)

    new_answer_index = new_options.index(correct)
    return new_options, new_answer_index

def reduce_dataframe_mcq_options(
    data: pd.DataFrame,
    options_col: str = "options",
    answer_col: str = "answer_index",
    k: int = 4,
    seed: int = 42,
    inplace: bool = True,
) -> pd.DataFrame:

    rng = random.Random(seed)

    def _apply(row):
        return reduce_options_keep_answer(
            row[options_col],
            row[answer_col],
            k=k,
            rng=rng,
        )

    reduced = data.apply(_apply, axis=1, result_type="expand")
    reduced.columns = ["_new_options", "_new_answer_index"]

    if inplace:
        data[options_col] = reduced["_new_options"]
        data[answer_col] = reduced["_new_answer_index"]
        return data

    out = data.copy()
    out[f"options_{k}"] = reduced["_new_options"]
    out[f"answer_index_{k}"] = reduced["_new_answer_index"]
    return out

for k in range(2, 11):

    data = reduce_dataframe_mcq_options(data, k=k, seed=RANDOM_STATE, inplace=False)