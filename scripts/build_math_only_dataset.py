"""Slice the MATH half out of the gsm8k_math mix, keeping the mixed test set.

Train on MATH only (7498) but validate on GSM8K test + MATH-500 (1819), so val-core/gsm8k/*
stays comparable to the archived logs while becoming a pure transfer measurement.

verl resolves data.train_files to ${vars.dir}/${TASK}/train.parquet and vars.dir is each run
script's own PROJECT_ROOT, so the directory has to exist in both the SDPO and SDPO_official
trees.
"""

import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path("/home/xiongrengrong.xrr/SDCL_A_TOKEN")
SRC_PARQUET = ROOT / "sdpo/SDPO/datasets/gsm8k_math"
SRC_JSON = ROOT / "sdpo/SDPO_official/datasets/gsm8k_math/train.json"
TREES = [ROOT / "sdpo/SDPO/datasets/math_only", ROOT / "sdpo/SDPO_official/datasets/math_only"]
JSON_OUT = ROOT / "datasets/math_only_train.json"


def main():
    train = pd.read_parquet(SRC_PARQUET / "train.parquet")
    math_train = train[train.data_source == "math"].reset_index(drop=True)
    assert len(math_train) == 7498, len(math_train)

    test = pd.read_parquet(SRC_PARQUET / "test.parquet")
    assert len(test) == 1819, len(test)

    for d in TREES:
        d.mkdir(parents=True, exist_ok=True)
        math_train.to_parquet(d / "train.parquet", index=False)
        shutil.copy2(SRC_PARQUET / "test.parquet", d / "test.parquet")
        print(f"{d}: train {len(math_train)}  test {len(test)}")

    records = json.load(open(SRC_JSON))
    math_records = [r for r in records if r.get("data_source") == "math"]
    if not math_records:
        # the json may not carry data_source; the mix is gsm8k[0:7473] then math[7473:]
        math_records = records[7473:]
    assert len(math_records) == 7498, len(math_records)
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(math_records, open(JSON_OUT, "w"))
    print(f"{JSON_OUT}: {len(math_records)} records, keys {sorted(math_records[0])}")


if __name__ == "__main__":
    main()
