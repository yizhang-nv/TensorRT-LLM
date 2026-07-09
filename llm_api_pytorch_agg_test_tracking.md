# LLM API PyTorch Aggregate Test Tracking

This document describes how to maintain
`llm_api_pytorch_agg_test_tracking.csv`. The CSV has one row per exact pytest
node and test-db file from `accuracy/test_llm_api_pytorch.py` for the tracked
DeepSeek, Kimi, GLM, and GPT-OSS aggregate tests. A pytest node that appears in
both `l0_b200.yml` and `l0_h100.yml` must have two rows; `test_db_file` must
never contain a semicolon-separated list. A node with no active test-db entry
has one row with an empty `test_db_file`.

## Validation status

`validation_status` must contain exactly one of these five values:

| Status | Meaning |
| --- | --- |
| `pending` | The case has not completed a validation run that can be recorded. A waived or resource-blocked case remains `pending`; describe the reason in `notes`. |
| `passed` | The case passed on the same canonical GPU model and GPU count as the target test configuration. |
| `partial pass` | The case passed, but the validation GPU model or GPU count differs from the target configuration. For example, using GB200 to validate a B200 case is `partial pass`, not `passed`. |
| `failed` | The validation invocation completed and the test failed. Record the failure summary and log path. |
| `waived` | The case is currently waived and is not expected to be validated. Keep the waiver source in `waive_status` and `waive_reasons`. |

GPU names are compared as canonical model names, not as substring matches.
For example, `B200` and `GB200` are different GPU models even though the
string `GB200` contains `B200`. A result can be `passed` only when the actual
GPU model in `validation_gpu` matches the intended model in `gpu_types` and
the test used the intended count in `validation_gpu_count`.

For tracking purposes, interpret the test-db wildcard combination
`*b100*;*b200*` as the canonical B200 target. The raw combination may remain
in the CSV; do not treat `*b100*` as a separate target when deciding validation
status.

`waive_status` records whether a CI waiver exists, while
`validation_status=waived` is the tracking state used when that waiver means
the case is not expected to be validated. If a waived case is intentionally
run, its validation status may be replaced by the actual result while
`waive_status` and `waive_reasons` remain unchanged.

## Updating a test result

1. Find the row by its complete `pytest_node` and `test_db_file`. Do not update
   rows by method name alone because parametrized cases share the same method,
   and the same pytest node can have different target-GPU rows.
2. Run that exact node and preserve its complete stdout and stderr logs.
3. Fill in the validation fields:
   - `validation_status`: one of the five values above.
   - `validation_node`: hostname on which the test ran.
   - `validation_gpu`: canonical GPU model, such as `H100`, `B200`, or
     `GB200`.
   - `validation_gpu_count`: number of GPUs actually used by the test, not
     merely the number installed in the node.
   - `validation_date`: local date in `YYYY-MM-DD` format.
   - `result_log`: absolute path to the primary stdout log. Mention the stderr
     log in `notes` when it contains relevant diagnostics.
   - `notes`: concise result details, such as accuracy, threshold, runtime,
     failure signature, or why a case remains pending.
4. Compare `validation_gpu` and `validation_gpu_count` with the intended
   `gpu_types` and `gpu_counts` before choosing `passed` versus
   `partial pass`.
5. Parse the CSV after editing to catch quoting or column-count mistakes:

   ```bash
   ruby -rcsv -e 'rows = CSV.read("llm_api_pytorch_agg_test_tracking.csv", headers: true); puts "rows=#{rows.size}"'
   ```

When a case is rerun, keep the most relevant current result. A same-GPU
`passed` result should not be downgraded by a later alternate-GPU
`partial pass`. A later same-GPU failure is a real regression and should be
recorded as `failed`. An alternate-GPU `partial pass` can be upgraded to
`passed` after the intended GPU run succeeds.

## Updating test metadata

The non-validation columns come from the repository's test configuration:

- `qa_lists`: membership in the QA function lists.
- `test_db_file`, `ci_stages`, `gpu_types`, `gpu_counts`, `backends`,
  `orchestrators`, and `auto_triggers`: active entries under
  `tests/integration/test_lists/test-db/`.
- `waive_status` and `waive_reasons`: entries in
  `tests/integration/test_lists/waives.txt`.

When those sources change, update only the affected rows and preserve the
validation fields. New parametrized pytest nodes should be added once for each
active test-db file. Removed nodes should be deleted only after confirming they
are no longer in any tracked QA list or active test-db entry.
