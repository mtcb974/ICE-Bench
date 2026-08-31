# Failure-Attribution Validation Questionnaire

This document corresponds to the participant questionnaire for the human validation of the automatic failure attribution. Field names are identical in the released rating files.

## 1. Scope

Each record is one sampled model output that failed the execution-based tests. The record shows the cumulative requirements, the model-generated code, the current-round tests, the test execution information, and the proposed automatic attribution (one primary category, at most one secondary category, and a short reason).

Annotate independently. Do not discuss item-level answers with other annotators.

## 2. Question

For each record, answer `Accept` or `Reject`:

`expert_decision`: Is the proposed failure attribution reasonable given the cumulative requirements, the model-generated code, the current-round tests, and the test execution information?

The question asks whether the proposed attribution is reasonable, not whether the annotator would have assigned the same category independently.

## 3. Reported Statistics

The released `statistics.json` reports the count of records whose three-annotator majority answer is `Accept` and the count of unanimous records. The study covers 180 sampled model outputs.
