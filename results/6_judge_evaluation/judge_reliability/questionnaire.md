# Judge-Reliability Questionnaire

This document corresponds to the participant questionnaire for the human reliability study of the LLM Judge. Field names are identical in the released rating files.

## 1. Scope

Each record is one generated-code turn from the Judge evaluation, sampled from RQ1 Base Full-History trajectories (20 tasks per benchmark subset, 317 turns in total). The record shows the cumulative requirements through that turn, the supporting context, and the generated code to assess. It does not show the Judge verdict, execution outcomes, model names, or source-benchmark names.

Annotate independently. Do not discuss individual records with other annotators or revise a submitted answer after seeing another annotator's answer.

## 2. Question

For each record, answer `Yes` or `No`:

`human_verdict`: Does the generated code satisfy all explicit cumulative requirements in the displayed scope?

Ambiguity that the displayed material does not resolve is assessed conservatively as `No`.

## 3. Reported Statistics

The released `statistics.json` reports the human-human reliability as nominal Krippendorff's alpha over the three annotators, and the Judge-human agreement as Cohen's kappa between the Judge verdict and the majority human verdict, together with the per-record comparison.
