# Manual Review Questionnaire

This document corresponds to the participant questionnaire for the manual review of the benchmark tasks derived from MRG-Bench and AutoCodeBench. Field names are identical in the released rating files.

## 1. Scope

Each record is one turn of a benchmark task. The record shows the task context, the cumulative requirements through that turn, the current-turn instruction, the current-turn reference solution, and the current-turn tests. Answer the two questions below for every record.

Annotate independently. Do not discuss item-level answers with other annotators, execute the code or tests, or search for the task online.

## 2. Questions

For each turn, answer `Yes` or `No`:

1. `requirement_actionability`: Are the cumulative requirements through this round, together with the provided context, sufficient for a competent developer to implement the intended behavior?
2. `evaluation_reasonableness`: Do this round's tests reasonably evaluate the newest requirement(s) introduced in this round, given the cumulative requirement context?

For question 2, judge only the requirement(s) newly introduced in the current round: a current-round test need not cover earlier-round requirements as long as it reasonably covers the newest one(s) under the cumulative context.

## 3. Reported Statistics

The released `statistics.json` reports, per question, the majority-Yes count and the count of records on which all three annotators agreed. The review covers 920 turns from 212 tasks (172 tasks derived from AutoCodeBench and 40 from MRG-Bench).
