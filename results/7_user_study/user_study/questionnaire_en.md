# ICE-Bench User Study Questionnaire

This document corresponds to the Simplified Chinese participant questionnaire. Field names and rating anchors are identical.

## 1. Scope

This study evaluates multi-turn implementation-oriented coding instructions. Assess the instruction trajectory in the supplied task context and the alignment of each turn's implementation and test with its instruction.

The study does not ask whether a task reproduces complete requirements engineering, stakeholder negotiation, architecture or design, or an end-to-end development lifecycle. `scenario_authenticity` asks only whether a developer could plausibly issue the instructions in the given order while implementing or modifying code.

Annotate independently. Do not discuss answers with other annotators, execute the code or tests, or search for a task online. Use only the task information shown in the annotation page.

## 2. Annotation and Export Rules

1. Enter the anonymous identifier assigned to you in the **Annotator ID** field.
2. Select one score from `1` through `5` for every assessable rating dimension.
3. Read `context` and `instruction_trajectory` and complete the task-level ratings before reading and rating the individual turns.
4. If a dimension cannot be judged from the supplied material, leave its score blank.
5. Comments are optional and are not included in the reported analysis.
6. Use **Export JSON** to create the response file. The exported file records blank scores as `""` and selected scores as JSON integers from `1` through `5`.

## 3. Task Context

Each task's `context` contains three fields, meant to be read in order to understand the full picture of the task:

### 3.1 `project_context` — where the code lives

The project environment surrounding the target code. Function tasks are marked as standalone (no repository context) and provide only the target function or class interface declaration. Repository tasks provide the repository name, target file path, target symbol, and relevant source fragments or API context.

### 3.2 `background_assumptions` — what must not change

Fixed maintenance boundaries that eliminate differences in implicit assumptions across annotators. Use the assumptions stated here as given; do not invent project narratives, version histories, or hidden dependencies.

- **Function tasks**: the implementation is self-contained and does not depend on repository state; the specification and declared interface are the complete task context.
- **Repository tasks**: this is a maintenance task within an existing repository; each turn replaces only the target symbol, and the rest of the project code remains unchanged. The listed dependencies and source context define the local APIs available for the task.

### 3.3 `task_details` — what needs to be done

The complete information about the original task. Function tasks provide the source specification, interface definition, input/output constraints, and execution assumptions. Repository tasks provide the source specification, original documentation or comments, original implementation, test file locations, and test context (when available).

## 4. Task-Level Ratings

### 4.1 Testability (`testability`)

Can the complete context and trajectory be translated into observable, testable behavior without guessing unstated essentials?

| Score | Anchor |
| --- | --- |
| 1 | Multiple central instructions are not operational or observable; reliable tests cannot be derived. |
| 2 | Some behavior is testable, but major inputs, outputs, boundaries, or dependencies require guessing. |
| 3 | Core behavior is testable, but a notable non-central assumption remains unclear. |
| 4 | Almost all behavior is directly testable; only minor wording or boundary details remain. |
| 5 | Every turn states clear behavior, constraints, and observable outcomes sufficient for unambiguous tests. |

### 4.2 Completeness (`completeness`)

Does the complete trajectory cover the goals and constraints explicitly stated in the source specification?

| Score | Anchor |
| --- | --- |
| 1 | The trajectory misses the central goal or ends at a materially different task. |
| 2 | It omits multiple central goals, interfaces, or constraints. |
| 3 | It covers the core goal but leaves a notable functional, boundary, or interface gap. |
| 4 | It covers the goal and major constraints, with only minor omissions. |
| 5 | It covers every explicit goal and constraint without unexplained loss. |

### 4.3 Distinctiveness (`distinctiveness`)

Does each turn introduce a distinguishable behavior, constraint, clarification, or change rather than unnecessary repetition?

| Score | Anchor |
| --- | --- |
| 1 | Several turns are substantially redundant, contradictory, or unrelated. |
| 2 | Multiple turns repeat the same content without a meaningful addition. |
| 3 | Most turns add identifiable content, but substantial overlap or redundancy remains. |
| 4 | Turns are mostly distinct, with only minor overlap. |
| 5 | Every turn makes a clear, non-redundant contribution. |

### 4.4 Scenario Authenticity (`scenario_authenticity`)

Given the supplied function or repository context, could a developer plausibly issue these coding instructions in this order while implementing or modifying code?

| Score | Anchor |
| --- | --- |
| 1 | The sequence is detached from the context and implausible as an implementation interaction. |
| 2 | Isolated details are plausible, but the overall sequence is forced or artificially fragmented. |
| 3 | The sequence could occur during implementation but has a noticeable synthetic or unnatural split. |
| 4 | It resembles ordinary incremental clarification, boundary handling, or implementation change. |
| 5 | Every step has a natural, credible, context-grounded implementation motivation. |

### 4.5 Logical Coherence (`logical_coherence`)

Do the ordering, dependencies, and explicit changes form an understandable and executable trajectory?

| Score | Anchor |
| --- | --- |
| 1 | The trajectory contains irreconcilable contradictions, inverted dependencies, or incompatible behavior. |
| 2 | Major jumps or conflicts make the intended sequence unstable. |
| 3 | The trajectory is basically executable, but a dependency or modification is unclear. |
| 4 | Ordering and dependencies are coherent, with only minor underspecified links. |
| 5 | Ordering, dependencies, and explicit replacements are consistently clear. |

Do not penalize a later instruction merely because it clearly replaces an earlier behavior.

## 5. Turn-Level Ratings

Rate every entry in `turn_artifacts`. Consider the current instruction together with all still-applicable earlier instructions.

### 5.1 Code-to-Instruction Alignment (`code_to_instruction_alignment`)

Does `solution` implement the current instruction and preserve still-applicable earlier requirements?

| Score | Anchor |
| --- | --- |
| 1 | It contradicts the instruction or omits the central behavior. |
| 2 | It implements surface behavior but misses major functionality, interface, or constraints. |
| 3 | It implements the core behavior with a notable semantic, boundary, or interface issue. |
| 4 | It is substantially aligned, with only a minor omission or defect. |
| 5 | It implements all applicable instructions precisely and completely. |

### 5.2 Test-to-Instruction Alignment (`test_to_instruction_alignment`)

Does `test` directly assess behavior stated in the current instruction without forcing an unstated library, API, type, return format, message text, or implementation choice?

| Score | Anchor |
| --- | --- |
| 1 | It is unrelated, contradictory, or primarily tests unstated implementation choices. |
| 2 | It covers some relevant behavior but relies on major unstated assumptions. |
| 3 | It covers the core behavior with a notable omission or over-constraint. |
| 4 | It is substantially aligned, with only a minor coverage or constraint issue. |
| 5 | It directly and sufficiently tests the stated behavior while allowing reasonable alternatives. |

A turn-level test need not repeat every earlier test. Judge whether it evaluates the current addition without introducing an implicit contract.
