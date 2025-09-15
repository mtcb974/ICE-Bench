# Annotation Guidelines

## Scoring Scheme

All dimensions are evaluated on a 5-point Likert scale, where higher scores indicate better quality:
- 1 (Very Poor): The instruction/code/test is unusable, severely flawed, or fails to meet the requirement.
- 2 (Poor): The instruction/code/test is partially understandable but contains major issues that prevent reliable use.
- 3 (Acceptable): The instruction/code/test meets the basic requirements but has noticeable limitations or deficiencies.
- 4 (Good): The instruction/code/test is mostly correct, coherent, and effective, with only minor issues.
- 5 (Excellent): The instruction/code/test is fully correct, precise, and of high quality, with no significant flaws.

## Evaluation Dimensions

### Instruction Quality

- **Testability**: Instructions should specify functions that can be validated through well-defined unit tests. They must avoid vague expressions such as “make the code look better” or “improve efficiency”.
- **Completeness**: The full sequence of instructions should cumulatively resolve the original high-level programming task in its entirety.
- **Distinctiveness**: Each instruction should introduce new requirements. Redundant or repeated requirements across turns should be avoided.
- **Scenario Authenticity**: The sequence of instructions should reflect realistic software development scenarios.
- **Logical Coherence**: The instructions should form a clear and logically consistent development trajectory. If the order of instructions does not convey a coherent story, the sequence should be revised to achieve this.

### Code and Test Quality

- **Test Case Quality**: Test cases must correctly reflect the requirements introduced in the current turn. They should not merely test the initial requirements or duplicate earlier tests. Inconsistencies between requirements and tests—such as mismatches in naming/signature or test messages—must not occur.
- **Code Quality**: Code should correctly implement the requirements introduced in the current turn while preserving the functionality of prior turns.