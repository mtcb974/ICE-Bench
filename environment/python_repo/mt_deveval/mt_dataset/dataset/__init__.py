from typing import TypedDict,List,Dict

class DevEvalTurnResult:
    turn: int
    requirement: str
    raw_completion: str
    completion: str

class DevEvalMTResult:
    namespace: str
    completions: List[DevEvalTurnResult]


class DevEvalDependency(TypedDict):
    intra_class: List[str]
    intra_file: List[str]
    cross_file: List[str]

class DevEvalRequirement(TypedDict):
    Functionality: str
    Arguments: str

class DevEvalInstance(TypedDict):
    namespace: str
    type: str
    project_path: str
    completion_path: str
    signature_position: List[int]
    body_position: List[int]
    dependency: DevEvalDependency
    requirement: DevEvalRequirement
    tests: List[str]
    indent: int
    gt: str

class PromptEntry(TypedDict):
    prompt: str
    ds_instance: DevEvalInstance
    system_prompt: str

class ShardDevEval(TypedDict):
    turn: int
    requirement: str
    test_code: str
    tests: List[str]
    gt: str

class MTDevEvalInstance(TypedDict):
    # Orignal dataset
    namespace: str
    type: str
    project_path: str
    completion_path: str
    signature_position: List[int]
    body_position: List[int]
    dependency: DevEvalDependency
    requirement: DevEvalRequirement
    tests: List[str]
    indent: int
    # Post-processing supplement
    domain: str
    gt: str
    context: str
    test_codes: List[str] # The original test code
    # Multi-turn related new fields
    mt: List[ShardDevEval] # The decomposed dataset, containing: (generated when decomposed)turn,requirement (test case synthesis)test_code,tests[test selector],gt
    mt_tests: Dict[int,List[str]] # Multi-turn test selector field, Key is turn, Value is the list of test selectors
    # Function signature
    function_signature:str
