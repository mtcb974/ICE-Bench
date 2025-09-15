from fastapi import FastAPI, UploadFile, File, HTTPException,Query
from pydantic import BaseModel
import json
from typing import List, Optional,Any,Dict,Tuple
from service.exec_service import exec_service
import logging
from collections import defaultdict
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from contextlib import asynccontextmanager
from bigcodebench.eval import PASS, untrusted_check
import asyncio

# config log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GROUND_TRUTH_PATH = "dataset/mt_bigcode.json"
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)
# (mt_id, turn) -> list of {task_id, test_code, entry_point, code}
ground_truth_map: Dict[str, Dict] = {} 

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Starting up, loading ground truth...")
    load_ground_truth()

    yield

    logging.info("Shutting down...")

app = FastAPI(lifespan=lifespan)

# Load the ground truth data into memory
def load_ground_truth():
    global ground_truth_map
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        mt_id = item["mt_id"]
        mt_data = item["mt_data"]
        for turn_data in mt_data:
            task_id = turn_data["task_id"]
            turn = turn_data["turn"]
            test_code = turn_data["test"]
            entry_point = turn_data.get("entry_point", "")
            code = turn_data["code"]

            key = (mt_id, turn)
            if key not in ground_truth_map:
                ground_truth_map[key] = []
            ground_truth_map[key].append({
                "task_id": task_id,
                "test_code": test_code,
                "entry_point": entry_point,
                "gt_code": code,
            })
    logger.info("Ground truth loaded.")
    print(f"len of grouth truth: {len(ground_truth_map)}")


class ExecuteRequest(BaseModel):
    code: str
    test: str
    entry_point: Optional[str] = ""

class ExecuteResponse(BaseModel):
    status: Any
    details: Any

# Create a dedicated thread pool
executor = ThreadPoolExecutor(max_workers=8)

@app.post("/verify/", response_model=ExecuteResponse)
async def verify(request: ExecuteRequest):
    try:
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(loop.run_in_executor(
            executor,
            exec_service,
            request.code,
            request.test,
            request.entry_point
        ),timeout=150)
        return result
    except asyncio.TimeoutError:
        # A TimeoutError will be thrown when the execution exceeds xx seconds.
        print("Execution timed out")
        raise HTTPException(
            status_code=408,  # Request Timeout
            detail="Code execution timed out"
        )
    except Exception as e:
        print(f"Execution error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Execution error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to BigCodeBench Execution API"}

def evaluate_one(mt_id: int, turns_dict: Dict[str, Tuple[List, List]]) -> Tuple[int, Dict]:
    """
    :param mt_id: sample's mt_id
    :param turns_dict: (solutions, gts) for each turn
    :return: (mt_id, {turn: [detailed_result]})
    """
    result_by_turn = dict()
    
    # Evaluate the data of each round in sequence
    for turn, (solutions, gts) in turns_dict.items():
        one_turn_results = []

        for sol, gt in zip(solutions, gts):
            code = sol["solution"]
            test_code = gt["test_code"]
            entry_point = gt["entry_point"]

            status, output = untrusted_check(
                code=code,
                test_code=test_code,
                entry_point=entry_point,
                max_as_limit=30 * 1024,
                max_data_limit=30 * 1024,
                max_stack_limit=10,
                min_time_limit=0.1,
                gt_time_limit=20.0
            )

            passed = (status == PASS)
            one_turn_results.append({
                "mt_id": mt_id,
                "turn": turn,
                "task_id": gt["task_id"],
                "passed": passed,
                "solution": code,
                "gt_test_code": test_code,
                "entry_point": entry_point,
                "output": output
            })

        result_by_turn[turn] = one_turn_results

    return mt_id, result_by_turn


def save_evaluation_report(
    metrics: Dict,
    detailed_results: List[Dict],
    llm: str,
    eval_type: str
):
    safe_llm = llm.replace("/", "_")  
    safe_type = eval_type.replace("/", "_")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = os.path.join(RESULT_DIR, f"evaluation_{safe_llm}_{safe_type}.json")

    report = {
        "timestamp": timestamp,
        "metrics": metrics,
        "llm": llm,
        "type": eval_type,
        "detailed_results": detailed_results
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    logging.info(f"Evaluation report saved to {filename}")

    return filename

# Multi-round evaluation interface
@app.post("/evaluate")
async def evaluate_result_file(
    upload_file: UploadFile = File(...),
    llm: str = Query("unknown", description="Large model name"),
    type: str = Query("default", description="eval type")
):
    # Read the evaluation script uploaded by the user
    contents = await upload_file.read()
    lines = contents.decode("utf-8").strip().splitlines()
    results = [json.loads(line) for line in lines]
    print(f"total task: {len(results)}")

    tasks = []
    mtid_turn_map = defaultdict(dict)  # mt_id -> {turn: (solutions, gts)}

    # Process each sample
    for item in results:
        mt_id = item["mt_id"]
        solutions = item["solutions"]  # list of {"turn": "1", "solution": "..."}
        turns = set(sol["turn"] for sol in solutions)

        for turn in turns:
            sols_per_turn = [s for s in solutions if s["turn"] == turn]
            gts_per_turn = ground_truth_map.get((mt_id, turn), [])
            if len(sols_per_turn) != len(gts_per_turn):
                logger.warning(f"Length mismatch for mt_id={mt_id}, turn={turn}")
                continue
            mtid_turn_map[mt_id][turn] = (sols_per_turn, gts_per_turn)
    tasks = [(mt_id, mtid_turn_map[mt_id]) for mt_id in mtid_turn_map]

    num_workers = min(multiprocessing.cpu_count(), 10)
    pass_counts = dict()
    total_counts = dict()
    detailed_results = []
    fully_completed = 0 

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(evaluate_one, mt_id, turns_dict)
            for mt_id, turns_dict in tasks
        ]
        per_mt_results = dict()

        for future in futures:
            mt_id, res_by_turn = future.result()
            per_mt_results[mt_id] = res_by_turn

            all_passed = True
            for turn, res_list in res_by_turn.items():
                passed_list = [r["passed"] for r in res_list]
                all_passed &= all(passed_list)

                pass_counts[turn] = pass_counts.get(turn, 0) + sum(1 for p in passed_list if p)
                total_counts[turn] = total_counts.get(turn, 0) + len(passed_list)
                detailed_results.extend(res_list)

            if all_passed:
                fully_completed += 1

    metrics = {}
    for turn in sorted(total_counts.keys()):
        total = total_counts[turn]
        passed = pass_counts[turn]
        accuracy = round(passed / total, 4) if total > 0 else 0
        metrics[f"turn_{turn}"] = {
            "total_samples": total,
            "correct": passed,
            "accuracy": accuracy,
        }
    
    metrics["fully_completed_samples"] = {
        "count": fully_completed,
        "total": len(per_mt_results),
        "ratio": round(fully_completed / len(per_mt_results), 4) if per_mt_results else 0
    }

    filename = save_evaluation_report(metrics, detailed_results,llm,type)

    return {
        "status": "success",
        "metrics": metrics,
        "llm": llm,
        "type": type,
        "saved_to": filename
    }