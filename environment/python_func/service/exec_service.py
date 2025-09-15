from bigcodebench.eval import untrusted_check

LIMIT_CONFIG = {
    "max_as_limit" : 50 * 1024,
    "max_data_limit" : 50 * 1024,
    "max_stack_limit" : 15,
    "min_time_limit" : 0.1,
    "gt_time_limit" : 2.0
}

def exec_service(code:str, test:str, entry_point:str) -> dict:
    full_code = code + "\n" + test
    stat, details = untrusted_check(
        full_code,
        test,
        entry_point,
        LIMIT_CONFIG["max_as_limit"],
        LIMIT_CONFIG["max_data_limit"],
        LIMIT_CONFIG["max_stack_limit"],
        LIMIT_CONFIG["min_time_limit"],
        LIMIT_CONFIG["gt_time_limit"],
    )

    return {
        "status": stat,   #  "pass", "fail", "error"
        "details": details, 
    }

