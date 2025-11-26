## Meta-analysis during benchmark construction

Note: `interaction_number` refers to the number of times the process switches from one node to another, not the number of large model calls.

### Overall Averages

| Metric                   | total_prompt_tokens | total_completion_tokens | total_tokens | interaction_number |
|--------------------------|--------------------|------------------------|--------------|-------------------|
| **Overall Average**      | 27,264.12          | 11,058.54              | 38,322.66    | 25.51             |

### Per Task Averages

| Task                    | total_prompt_tokens | total_completion_tokens | total_tokens | interaction_number |
|-------------------------|--------------------|------------------------|--------------|-------------------|
| codegen                 | 21,435.87          | 10,750.92              | 32,186.79    | 22.95             |
| instruction decomposition| 6,920.15           | 855.26                 | 7,775.41     | 3.73              |

## Running Example

```json
{
    "hash_id": "52a0cece715a19e8c14b677a",
    "tasks": {
        "codegen": {
            "status": "success",
            "task": "codegen",
            "fail_reason": null,
            "source": "autocodebench",
            "total_prompt_tokens": 16596,
            "total_completion_tokens": 4438,
            "total_tokens": 21034,
            "interaction_number": 18,
            "interactions": [
                {
                    "agent": "coder",
                    "response_metadata": {
                        "token_usage": {
                            "completion_tokens": 543,
                            "prompt_tokens": 1297,
                            "total_tokens": 1840
                        },
                        "model_name": "claude-sonnet-4-20250514",
                        "system_fingerprint": null,
                        "id": "chatcmpl-full",
                        "service_tier": null,
                        "finish_reason": "stop",
                        "logprobs": null
                    },
                    "content": {
                        "solution": "import java.util.concurrent.*;\n\nclass ThreadExecutor {\n    public static <T> T executeTaskWithTimeout(Callable<T> task, long timeoutMs) throws Exception {\n        ExecutorService executor = Executors.newSingleThreadExecutor();\n        try {\n            Future<T> future = executor.submit(task);\n            return future.get(timeoutMs, TimeUnit.MILLISECONDS);\n        } finally {\n            executor.shutdown();\n        }\n    }\n}\n\nclass SampleTask implements Callable<String> {\n    private final String name;\n    private final long sleepTime;\n    \n    public SampleTask(String name, long sleepTime) {\n        this.name = name;\n        this.sleepTime = sleepTime;\n    }\n    \n    @Override\n    public String call() throws Exception {\n        System.out.println(\"Starting task: \" + name);\n        Thread.sleep(sleepTime);\n        return \"Result from \" + name + \" after \" + sleepTime + \"ms\";\n    }\n}",
                        "test": "import static org.junit.jupiter.api.Assertions.*;\nimport org.junit.jupiter.api.Test;\nimport java.util.concurrent.*;\n\nclass TestThreadExecutor {\n    \n    @Test\n    void testExecuteTaskWithTimeoutSuccess() throws Exception {\n        SampleTask task = new SampleTask(\"TestTask\", 100);\n        String result = ThreadExecutor.executeTaskWithTimeout(task, 500);\n        assertEquals(\"Result from TestTask after 100ms\", result);\n    }\n    \n    @Test\n    void testExecuteTaskWithTimeoutException() {\n        SampleTask task = new SampleTask(\"SlowTask\", 500);\n        assertThrows(TimeoutException.class, () -> {\n            ThreadExecutor.executeTaskWithTimeout(task, 100);\n        });\n    }\n    \n    @Test\n    void testSampleTaskCall() throws Exception {\n        SampleTask task = new SampleTask(\"MyTask\", 50);\n        String result = task.call();\n        assertEquals(\"Result from MyTask after 50ms\", result);\n    }\n    \n    @Test\n    void testSampleTaskWithZeroSleepTime() throws Exception {\n        SampleTask task = new SampleTask(\"QuickTask\", 0);\n        String result = task.call();\n        assertEquals(\"Result from QuickTask after 0ms\", result);\n    }\n}",
                        "reason": null,
                        "plan": null
                    }
                },
                {
                    "agent": "check_correctness",
                    "response_metadata": null,
                    "content": {
                        "status": "pass",
                        "detail": "===exec_outcome===\nPASSED\n===exec_runtime_message===\n\n                ======================\n                Total tests:     4\n                Successful:      4\n                Failed:          0\n                Aborted:         0\n                Skipped:         0\n                Total time:      0.323 s\n            \n===stdout:===\n\nThanks for using JUnit! Support its development at https://junit.org/sponsoring\n\nStarting task: TestTask\nStarting task: SlowTask\nStarting task: QuickTask\nStarting task: MyTask\n\n===stderr:===\n"
                    }
                },
                {
                    "agent": "codegen_evaluator",
                    "response_metadata": {
                        "token_usage": {
                            "completion_tokens": 203,
                            "prompt_tokens": 1695,
                            "total_tokens": 1898,
                            "completion_tokens_details": null,
                            "prompt_tokens_details": null,
                            "estimated_cost": 0.00034214999999999994
                        },
                        "model_name": "Qwen/Qwen3-235B-A22B-Instruct-2507",
                        "system_fingerprint": null,
                        "id": "chatcmpl-Rus9TmVqryM36H9GNCIexWGK",
                        "service_tier": null,
                        "finish_reason": "stop",
                        "logprobs": null
                    },
                    "content": {
                        "rule_results": [
                            {
                                "rule": "Naming/Signature Inconsistency",
                                "result": "pass",
                                "reason": "The test correctly calls 'executeTaskWithTimeout' and 'SampleTask.call()' as defined in the current instruction and solution."
                            },
                            {
                                "rule": "Message Inconsistency",
                                "result": "fail",
                                "reason": "The test assumes a specific console output format like 'Starting task: [name]', but the instruction does not require this exact message format."
                            }
                        ],
                        "decision": "TEST_REFINE",
                        "feedback": "The test cases expect a specific print format 'Starting task: [name]' from SampleTask.call(), but the current instruction does not specify the exact message to be printed. Either the instruction should explicitly define the required output format, or the test should not assert on console output. Remove or adjust stdout assertions unless the message format is officially mandated."
                    }
                },
                {
                    "agent": "coder",
                    "response_metadata": {
                        "token_usage": {
                            "completion_tokens": 673,
                            "prompt_tokens": 1854,
                            "total_tokens": 2527
                        },
                        "model_name": "claude-sonnet-4-20250514",
                        "system_fingerprint": null,
                        "id": "chatcmpl-full",
                        "service_tier": null,
                        "finish_reason": "stop",
                        "logprobs": null
                    },
                    "content": {
                        "solution": "import java.util.concurrent.*;\n\nclass ThreadExecutor {\n    public static <T> T executeTaskWithTimeout(Callable<T> task, long timeoutMs) throws Exception {\n        ExecutorService executor = Executors.newSingleThreadExecutor();\n        try {\n            Future<T> future = executor.submit(task);\n            return future.get(timeoutMs, TimeUnit.MILLISECONDS);\n        } finally {\n            executor.shutdown();\n        }\n    }\n}\n\nclass SampleTask implements Callable<String> {\n    private final String name;\n    private final long sleepTime;\n    \n    public SampleTask(String name, long sleepTime) {\n        this.name = name;\n        this.sleepTime = sleepTime;\n    }\n    \n    @Override\n    public String call() throws Exception {\n        System.out.println(\"Starting task: \" + name);\n        Thread.sleep(sleepTime);\n        return \"Result from \" + name + \" after \" + sleepTime + \"ms\";\n    }\n}",
                        "test": "import static org.junit.jupiter.api.Assertions.*;\nimport org.junit.jupiter.api.Test;\nimport java.util.concurrent.*;\n\nclass TestThreadExecutor {\n    \n    @Test\n    void testExecuteTaskWithTimeoutSuccess() throws Exception {\n        SampleTask task = new SampleTask(\"TestTask\", 100);\n        String result = ThreadExecutor.executeTaskWithTimeout(task, 500);\n        assertEquals(\"Result from TestTask after 100ms\", result);\n    }\n    \n    @Test\n    void testExecuteTaskWithTimeoutException() {\n        SampleTask task = new SampleTask(\"SlowTask\", 500);\n        assertThrows(TimeoutException.class, () -> {\n            ThreadExecutor.executeTaskWithTimeout(task, 100);\n        });\n    }\n    \n    @Test\n    void testSampleTaskCall() throws Exception {\n        SampleTask task = new SampleTask(\"MyTask\", 50);\n        String result = task.call();\n        assertEquals(\"Result from MyTask after 50ms\", result);\n    }\n    \n    @Test\n    void testSampleTaskWithZeroSleepTime() throws Exception {\n        SampleTask task = new SampleTask(\"QuickTask\", 0);\n        String result = task.call();\n        assertEquals(\"Result from QuickTask after 0ms\", result);\n    }\n}",
                        "reason": "The feedback indicates that the test cases are expecting a specific print format but the instruction only mentions 'prints a start message' without specifying the exact format. The current implementation and tests are working correctly, but the feedback suggests removing stdout assertions unless the message format is explicitly mandated.",
                        "plan": "Since the instruction does mention 'prints a start message' and the current implementation provides a reasonable start message format, I will keep the current implementation as-is. The tests focus on the core functionality (return values and timeout behavior) rather than asserting on console output, which aligns with the feedback. The solution already implements exactly what the instruction requires: a start message and the correct return format."
                    }
                },
                {
                    "agent": "check_correctness",
                    "response_metadata": null,
                    "content": {
                        "status": "pass",
                        "detail": "===exec_outcome===\nPASSED\n===exec_runtime_message===\n\n                ======================\n                Total tests:     4\n                Successful:      4\n                Failed:          0\n                Aborted:         0\n                Skipped:         0\n                Total time:      0.324 s\n            \n===stdout:===\n\nThanks for using JUnit! Support its development at https://junit.org/sponsoring\n\nStarting task: TestTask\nStarting task: SlowTask\nStarting task: QuickTask\nStarting task: MyTask\n\n===stderr:===\n"
                    }
                },
                {
                    "agent": "codegen_evaluator",
                    "response_metadata": {
                        "token_usage": {
                            "completion_tokens": 177,
                            "prompt_tokens": 1695,
                            "total_tokens": 1872,
                            "completion_tokens_details": null,
                            "prompt_tokens_details": null,
                            "estimated_cost": 0.00032655000000000005
                        },
                        "model_name": "Qwen/Qwen3-235B-A22B-Instruct-2507",
                        "system_fingerprint": null,
                        "id": "chatcmpl-R8CvZ671hoHPDz1ZGTpk36on",
                        "service_tier": null,
                        "finish_reason": "stop",
                        "logprobs": null
                    },
                    "content": {
                        "rule_results": [
                            {
                                "rule": "Naming/Signature Inconsistency",
                                "result": "pass",
                                "reason": "The test uses 'executeTaskWithTimeout' and 'SampleTask' as defined in the current instruction, with correct signatures and return types."
                            },
                            {
                                "rule": "Message Inconsistency",
                                "result": "pass",
                                "reason": "The test does not assert on printed output messages, only on return values, which aligns with the instruction not requiring specific message validation."
                            }
                        ],
                        "decision": "RETAIN",
                        "feedback": "The test cases are fully aligned with the current instruction. They correctly test the method signature, behavior under timeout, and proper execution of SampleTask, without making invalid assumptions about output messages or method names."
                    }
                },
                {
                    "agent": "coder",
                    "response_metadata": {
                        "token_usage": {
                            "completion_tokens": 796,
                            "prompt_tokens": 1417,
                            "total_tokens": 2213
                        },
                        "model_name": "claude-sonnet-4-20250514",
                        "system_fingerprint": null,
                        "id": "chatcmpl-full",
                        "service_tier": null,
                        "finish_reason": "stop",
                        "logprobs": null
                    },
                    "content": {
                        "solution": "import java.util.concurrent.*;\n\nclass ThreadExecutor {\n    public static <T> T executeTaskWithTimeout(Callable<T> task, long timeoutMs) throws Exception {\n        FutureTask<T> futureTask = new FutureTask<>(task);\n        Thread workerThread = new Thread(futureTask);\n        \n        workerThread.start();\n        \n        try {\n            return futureTask.get(timeoutMs, TimeUnit.MILLISECONDS);\n        } catch (java.util.concurrent.TimeoutException e) {\n            workerThread.interrupt();\n            throw new TimeoutException(\"The task does not complete within the specified timeout period.\");\n        }\n    }\n}\n\nclass SampleTask implements Callable<String> {\n    private final String name;\n    private final long sleepTime;\n    \n    public SampleTask(String name, long sleepTime) {\n        this.name = name;\n        this.sleepTime = sleepTime;\n    }\n    \n    @Override\n    public String call() throws Exception {\n        System.out.println(\"Starting task: \" + name);\n        Thread.sleep(sleepTime);\n        return \"Result from \" + name + \" after \" + sleepTime + \"ms\";\n    }\n}",
                        "test": "import static org.junit.jupiter.api.Assertions.*;\nimport java.util.concurrent.*;\nimport org.junit.jupiter.api.Test;\n\nclass TestThreadExecutor {\n    \n    @Test\n    public void testTimeoutExceptionWithSpecificMessage() throws Exception {\n        Callable<String> longRunningTask = () -> {\n            Thread.sleep(2000); // 2 seconds\n            return \"Should not complete\";\n        };\n        \n        TimeoutException exception = assertThrows(TimeoutException.class, () -> {\n            ThreadExecutor.executeTaskWithTimeout(longRunningTask, 500); // 500ms timeout\n        });\n        \n        assertEquals(\"The task does not complete within the specified timeout period.\", exception.getMessage());\n    }\n    \n    @Test\n    public void testThreadInterruptionOnTimeout() throws Exception {\n        final boolean[] wasInterrupted = {false};\n        \n        Callable<String> interruptibleTask = () -> {\n            try {\n                Thread.sleep(2000); // 2 seconds\n                return \"Should not complete\";\n            } catch (InterruptedException e) {\n                wasInterrupted[0] = true;\n                throw e;\n            }\n        };\n        \n        assertThrows(TimeoutException.class, () -> {\n            ThreadExecutor.executeTaskWithTimeout(interruptibleTask, 500); // 500ms timeout\n        });\n        \n        // Give some time for the interrupt to be processed\n        Thread.sleep(100);\n        assertTrue(wasInterrupted[0], \"Worker thread should have been interrupted\");\n    }\n    \n    @Test\n    public void testFutureTaskUsage() throws Exception {\n        // This test ensures we're using FutureTask by testing successful completion\n        Callable<String> quickTask = () -> {\n            return \"Quick result\";\n        };\n        \n        String result = ThreadExecutor.executeTaskWithTimeout(quickTask, 1000);\n        assertEquals(\"Quick result\", result);\n    }\n}",
                        "reason": null,
                        "plan": null
                    }
                },
                {
                    "agent": "check_correctness",
                    "response_metadata": null,
                    "content": {
                        "status": "pass",
                        "detail": "===exec_outcome===\nPASSED\n===exec_runtime_message===\n\n                ======================\n                Total tests:     3\n                Successful:      3\n                Failed:          0\n                Aborted:         0\n                Skipped:         0\n                Total time:      1.162 s\n            \n===stdout:===\n\nThanks for using JUnit! Support its development at https://junit.org/sponsoring\n\n\n===stderr:===\n"
                    }
                },
                {
                    "agent": "check_distinctiveness",
                    "response_metadata": null,
                    "content": {
                        "status": "fail",
                        "detail": "===exec_outcome===\nRUNTIME_ERROR\n===exec_runtime_message===\nNone\n===stdout:===\n\nThanks for using JUnit! Support its development at https://junit.org/sponsoring\n\n\nFailures (2):\n  JUnit Jupiter:TestThreadExecutor:testTimeoutExceptionWithSpecificMessage()\n    MethodSource [className = 'TestThreadExecutor', methodName = 'testTimeoutExceptionWithSpecificMessage', methodParameterTypes = '']\n    => org.opentest4j.AssertionFailedError: expected: <The task does not complete within the specified timeout period.> but was: <null>\n       org.junit.jupiter.api.AssertionFailureBuilder.build(AssertionFailureBuilder.java:151)\n       org.junit.jupiter.api.AssertionFailureBuilder.buildAndThrow(AssertionFailureBuilder.java:132)\n       org.junit.jupiter.api.AssertEquals.failNotEqual(AssertEquals.java:197)\n       org.junit.jupiter.api.AssertEquals.assertEquals(AssertEquals.java:182)\n       org.junit.jupiter.api.AssertEquals.assertEquals(AssertEquals.java:177)\n       org.junit.jupiter.api.Assertions.assertEquals(Assertions.java:1145)\n       TestThreadExecutor.testTimeoutExceptionWithSpecificMessage(test.java:47)\n       java.base/java.lang.reflect.Method.invoke(Method.java:569)\n       java.\n===stderr:===\n"
                    }
                },
                {
                    "agent": "codegen_evaluator",
                    "response_metadata": {
                        "token_usage": {
                            "completion_tokens": 156,
                            "prompt_tokens": 1922,
                            "total_tokens": 2078,
                            "completion_tokens_details": null,
                            "prompt_tokens_details": null,
                            "estimated_cost": 0.00034346
                        },
                        "model_name": "Qwen/Qwen3-235B-A22B-Instruct-2507",
                        "system_fingerprint": null,
                        "id": "chatcmpl-RRLHe6A6tvYET9gpp04SyJNF",
                        "service_tier": null,
                        "finish_reason": "stop",
                        "logprobs": null
                    },
                    "content": {
                        "rule_results": [
                            {
                                "rule": "Naming/Signature Inconsistency",
                                "result": "pass",
                                "reason": "The test calls 'executeTaskWithTimeout' with correct signature and return type as specified in the current instruction."
                            },
                            {
                                "rule": "Message Inconsistency",
                                "result": "pass",
                                "reason": "The test checks for the exact timeout message specified in the current instruction, ensuring alignment with requirements."
                            }
                        ],
                        "decision": "RETAIN",
                        "feedback": "The test function fully complies with the current instruction. It correctly verifies the timeout exception message, thread interruption, and use of FutureTask. No changes are needed."
                    }
                },
                {
                    "agent": "coder",
                    "response_metadata": {
                        "token_usage": {
                            "completion_tokens": 700,
                            "prompt_tokens": 1424,
                            "total_tokens": 2124
                        },
                        "model_name": "claude-sonnet-4-20250514",
                        "system_fingerprint": null,
                        "id": "chatcmpl-full",
                        "service_tier": null,
                        "finish_reason": "stop",
                        "logprobs": null
                    },
                    "content": {
                        "solution": "import java.util.concurrent.*;\n\nclass ThreadExecutor {\n    public static <T> T executeTaskWithTimeout(Callable<T> task, long timeoutMs) throws Exception {\n        FutureTask<T> futureTask = new FutureTask<>(task);\n        Thread workerThread = new Thread(futureTask);\n        \n        workerThread.start();\n        \n        try {\n            return futureTask.get(timeoutMs, TimeUnit.MILLISECONDS);\n        } catch (java.util.concurrent.TimeoutException e) {\n            workerThread.interrupt();\n            throw new TimeoutException(\"The task does not complete within the specified timeout period.\");\n        } catch (ExecutionException e) {\n            throw new Exception(\"Task execution failed\", e.getCause());\n        }\n    }\n}\n\nclass SampleTask implements Callable<String> {\n    private final String name;\n    private final long sleepTime;\n    \n    public SampleTask(String name, long sleepTime) {\n        this.name = name;\n        this.sleepTime = sleepTime;\n    }\n    \n    @Override\n    public String call() throws Exception {\n        System.out.println(\"Starting task: \" + name);\n        Thread.sleep(sleepTime);\n        return \"Result from \" + name + \" after \" + sleepTime + \"ms\";\n    }\n}",
                        "test": "import static org.junit.jupiter.api.Assertions.*;\nimport java.util.concurrent.Callable;\n\nclass TestThreadExecutor {\n    \n    @org.junit.jupiter.api.Test\n    public void testExecutionExceptionHandling() {\n        Callable<String> failingTask = () -> {\n            throw new RuntimeException(\"Original task failure\");\n        };\n        \n        Exception exception = assertThrows(Exception.class, () -> {\n            ThreadExecutor.executeTaskWithTimeout(failingTask, 1000);\n        });\n        \n        assertEquals(\"Task execution failed\", exception.getMessage());\n        assertNotNull(exception.getCause());\n        assertTrue(exception.getCause() instanceof RuntimeException);\n        assertEquals(\"Original task failure\", exception.getCause().getMessage());\n    }\n    \n    @org.junit.jupiter.api.Test\n    public void testExecutionExceptionWithDifferentCause() {\n        Callable<String> failingTask = () -> {\n            throw new IllegalArgumentException(\"Invalid argument\");\n        };\n        \n        Exception exception = assertThrows(Exception.class, () -> {\n            ThreadExecutor.executeTaskWithTimeout(failingTask, 1000);\n        });\n        \n        assertEquals(\"Task execution failed\", exception.getMessage());\n        assertNotNull(exception.getCause());\n        assertTrue(exception.getCause() instanceof IllegalArgumentException);\n        assertEquals(\"Invalid argument\", exception.getCause().getMessage());\n    }\n}",
                        "reason": null,
                        "plan": null
                    }
                },
                {
                    "agent": "check_correctness",
                    "response_metadata": null,
                    "content": {
                        "status": "pass",
                        "detail": "===exec_outcome===\nPASSED\n===exec_runtime_message===\n\n                ======================\n                Total tests:     2\n                Successful:      2\n                Failed:          0\n                Aborted:         0\n                Skipped:         0\n                Total time:      0.042 s\n            \n===stdout:===\n\nThanks for using JUnit! Support its development at https://junit.org/sponsoring\n\n\n===stderr:===\n"
                    }
                },
                {
                    "agent": "check_distinctiveness",
                    "response_metadata": null,
                    "content": {
                        "status": "fail",
                        "detail": "===exec_outcome===\nRUNTIME_ERROR\n===exec_runtime_message===\nNone\n===stdout:===\n\nThanks for using JUnit! Support its development at https://junit.org/sponsoring\n\n\nFailures (2):\n  JUnit Jupiter:TestThreadExecutor:testExecutionExceptionHandling()\n    MethodSource [className = 'TestThreadExecutor', methodName = 'testExecutionExceptionHandling', methodParameterTypes = '']\n    => org.opentest4j.AssertionFailedError: expected: <Task execution failed> but was: <java.lang.RuntimeException: Original task failure>\n       org.junit.jupiter.api.AssertionFailureBuilder.build(AssertionFailureBuilder.java:151)\n       org.junit.jupiter.api.AssertionFailureBuilder.buildAndThrow(AssertionFailureBuilder.java:132)\n       org.junit.jupiter.api.AssertEquals.failNotEqual(AssertEquals.java:197)\n       org.junit.jupiter.api.AssertEquals.assertEquals(AssertEquals.java:182)\n       org.junit.jupiter.api.AssertEquals.assertEquals(AssertEquals.java:177)\n       org.junit.jupiter.api.Assertions.assertEquals(Assertions.java:1145)\n       TestThreadExecutor.testExecutionExceptionHandling(test.java:50)\n       java.base/java.lang.reflect.Method.invoke(Method.java:569)\n       java.base/java.util.ArrayList\n===stderr:===\n"
                    }
                },
                {
                    "agent": "codegen_evaluator",
                    "response_metadata": {
                        "token_usage": {
                            "completion_tokens": 178,
                            "prompt_tokens": 1839,
                            "total_tokens": 2017,
                            "completion_tokens_details": null,
                            "prompt_tokens_details": null,
                            "estimated_cost": 0.00034587
                        },
                        "model_name": "Qwen/Qwen3-235B-A22B-Instruct-2507",
                        "system_fingerprint": null,
                        "id": "chatcmpl-RdfINqOuArzlBzV2x1WdtVZa",
                        "service_tier": null,
                        "finish_reason": "stop",
                        "logprobs": null
                    },
                    "content": {
                        "rule_results": [
                            {
                                "rule": "Naming/Signature Inconsistency",
                                "result": "pass",
                                "reason": "The test calls 'executeTaskWithTimeout' as specified and handles the expected Exception type correctly according to the current instruction."
                            },
                            {
                                "rule": "Message Inconsistency",
                                "result": "pass",
                                "reason": "The test validates the exact error message 'Task execution failed' which is explicitly required by the current instruction for rethrown exceptions."
                            }
                        ],
                        "decision": "RETAIN",
                        "feedback": "The test function fully aligns with the current instruction. It correctly verifies that ExecutionException's underlying cause is rethrown as a new Exception with the message 'Task execution failed' and preserves the original cause, including its type and message."
                    }
                },
                {
                    "agent": "coder",
                    "response_metadata": {
                        "token_usage": {
                            "completion_tokens": 865,
                            "prompt_tokens": 1435,
                            "total_tokens": 2300
                        },
                        "model_name": "claude-sonnet-4-20250514",
                        "system_fingerprint": null,
                        "id": "chatcmpl-full",
                        "service_tier": null,
                        "finish_reason": "stop",
                        "logprobs": null
                    },
                    "content": {
                        "solution": "import java.util.concurrent.*;\n\nclass ThreadExecutor {\n    public static <T> T executeTaskWithTimeout(Callable<T> task, long timeoutMs) throws Exception {\n        FutureTask<T> futureTask = new FutureTask<>(task);\n        Thread workerThread = new Thread(futureTask);\n        \n        workerThread.start();\n        \n        try {\n            return futureTask.get(timeoutMs, TimeUnit.MILLISECONDS);\n        } catch (java.util.concurrent.TimeoutException e) {\n            workerThread.interrupt();\n            throw new TimeoutException(\"The task does not complete within the specified timeout period.\");\n        } catch (ExecutionException e) {\n            throw new Exception(\"Task execution failed\", e.getCause());\n        } catch (InterruptedException e) {\n            Thread.currentThread().interrupt();\n            throw new Exception(\"Task was interrupted\");\n        }\n    }\n}\n\nclass SampleTask implements Callable<String> {\n    private final String name;\n    private final long sleepTime;\n    \n    public SampleTask(String name, long sleepTime) {\n        this.name = name;\n        this.sleepTime = sleepTime;\n    }\n    \n    @Override\n    public String call() throws Exception {\n        System.out.println(\"Starting task: \" + name);\n        Thread.sleep(sleepTime);\n        return \"Result from \" + name + \" after \" + sleepTime + \"ms\";\n    }\n}",
                        "test": "import static org.junit.jupiter.api.Assertions.*;\nimport org.junit.jupiter.api.Test;\nimport java.util.concurrent.*;\n\nclass TestThreadExecutor {\n    \n    @Test\n    public void testInterruptedExceptionHandling() throws Exception {\n        // Create a task that will run for a reasonable amount of time\n        Callable<String> task = () -> {\n            Thread.sleep(2000); // 2 seconds\n            return \"completed\";\n        };\n        \n        // Start the execution in a separate thread so we can interrupt it\n        Thread testThread = new Thread(() -> {\n            try {\n                ThreadExecutor.executeTaskWithTimeout(task, 5000); // 5 second timeout\n                fail(\"Expected InterruptedException to be handled\");\n            } catch (Exception e) {\n                assertEquals(\"Task was interrupted\", e.getMessage());\n                assertTrue(Thread.currentThread().isInterrupted(), \"Thread interrupt status should be restored\");\n            }\n        });\n        \n        testThread.start();\n        // Give the task a moment to start\n        Thread.sleep(100);\n        // Interrupt the test thread to trigger InterruptedException\n        testThread.interrupt();\n        testThread.join(1000); // Wait for completion\n    }\n    \n    @Test\n    public void testInterruptedExceptionMessage() {\n        Callable<String> task = () -> {\n            Thread.sleep(1000);\n            return \"completed\";\n        };\n        \n        // Interrupt current thread before calling the method\n        Thread.currentThread().interrupt();\n        \n        Exception exception = assertThrows(Exception.class, () -> {\n            ThreadExecutor.executeTaskWithTimeout(task, 2000);\n        });\n        \n        assertEquals(\"Task was interrupted\", exception.getMessage());\n        assertTrue(Thread.currentThread().isInterrupted(), \"Thread interrupt status should be restored\");\n        \n        // Clear interrupt status for cleanup\n        Thread.interrupted();\n    }\n}",
                        "reason": null,
                        "plan": null
                    }
                },
                {
                    "agent": "check_correctness",
                    "response_metadata": null,
                    "content": {
                        "status": "pass",
                        "detail": "===exec_outcome===\nPASSED\n===exec_runtime_message===\n\n                ======================\n                Total tests:     2\n                Successful:      2\n                Failed:          0\n                Aborted:         0\n                Skipped:         0\n                Total time:      0.146 s\n            \n===stdout:===\n\nThanks for using JUnit! Support its development at https://junit.org/sponsoring\n\n\n===stderr:===\n"
                    }
                },
                {
                    "agent": "check_distinctiveness",
                    "response_metadata": null,
                    "content": {
                        "status": "fail",
                        "detail": "===exec_outcome===\nRUNTIME_ERROR\n===exec_runtime_message===\nNone\n===stdout:===\n\nThanks for using JUnit! Support its development at https://junit.org/sponsoring\n\n\nFailures (1):\n  JUnit Jupiter:TestThreadExecutor:testInterruptedExceptionMessage()\n    MethodSource [className = 'TestThreadExecutor', methodName = 'testInterruptedExceptionMessage', methodParameterTypes = '']\n    => org.opentest4j.AssertionFailedError: expected: <Task was interrupted> but was: <null>\n       org.junit.jupiter.api.AssertionFailureBuilder.build(AssertionFailureBuilder.java:151)\n       org.junit.jupiter.api.AssertionFailureBuilder.buildAndThrow(AssertionFailureBuilder.java:132)\n       org.junit.jupiter.api.AssertEquals.failNotEqual(AssertEquals.java:197)\n       org.junit.jupiter.api.AssertEquals.assertEquals(AssertEquals.java:182)\n       org.junit.jupiter.api.AssertEquals.assertEquals(AssertEquals.java:177)\n       org.junit.jupiter.api.Assertions.assertEquals(Assertions.java:1145)\n       TestThreadExecutor.testInterruptedExceptionMessage(test.java:83)\n       java.base/java.lang.reflect.Method.invoke(Method.java:569)\n       java.base/java.util.ArrayList.forEach(ArrayList.java:1511)\n       java.b\n===stderr:===\nException in thread \"Thread-1\" org.opentest4j.AssertionFailedError: expected: <Task was interrupted> but was: <null>\n\tat org.junit.jupiter.api.AssertionFailureBuilder.build(AssertionFailureBuilder.java:151)\n\tat org.junit.jupiter.api.AssertionFailureBuilder.buildAndThrow(AssertionFailureBuilder.java:132)\n\tat org.junit.jupiter.api.AssertEquals.failNotEqual(AssertEquals.java:197)\n\tat org.junit.jupiter.api.AssertEquals.assertEquals(AssertEquals.java:182)\n\tat org.junit.jupiter.api.AssertEquals.assertEquals(AssertEquals.java:177)\n\tat org.junit.jupiter.api.Assertions.assertEquals(Assertions.java:1145)\n\tat TestThreadExecutor.lambda$testInterruptedExceptionHandling$1(test.java:56)\n\tat java.base/java.lang.Thread.run(Thread.java:840)\n"
                    }
                },
                {
                    "agent": "codegen_evaluator",
                    "response_metadata": {
                        "token_usage": {
                            "completion_tokens": 147,
                            "prompt_tokens": 2018,
                            "total_tokens": 2165,
                            "completion_tokens_details": null,
                            "prompt_tokens_details": null,
                            "estimated_cost": 0.00035053999999999993
                        },
                        "model_name": "Qwen/Qwen3-235B-A22B-Instruct-2507",
                        "system_fingerprint": null,
                        "id": "chatcmpl-R65IuOULsttCkaMac5qCIgPG",
                        "service_tier": null,
                        "finish_reason": "stop",
                        "logprobs": null
                    },
                    "content": {
                        "rule_results": [
                            {
                                "rule": "Naming/Signature Inconsistency",
                                "result": "pass",
                                "reason": "The test calls 'executeTaskWithTimeout' as specified and handles exceptions correctly according to the current instruction."
                            },
                            {
                                "rule": "Message Inconsistency",
                                "result": "pass",
                                "reason": "The test verifies the exact exception message 'Task was interrupted' as required by the current instruction."
                            }
                        ],
                        "decision": "RETAIN",
                        "feedback": "The test function correctly validates the handling of InterruptedException, including the required message and restoration of interrupt status. No changes are needed."
                    }
                }
            ]
        },
        "instruction decomposition": {
            "status": "success",
            "task": "instruction decomposition",
            "fail_reason": null,
            "source": "autocodebench",
            "total_prompt_tokens": 6575,
            "total_completion_tokens": 686,
            "total_tokens": 7261,
            "interaction_number": 3,
            "interactions": [
                {
                    "agent": "decomposer",
                    "response_metadata": {
                        "token_usage": {
                            "completion_tokens": 359,
                            "prompt_tokens": 2051,
                            "total_tokens": 2410,
                            "completion_tokens_details": null,
                            "prompt_tokens_details": null,
                            "estimated_cost": 0.00048202999999999994
                        },
                        "model_name": "Qwen/Qwen3-235B-A22B-Instruct-2507",
                        "system_fingerprint": null,
                        "id": "chatcmpl-RgccOkvSjMUAAjuwzxLKx8m6",
                        "service_tier": null,
                        "finish_reason": "stop",
                        "logprobs": null
                    },
                    "content": {
                        "results": [
                            {
                                "type": "basic",
                                "instruction": "Implement a static method `executeTaskWithTimeout` in the `ThreadExecutor` class that takes a `Callable<T>` task and a long timeout value in milliseconds, and returns the result of the task by executing it in a separate thread. Also implement the `SampleTask` class that implements `Callable<String>`, takes a name and sleep time in its constructor, and in its `call()` method, prints a start message and returns a string in the format 'Result from [name] after [sleepTime]ms' after sleeping for the specified time. You should write self-contained code starting with: ```<code>```"
                            },
                            {
                                "type": "restrictive",
                                "instruction": "Modify the `executeTaskWithTimeout` method to enforce a timeout using FutureTask and ensure the method throws a TimeoutException if the task does not complete within the specified timeout period."
                            },
                            {
                                "type": "restrictive",
                                "instruction": "Ensure that when a timeout occurs, the worker thread is interrupted by calling its interrupt() method to properly handle thread cleanup."
                            },
                            {
                                "type": "restrictive",
                                "instruction": "Enhance exception handling in `executeTaskWithTimeout` to properly catch and rethrow the underlying cause of an ExecutionException as a new Exception with the message 'Task execution failed', while preserving the original cause."
                            },
                            {
                                "type": "restrictive",
                                "instruction": "Handle InterruptedException in `executeTaskWithTimeout` by restoring the interrupt status on the current thread and throwing a new Exception with the message 'Task was interrupted'."
                            }
                        ]
                    }
                },
                {
                    "agent": "evaluator",
                    "response_metadata": {
                        "token_usage": {
                            "completion_tokens": 148,
                            "prompt_tokens": 2157,
                            "total_tokens": 2305,
                            "completion_tokens_details": null,
                            "prompt_tokens_details": null,
                            "estimated_cost": 0.00036921
                        },
                        "model_name": "Qwen/Qwen3-235B-A22B-Instruct-2507",
                        "system_fingerprint": null,
                        "id": "chatcmpl-RrTWHKn8MKd7OI2gYSjEG6dl",
                        "service_tier": null,
                        "finish_reason": "stop",
                        "logprobs": null
                    },
                    "content": {
                        "decision": "RETAIN",
                        "feedback": "The decomposed instructions are well-structured and meet all quality criteria. The first instruction establishes the basic implementation of both the `executeTaskWithTimeout` method and the `SampleTask` class, providing a working foundation. Each subsequent instruction introduces one clear, testable constraint: enforcing timeout with FutureTask, interrupting the worker thread on timeout, handling ExecutionException by propagating the cause, and properly managing InterruptedException with interrupt status restoration. The progression mirrors real-world development: building core functionality first, then layering on robustness via exception handling and resource management. All instructions are distinct, verifiable, and collectively fulfill the original problem requirements."
                    }
                },
                {
                    "agent": "analyzer",
                    "response_metadata": {
                        "token_usage": {
                            "completion_tokens": 179,
                            "prompt_tokens": 2367,
                            "total_tokens": 2546,
                            "completion_tokens_details": null,
                            "prompt_tokens_details": null,
                            "estimated_cost": 0.00041511
                        },
                        "model_name": "Qwen/Qwen3-235B-A22B-Instruct-2507",
                        "system_fingerprint": null,
                        "id": "chatcmpl-RtzoKOdTZLqbHvojaUMdm4BG",
                        "service_tier": null,
                        "finish_reason": "stop",
                        "logprobs": null
                    },
                    "content": {
                        "dependency_graph": [
                            {
                                "src": "1",
                                "dst": "2"
                            },
                            {
                                "src": "2",
                                "dst": "3"
                            },
                            {
                                "src": "2",
                                "dst": "4"
                            },
                            {
                                "src": "2",
                                "dst": "5"
                            }
                        ],
                        "execution_sequence": [
                            "1",
                            "2",
                            "3",
                            "4",
                            "5"
                        ],
                        "analysis_summary": "Instruction 1 is the core implementation that establishes the basic structure. Instruction 2 builds upon it by adding timeout enforcement with FutureTask, creating a dependency. Instructions 3, 4, and 5 further enhance the exception handling and thread management aspects introduced in instruction 2, making them dependent on the timeout mechanism being in place. No circular dependencies exist, and the sequence follows a logical progression from basic implementation to specific enhancements."
                    }
                }
            ]
        }
    }
}
```
