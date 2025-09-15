# Execution environment setup in ICE-Bench

Researchers need to configure the execution environment for each sub-dataset separately. For details, please refer to the `README.md` in each subfolder.

We recommend that users configure the sandbox with the following port settings, so that there is no need to modify the code.
- python_func:  `8199`
- java_func: `7887`

If you need to make modifications, it's also very simple; please refer to the `data_construction/service/sandbox_client.py` file.  