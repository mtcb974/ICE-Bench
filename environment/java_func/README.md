# Install the Sandbox environment of AutoCodeBench

Install sandbox:

```bash
docker pull hunyuansandbox/multi-language-sandbox:v1

docker run -d \
  --name sandbox-service \
  -p 7887:8080 \
  --cap-add=NET_ADMIN \
  hunyuansandbox/multi-language-sandbox:v1
```

Test service health status. If the response contains `"exec_outcome": "PASSED"` in the JSON, it indicates the service is running normally.

```bash
curl -X POST http://127.0.0.1:7887/submit \
  -H "Content-Type: application/json" \
  -d '{"src_uid": "test-001", "lang": "python", "source_code": "print(\"Hello World\")"}'
```