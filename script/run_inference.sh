llm_providers=("deepinfra" "deepinfra") 
llms=("openai/gpt-oss-120b" "Qwen/Qwen3-Coder-480B-A35B-Instruct")
# base golden
context_settings=("base" "golden")
# fh, ce, ci
prompt_settings=("fh")
# For gpt-oss series with thinking mode, use `oss-think`
# For qwen3 series with thinking mode, use `qwen3-think`
think_flags=("oss-think" "None")

# --lang: python, java, java-repo
for i in "${!llm_providers[@]}"; do
  for context in "${context_settings[@]}"; do
    for prompt in "${prompt_settings[@]}"; do
      echo "Running inference: llm_provider=${llm_providers[$i]}, llm=${llms[$i]}, context_setting=$context, prompt_setting=$prompt"
      PYTHONPATH=. python -m evaluation.eval \
        --command inference \
        --lang java-repo \
        --llm_provider "${llm_providers[$i]}" \
        --llm "${llms[$i]}" \
        --context_setting "$context" \
        --prompt_setting "$prompt" \
        --think_flag "${think_flags[$i]}" 
    done
  done
done