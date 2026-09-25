# 003: The boxed instruction is the system prompt

**Decision**: every training and evaluation prompt is
`[{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": question}]` with
`INSTRUCTION = "Please reason step by step, and put your final answer within \boxed{}."`,
built by one helper, `mc_data.schema.build_messages`, used by the parquet builder, the eval
harness, the throughput bench and the preflight. The parquet files feed both training (verl
reads `prompt`) and evaluation, so the two cannot drift.

**Why**: the brief fixes the text and says "system prompt, used identically for training and
evaluation". Qwen's model card appends the same sentence to the user turn instead; both render
through the same chat template and the tokenizer preflight checks the rendered prompt ends with
the empty `<think></think>` block either way.

**Consequence**: changing the instruction means regenerating the parquet files (`make data`);
`scripts/check_env.py --only-config` verifies the system prompt is rendered verbatim.
