from vllm import LLM, SamplingParams

llm = LLM(
    model="/home/lidong1/qilongma/blob/public_models/Meta-Llama-2-7B-hf",
    trust_remote_code=True,
    tensor_parallel_size=4,
    gpu_memory_utilization=0.9,
)
sampling_params = SamplingParams(
    temperature=0.0,
    top_p=0.95,
    max_tokens=64,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None
)

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# outputs = []
# for prompt in prompts:
#     outputs.append(llm.generate([prompt], sampling_params)) aaa