from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from huggingface_hub import login
import time
import torch

login("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")

model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Check if GPU is available. If available, move the model to GPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


prompt = "Write a python script to add 2 numbers"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

start_time = time.time()
output = model.generate(
    inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_length=1000,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.7
)
end_time = time.time()
time_taken = end_time - start_time
print(f"Time taken = {time_taken} seconds")

generated_text = tokenizer.decode(output[0])
print(generated_text)

print("done")
