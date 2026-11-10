
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading Hugging Face model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="cuda"
)

print("Model loaded successfully!")
print("Using GPU:", torch.cuda.get_device_name(0))

while True:
    prompt = input("\nYou: ")

    if prompt.lower() in ["exit", "quit"]:
        print("AI session closed.")
        break

    messages = [
        {"role": "user", "content": prompt}
    ]

    # Convert prompt into input tokens
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to("cuda")

    # Generate AI response
    outputs = model.generate(
        input_ids,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

    # Extract only the generated response
    response = tokenizer.decode(
        outputs[0][input_ids.shape[-1]:],
        skip_special_tokens=True
    )

    print("\nAI:", response)
