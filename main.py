from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

MODEL_PATH = "./jarvis_light_trained"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained("anupammehar640/jarvis-light")
tokenizer = AutoTokenizer.from_pretrained("anupammehar640/jarvis-light")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print("Model loaded successfully!")

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(request: ChatRequest):
    input_text = request.message + tokenizer.eos_token

    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove input text from output
    reply = response[len(request.message):].strip()

    return {"response": reply}
