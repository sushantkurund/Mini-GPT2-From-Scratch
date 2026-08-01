from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
import sys
import torch

# ---------------------------------------
# Project Paths
# ---------------------------------------

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.dataset import ShakespeareDataset
from src.model import GPTLanguageModel

# ---------------------------------------
# FastAPI App
# ---------------------------------------

app = FastAPI(
    title="Mini GPT-2 From Scratch",
    description="Character-level GPT built from scratch using PyTorch",
    version="1.0"
)

# Allow React frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

# ---------------------------------------
# Load Tokenizer & Dataset
# ---------------------------------------

dataset = ShakespeareDataset(
    file_path=os.path.join(ROOT_DIR, "data", "tinyshakespeare.txt"),
    block_size=128
)

# ---------------------------------------
# Load Model
# ---------------------------------------

model = GPTLanguageModel(
    vocab_size=dataset.vocab_size,
    block_size=128,
    n_embd=128,
    n_head=4,
    n_layer=4,
    dropout=0.2
).to(device)

checkpoint_path = os.path.join(ROOT_DIR, "checkpoints", "model.pth")

model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location=device
    )
)

model.eval()

print("✅ GPT Model Loaded Successfully!")

# ---------------------------------------
# Request Schema
# ---------------------------------------

class GenerateRequest(BaseModel):
    prompt: str = ""
    max_tokens: int = Field(default=300, ge=50, le=1000)
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)

# ---------------------------------------
# Home Route
# ---------------------------------------

@app.get("/")
def home():
    return {
        "status": "running",
        "message": "Mini GPT-2 API"
    }

# ---------------------------------------
# Generate Route
# ---------------------------------------

@app.post("/generate")
def generate(request: GenerateRequest):

    try:

        if request.prompt.strip() == "":
            context = torch.zeros(
                (1, 1),
                dtype=torch.long,
                device=device
            )
        else:

            encoded = dataset.tokenizer.encode(request.prompt)

            context = torch.tensor(
                [encoded],
                dtype=torch.long,
                device=device
            )

        with torch.no_grad():

            generated = model.generate(
                context,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature
            )

        output = dataset.tokenizer.decode(
            generated[0].tolist()
        )

        return {
            "generated_text": output
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )