import streamlit as st
import torch

from src.dataset import ShakespeareDataset
from src.model import GPTLanguageModel

# ------------------------
# Page Config
# ------------------------

st.set_page_config(
    page_title="Mini GPT-2 From Scratch",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Mini GPT-2 From Scratch")
st.markdown(
    "Built using **PyTorch + Transformer Decoder + Streamlit**"
)

# ------------------------
# Device
# ------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# Model Config
# ------------------------

block_size = 128

n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.2

# ------------------------
# Load Dataset
# ------------------------

@st.cache_resource
def load_everything():

    dataset = ShakespeareDataset(
        "data/tinyshakespeare.txt",
        block_size=block_size
    )

    model = GPTLanguageModel(
        vocab_size=dataset.vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(
        torch.load(
            "checkpoints/model.pth",
            map_location=device,
        )
    )

    model.eval()

    return dataset, model


dataset, model = load_everything()

st.success("✅ Model Loaded Successfully")

# ------------------------
# Sidebar
# ------------------------

st.sidebar.header("Generation Settings")

temperature = st.sidebar.slider(
    "Temperature",
    0.2,
    2.0,
    0.8,
    0.1,
)

max_tokens = st.sidebar.slider(
    "Maximum Tokens",
    50,
    600,
    300,
)

# ------------------------
# Prompt
# ------------------------

prompt = st.text_area(
    "Enter Prompt",
    value="KING",
    height=150,
)

# ------------------------
# Generate
# ------------------------

if st.button("Generate Text"):

    with st.spinner("Generating..."):

        if prompt.strip() == "":

            context = torch.zeros(
                (1, 1),
                dtype=torch.long,
                device=device,
            )

        else:

            encoded = dataset.tokenizer.encode(prompt)

            context = torch.tensor(
                [encoded],
                dtype=torch.long,
                device=device,
            )

        with torch.no_grad():

            generated = model.generate(
                context,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )

        output = dataset.tokenizer.decode(
            generated[0].tolist()
        )

    st.subheader("Generated Text")

    st.code(output)
