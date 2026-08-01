# Mini GPT-2 From Scratch

A complete implementation of a **Mini GPT-2 inspired Decoder-Only Transformer** built entirely from scratch using **PyTorch**, trained on the **Tiny Shakespeare** dataset, deployed through a **FastAPI** backend, and served with a **React + Vite** frontend.

This project demonstrates the complete workflow of building a Transformer-based language model from the ground up, including data preprocessing, model training, autoregressive text generation, API deployment, and an interactive web interface.

---

# Project Overview

The objective of this project is to gain a deep understanding of modern Transformer architectures by implementing every major component manually instead of relying on high-level libraries.

The implementation includes:

- Character-Level Tokenization
- Embedding Layers
- Positional Embeddings
- Multi-Head Self-Attention
- Transformer Decoder Blocks
- Feed Forward Networks
- Autoregressive Text Generation
- Model Training using PyTorch
- REST API Deployment using FastAPI
- Interactive React Frontend

The trained model generates Shakespeare-style text from user prompts while demonstrating the complete end-to-end pipeline of a Transformer-based language model.

---

# Features

- GPT-2 Inspired Decoder-Only Transformer
- Character-Level Tokenizer
- Multi-Head Self-Attention
- Positional Embeddings
- Transformer Decoder Blocks
- Feed Forward Network
- Temperature-Based Sampling
- Character-Level Text Generation
- Automatic Model Checkpoint Saving
- FastAPI REST API
- Interactive Swagger Documentation
- React + Vite Frontend
- Adjustable Temperature
- Adjustable Maximum Tokens
- Copy Generated Text
- Download Generated Text
- Responsive User Interface

---

# Project Architecture

```text
                  Tiny Shakespeare Dataset
                            │
                            ▼
                  Character Tokenizer
                            │
                            ▼
                Character Embedding Layer
                            │
                            ▼
               Positional Embedding Layer
                            │
                            ▼
      Decoder-Only Transformer (GPT Style)
                            │
                            ▼
             Multi-Head Self Attention
                            │
                            ▼
                  Feed Forward Network
                            │
                            ▼
                   Language Model Head
                            │
                            ▼
                Next Character Prediction
                            │
                            ▼
                  Generated Text Output
```

---

# Project Structure

```text
Mini-GPT2-From-Scratch/

├── assets/
│   ├── home.png
│   ├── output.png
│   └── swagger.png
│
├── backend/
│   └── main.py
│
├── checkpoints/
│   └── model.pth
│
├── data/
│   └── tinyshakespeare.txt
│
├── frontend/
│   ├── public/
│   ├── src/
│   ├── package.json
│   ├── package-lock.json
│   ├── vite.config.js
│   └── index.html
│
├── notebook/
│   └── MiniGPT2.ipynb
│
├── outputs/
│   └── sample_output.txt
│
├── src/
│   ├── __init__.py
│   ├── attention.py
│   ├── dataset.py
│   ├── generate.py
│   ├── model.py
│   ├── tokenizer.py
│   ├── train.py
│   └── transformer.py
│
├── .gitignore
├── README.md
├── requirements.txt
├── test_dataset.py
└── test_tokenizer.py
```

---

# Technologies Used

| Category | Technology |
|-----------|------------|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| Backend Framework | FastAPI |
| Frontend Framework | React |
| Build Tool | Vite |
| API Communication | Axios |
| Styling | CSS |
| Dataset | Tiny Shakespeare |

---

# Model Configuration

| Property | Value |
|----------|-------|
| Model Type | GPT-2 Inspired Decoder-Only Transformer |
| Tokenization | Character-Level |
| Dataset | Tiny Shakespeare |
| Context Length | 128 |
| Embedding Dimension | 128 |
| Attention Heads | 4 |
| Transformer Layers | 4 |
| Dropout | 0.2 |
| Optimizer | AdamW |
| Best Validation Loss | 1.6918 |

---

# Application Preview

## Home Page

![Home Page](assets/home.png)

---

## Generated Output

![Generated Output](assets/output.png)

---

## Swagger API Documentation

![Swagger UI](assets/swagger.png)

---

# Getting Started

Navigate to the project directory:

```bash
cd Mini-GPT2-From-Scratch
```

---

## Create a Virtual Environment

### Windows

```bash
python -m venv env
env\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv env
source env/bin/activate
```

---

## Install Dependencies

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install frontend dependencies:

```bash
cd frontend
npm install
```
---

# Model Training

Train the model using:

```bash
python -m src.train
```

During training, the model:

- Loads the Tiny Shakespeare dataset
- Performs character-level tokenization
- Trains the decoder-only Transformer architecture
- Evaluates validation loss during training
- Saves the best-performing model checkpoint automatically

The trained model is stored in:

```text
checkpoints/model.pth
```

---

# Generate Text from Terminal

Generate text directly from the terminal:

```bash
python -m src.generate
```

Example Prompt

```text
KING
```

Example Output

```text
KING EDWARD IV:
Roman the shall law and his shall we, thou house;
And many we deservilure the maniston all the praying...
```

---

# Running the Backend

Start the FastAPI server:

```bash
uvicorn backend.main:app --reload
```

Backend URL

```
http://127.0.0.1:8000
```

Swagger API Documentation

```
http://127.0.0.1:8000/docs
```

---

# Running the Frontend

Navigate to the frontend directory:

```bash
cd frontend
```

Install the required packages:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

Open the application in your browser:

```
http://localhost:5173
```

---

# Web Application

The web interface provides the following functionality:

- Enter custom prompts
- Adjust Temperature
- Adjust Maximum Tokens
- Generate Shakespeare-style text
- Copy generated text
- Download generated text
- Clear generated output

---

# Sample Outputs

Sample generated outputs are available in:

```text
outputs/
```

These samples demonstrate the language generation capability of the trained model.

---

# Notebook

The complete implementation notebook is available in:

```text
notebook/
```

The notebook documents the complete development workflow, including preprocessing, model implementation, training, evaluation, and experimentation.

---

# Learning Outcomes

This project provided practical experience with:

- Character-Level Tokenization
- Embedding Layers
- Positional Embeddings
- Self-Attention Mechanism
- Multi-Head Self-Attention
- Transformer Decoder Architecture
- Autoregressive Language Modeling
- Temperature-Based Sampling
- Model Training using PyTorch
- FastAPI Backend Development
- REST API Design
- React Frontend Development
- Full-Stack AI Application Deployment

---

# Future Improvements

Potential enhancements include:

- Top-k Sampling
- Top-p (Nucleus) Sampling
- Beam Search Decoding
- Learning Rate Scheduling
- Mixed Precision Training
- Word-Level or Subword Tokenization
- Multilingual Dataset Training
- Larger GPT Model Configurations
- GPU Optimized Training
- Docker Deployment
- Cloud Deployment
- User Authentication and Prompt History

---

# References

- Andrej Karpathy – Neural Networks: Zero to Hero
- PyTorch Documentation
- FastAPI Documentation
- React Documentation
- Tiny Shakespeare Dataset

---

# Acknowledgements

This project was inspired by the educational work of **Andrej Karpathy** on implementing GPT-style language models from scratch.

The architecture and implementation were developed independently using PyTorch to gain a practical understanding of Transformer models, self-attention mechanisms, autoregressive language modeling, and deployment using modern web technologies.

---

# Author

**Sushant Kurund**

Master of Computer Applications (Data Science)

GitHub: https://github.com/sushantkurund

---

# License

This project is intended for educational and learning purposes. It demonstrates the implementation of a Mini GPT-2 inspired Decoder-Only Transformer architecture built entirely from scratch using PyTorch.
