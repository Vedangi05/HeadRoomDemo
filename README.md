# HeadRoom Demo: LLM Answer With vs Without Headroom

An interactive Streamlit application demonstrating prompt context compression using [Headroom AI](https://headroom.ai). The app compares model response quality, latency, and real end-to-end token consumption when sending raw context versus Headroom-compressed context to LLMs via an OpenAI-compatible API (such as the University of Florida AI Navigator / LiteLLM proxy).

---

## Features

- **Context Compression via Headroom**: Uses `headroom-ai` to compress redundant, bulky, and structured context (logs, JSON, docs).
- **Side-by-Side Comparison**: Compares prompt tokens, completion tokens, latency, and generated answers.
- **Dynamic Model Selection**: Automatically detects and lists models permitted on your API key.
- **Configurable Savings Profiles**: Switch between aggressive (`agent-90` targeting ~90% savings) and `balanced` (~70% target) profiles.

---

## Prerequisites

- **Python 3.10+** (Python 3.12 recommended)
- **Git**
- An active API key for UF AI Navigator (or any OpenAI-compatible endpoint)

---

## Setup & Running the Project

### 1. Clone the Repository
```bash
git clone https://github.com/Vedangi05/HeadRoomDemo.git
cd HeadRoomDemo
```

### 2. Create and Activate a Virtual Environment

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (Command Prompt / PowerShell):**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy the example environment file:
```bash
cp .env.example .env
```

Open `.env` in your text editor and provide your credentials:
```env
# UF AI Navigator / LiteLLM Proxy Configuration
OPENAI_API_KEY=your_actual_api_key_here
OPENAI_BASE_URL=https://api.navigator.ai.ufl.edu
OPENAI_MODEL=llama-3.3-70b-instruct
```

> **Note on Models:** If using UF Navigator, ensure `OPENAI_MODEL` is set to an authorized model for your team key (e.g., `llama-3.3-70b-instruct`, `mistral-small-3.1`).

### 5. Run the Streamlit Application
```bash
streamlit run app.py
```

Open your browser and navigate to:
```
http://localhost:8501
```

---

## Quick One-Liner (macOS / Linux)

If your environment is already set up:
```bash
source .venv/bin/activate && streamlit run app.py
```

---

## Project Structure

```text
HeadRoomDemo/
├── .env                # Local environment secrets (ignored by git)
├── .env.example        # Environment variable template
├── app.py              # Main Streamlit application
├── requirements.txt    # Project Python dependencies
└── README.md           # Documentation and run instructions
```