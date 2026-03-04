# 📊 Autonomous Data Analyst
**Moving Beyond Context Stuffing with Reinforcement Learning(GRPO) & Agentic Tool Use**

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Model](https://img.shields.io/badge/Model-14B_Local-green)

## 💡 Overview
Current approaches to analyzing tabular data with LLMs rely on "Context Stuffing" (feeding the entire CSV into the prompt) or standard RAG. These methods fail at enterprise scale: they crash on large files (e.g., 100MB+), cost too much, and suffer from "lost-in-the-middle" hallucination. 

This project introduces an **Autonomous Data Analyst Agent** powered by a local 14B model. Instead of reading the raw data, the agent acts like a human analyst: it writes deterministic Python code to query local data, executes it in a secure sandbox, and reasons over the metadata to deliver 100% precise answers.

## 🚀 Key Features
* **Unlimited Scalability:** Processes a 1 Million row (130MB) dataset using less than 500 tokens by only reading schema metadata and execution results.
* **Privacy-First:** Data never leaves the local Python sandbox. Only code and aggregated results interact with the LLM.
* **Trained with GRPO (Reinforcement Learning):** Moved beyond standard SFT. The model is fine-tuned with dense rewards to eliminate "blind guessing" (KeyErrors) and enforce an analyst protocol (e.g., always checking `df.head()` before querying).

## 📈 Performance (TabMWP Hard Subset)
By combining SFT and RL (GRPO), the model transitions from a "probabilistic guesser" to a "grounded analyst."

| Model State | Accuracy | Behavioral Note |
| :--- | :--- | :--- |
| Base Model | 20% | Relies on mental math; high hallucination rate. |
| SFT Model | 66% | Learns syntax but lacks strategy (guesses column names). |
| **RL Agent (ART)** | **82%** | **Learns defensive coding and schema verification (>95% inspection rate).** |

## 🧠 System Architecture (The Analyst Loop)
1. **User Query:** e.g., "What was the total revenue for APAC in 2023?"
2. **LLM Reasoning:** Generates `pandas` code instead of attempting mental math.
3. **Local Sandbox:** Executes the code securely.
4. **Observation:** Returns the calculated result or error traceback.
5. **Final Answer:** Model synthesizes the final response.

## 🛠️ Quick Start
```bash
git clone [https://github.com/yourusername/autonomous-data-analyst.git](https://github.com/yourusername/autonomous-data-analyst.git)
cd autonomous-data-analyst
pip install -r requirements.txt
python train_grpo.py  # To get the GRPO model and results
python train_sft.py # To get the SFT model and results
```

## 🔮 Future Work
* **Multi-File Reasoning**: Expanding to SQL-like JOIN operations across multiple CSVs.

* **Test-Time Compute**: Enforcing a Chain-of-Thought "Planning" phase before code generation to handle complex, multi-step queries.

* **Multimodal Integration**: Enabling the agent to read PDF reports and output visual charts (matplotlib).


## 🙏 Acknowledgments
This code is base on the ART training framework. You can check out the framework and repository here: [https://github.com/OpenPipe/ART].