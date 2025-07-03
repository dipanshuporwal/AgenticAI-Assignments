# Hierarchical Workflow Agent using LangGraph & Google Gemini

This project demonstrates a multi-node hierarchical agent built using [LangGraph](https://github.com/langchain-ai/langgraph) and Google Gemini (via `langchain_google_genai`). The agent simulates a real-world workflow involving **supervision**, **research**, **summarization**, and **report generation**.

## 🧠 Project Overview

The agent performs the following steps:
1. **Supervisor Node** determines the topic of the query.
2. Based on the topic, either:
   - `MedicalResearch` or
   - `FinancialResearch` node is invoked.
3. The research output is passed to a `CreateSummary` node.
4. The summary is saved as a `.docx` file via `SaveSummary`.

This is orchestrated as a stateful LangGraph with branching logic using conditional edges.

---

## 📁 Folder Structure

├── main.py # Main application file with graph logic
├── output/ # Contains saved .docx summaries
├── .env # Environment file with API key(s)
└── README.md # Project documentation


---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/hierarchical-workflow-agent.git
cd hierarchical-workflow-agent

python main.py
