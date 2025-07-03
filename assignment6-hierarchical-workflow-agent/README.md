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

git clone https://github.com/yourusername/hierarchical-workflow-agent.git
cd hierarchical-workflow-agent

### 2. Set Up a Virtual Environment (Optional but Recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

### 3. Install Dependencies
pip install -r requirements.txt

### 4. Set Up Environment Variables
GOOGLE_API_KEY=your_google_genai_api_key

### 5. 🧪 Running the App
python main.py


## Example query:
1. What are the latest treatments for type 2 diabetes?
2. Can you analyze how Tesla's stock has performed over the past 6 months and provide a summary of key financial events that may have influenced it?
