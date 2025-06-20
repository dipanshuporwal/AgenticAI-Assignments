# 🌍 AI-Powered Travel Planner

This project is an **AI-powered travel assistant** that takes a natural language query (like "Plan my trip to Tokyo in July") and generates a complete travel plan — including weather, top attractions, hotel costs, currency conversion, itinerary, and a friendly summary — using a LangGraph workflow and multiple APIs.

---

## 🧠 Features

- ✨ **LLM-Powered Info Extraction** (city, dates, currency)
- 🌤️ **Live Weather Forecast** (via OpenWeather API)
- 📍 **Top Attractions** (via Google Places API)
- 🏨 **Hotel Cost Estimation**
- 💱 **Live Currency Conversion** (via ExchangeRate API)
- 📝 **Itinerary & Summary Generation** (via Groq’s LLM)
- 🔁 **LangGraph State Machine** for clean, modular execution

---

## 📦 Requirements

- Python 3.8+
- API Keys:
  - OpenAI (for structured extraction and summary)
  - OpenWeather
  - ExchangeRate
  - Google Places

---

## 🔧 Setup

1. **Clone the repository**

```bash
git clone https://github.com/your-username/travel-planner-ai.git
cd travel-planner-ai
