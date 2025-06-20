import streamlit as st
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
import requests
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
import re

# Load environment
load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
EXCHANGE_RATE_API_KEY = os.getenv("EXCHANGE_RATE_API_KEY")
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")

# Define TripInfo Schema
class TripInfo(BaseModel):
    city: str
    start_date: str
    end_date: str
    currency: str

# Define shared state
class TravelState(TypedDict):
    city: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    weather: Optional[str]
    attractions: Optional[list]
    hotel_cost: Optional[float]
    exchange_rate: Optional[float]
    total_cost: Optional[float]
    itinerary: Optional[str]
    summary: Optional[str]
    user_query: Optional[str]

# Nodes
def extract_info_with_ai(state: TravelState) -> TravelState:
    query = state.get("user_query", "")
    prompt = ChatPromptTemplate.from_template("""
    You are a travel assistant. Extract the city, start date, end date, and preferred currency from this user query.
    Return only a JSON object like this:
    {{  
    "city": "Tokyo",
    "start_date": "2025-07-15",
    "end_date": "2025-07-20",
    "currency": "USD"
    }}

    Query: "{input}"
    """)
    model = ChatGroq(model="deepseek-r1-distill-llama-70b")
    structured_model = model.with_structured_output(TripInfo)
    chain = prompt | structured_model

    try:
        result = chain.invoke({"input": query})
        return {
            **state,
            "city": result.city,
            "start_date": result.start_date,
            "end_date": result.end_date,
            "currency": result.currency.upper()
        }
    except Exception as e:
        st.error(f"❌ AI extraction failed: {e}")
        return state

def ask_missing_info(state: TravelState) -> TravelState:
    if not state.get("city"):
        state["city"] = st.session_state.get("city_manual")
    if not state.get("start_date"):
        state["start_date"] = st.session_state.get("start_date_manual")
    if not state.get("end_date"):
        state["end_date"] = st.session_state.get("end_date_manual")
    return state

def fetch_weather(state: TravelState) -> TravelState:
    city = state['city']
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()

    if "list" in data:
        forecasts = data["list"][:5]
        forecast_text = "\n".join([f"{f['dt_txt']}: {f['main']['temp']}°C, {f['weather'][0]['description']}" for f in forecasts])
        state["weather"] = forecast_text
    else:
        state["weather"] = "Weather data unavailable."
    return state

def fetch_attractions(state: TravelState) -> TravelState:
    city = state['city']
    url = f"https://maps.googleapis.com/maps/api/place/textsearch/json?query=top+places+to+visit+in+{city}&key={GOOGLE_PLACES_API_KEY}"
    resp = requests.get(url).json()
    places = [result["name"] for result in resp.get("results", [])[:5]]
    state["attractions"] = places or ["No attractions found."]
    return state

def estimate_hotel_cost(state: TravelState) -> TravelState:
    state["hotel_cost"] = 100 * 5
    return state

def convert_currency(state: TravelState) -> TravelState:
    base_currency = state.get("currency", "USD").upper()
    url = f"https://v6.exchangerate-api.com/v6/{EXCHANGE_RATE_API_KEY}/latest/{base_currency}"
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
        rate = data["conversion_rates"]["INR"]
        state["exchange_rate"] = rate
        state["total_cost"] = state["hotel_cost"] * rate
    except Exception as e:
        st.error(f"❌ Currency conversion failed: {e}")
        state["exchange_rate"] = None
    return state

def generate_itinerary(state: TravelState) -> TravelState:
    attractions = state.get("attractions", [])
    itinerary_lines = [f"Day {i+1}: Visit {place}" for i, place in enumerate(attractions)]
    state["itinerary"] = "\n".join(itinerary_lines) or "Itinerary could not be generated."
    return state

def generate_summary(state: TravelState) -> TravelState:
    prompt = ChatPromptTemplate.from_template("{input}")
    model = ChatGroq(model="deepseek-r1-distill-llama-70b")
    chain = prompt | model

    input_summary = f"""
    A user is planning a trip to {state.get("city")} from {state.get("start_date")} to {state.get("end_date")}.
    Weather forecast: {state.get("weather")}.
    Top attractions: {', '.join(state.get('attractions') or [])}.
    Hotel cost: ${state.get('hotel_cost')}.
    Exchange rate: {state.get('exchange_rate')} INR per USD.
    Total cost: ₹{state.get('total_cost', 0.0):,.2f}.
    Itinerary:
    {state.get("itinerary")}

    Write a short, friendly summary of this trip.
    Only return the final summary. Do not include any reasoning, internal thoughts, or explanations.
    Limit the output to 100-120 words.
    """
    try:
        llm_response = chain.invoke({"input": input_summary})
        raw_output = llm_response.content.strip()
        # Remove <think>...</think> tags
        cleaned_output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()
        state["summary"] = cleaned_output
    except Exception as e:
        state["summary"] = "Summary generation failed."
    return state

# Define the LangGraph
workflow = StateGraph(TravelState)
workflow.add_node("extract_info_with_ai", extract_info_with_ai)
workflow.add_node("ask_missing_info", ask_missing_info)
workflow.add_node("fetch_weather", fetch_weather)
workflow.add_node("fetch_attractions", fetch_attractions)
workflow.add_node("estimate_hotel_cost", estimate_hotel_cost)
workflow.add_node("convert_currency", convert_currency)
workflow.add_node("generate_itinerary", generate_itinerary)
workflow.add_node("generate_summary", generate_summary)

workflow.set_entry_point("extract_info_with_ai")
workflow.add_edge("extract_info_with_ai", "ask_missing_info")
workflow.add_edge("ask_missing_info", "fetch_weather")
workflow.add_edge("fetch_weather", "fetch_attractions")
workflow.add_edge("fetch_attractions", "estimate_hotel_cost")
workflow.add_edge("estimate_hotel_cost", "convert_currency")
workflow.add_edge("convert_currency", "generate_itinerary")
workflow.add_edge("generate_itinerary", "generate_summary")
workflow.add_edge("generate_summary", END)

app = workflow.compile()

# Streamlit UI
st.set_page_config(page_title="✈️ Travel Planner AI", layout="centered")
st.title("✈️ AI-Powered Travel Planner")

with st.form("query_form"):
    user_query = st.text_area("Enter your travel request (in natural language)", "")
    city_manual = st.text_input("City (if not extracted)")
    start_date_manual = st.text_input("Start Date (YYYY-MM-DD)")
    end_date_manual = st.text_input("End Date (YYYY-MM-DD)")
    submitted = st.form_submit_button("Plan My Trip")

if submitted:
    st.session_state["city_manual"] = city_manual
    st.session_state["start_date_manual"] = start_date_manual
    st.session_state["end_date_manual"] = end_date_manual

    initial_state: TravelState = {
        "city": None,
        "start_date": None,
        "end_date": None,
        "weather": None,
        "attractions": None,
        "hotel_cost": None,
        "exchange_rate": None,
        "total_cost": None,
        "itinerary": None,
        "summary": None,
        "user_query": user_query,
    }

    final_state = app.invoke(initial_state)

    st.subheader("📍 Trip Summary")
    st.markdown(final_state["summary"])

    with st.expander("📅 Full Itinerary"):
        st.text(final_state["itinerary"])

    with st.expander("🌤️ Weather Forecast"):
        st.text(final_state["weather"])

    with st.expander("📍 Top Attractions"):
        st.write(final_state["attractions"])

    with st.expander("💰 Cost Estimate"):
        st.write(f"Hotel Cost (USD): ${final_state['hotel_cost']}")
        st.write(f"Exchange Rate: {final_state['exchange_rate']} INR/USD")
        st.write(f"Total Estimated Cost: ₹{final_state['total_cost']:,.2f}")
