from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
import requests
import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

# Load environment
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
EXCHANGE_RATE_API_KEY = os.getenv("EXCHANGE_RATE_API_KEY")
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")

class TripInfo(BaseModel):
    city: str
    start_date: str
    end_date: str
    currency: str

# 1. Shared State
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

# 2. AI extraction node
def extract_info_with_ai(state: TravelState) -> TravelState:
    query = state.get("user_query") or ""
    print("🤖 Extracting city and dates using AI...")

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
        print("🧠 Extracted:", result)
        return {
            **state,
            "city": result.city,
            "start_date": result.start_date,
            "end_date": result.end_date,
            "currency": result.currency.upper()
        }

    except Exception as e:
        print(f"❌ AI extraction failed: {e}")
        return state

# 3. Fallback prompt
def ask_missing_info(state: TravelState) -> TravelState:
    if not state.get("city"):
        state["city"] = input("Which city are you planning to visit? ")
    if not state.get("start_date"):
        state["start_date"] = input("What is your start date? (YYYY-MM-DD): ")
    if not state.get("end_date"):
        state["end_date"] = input("What is your end date? (YYYY-MM-DD): ")
    return state

# 4. Other Nodes
def fetch_weather(state: TravelState) -> TravelState:
    city = state['city']
    print(f"🌦️ Fetching real weather for {city}...")

    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()

    if "list" in data:
        forecasts = data["list"][:5]
        forecast_text = "\n".join([f"{f['dt_txt']}: {f['main']['temp']}°C, {f['weather'][0]['description']}" for f in forecasts])
        state["weather"] = forecast_text
    else:
        state["weather"] = "Weather data unavailable."

    print(f"🌦️ Weather in {city}: {state['weather']}")
    return state

def fetch_attractions(state: TravelState) -> TravelState:
    city = state['city']
    print(f"📍 Getting real places for {city}...")

    url = f"https://maps.googleapis.com/maps/api/place/textsearch/json?query=top+places+to+visit+in+{city}&key={GOOGLE_PLACES_API_KEY}"
    resp = requests.get(url).json()
    
    places = [result["name"] for result in resp.get("results", [])[:5]]
    state["attractions"] = places or ["No attractions found."]
    print(f"📍 Places in {city}: {places}")
    return state

def estimate_hotel_cost(state: TravelState) -> TravelState:
    print("🏨 Estimating hotel cost...")
    state["hotel_cost"] = 100 * 5
    print(f"🏨 Hotel cost: {state['hotel_cost']}")
    return state

def convert_currency(state: TravelState) -> TravelState:
    print("💱 Getting real exchange rate...")
    base_currency = state.get("currency", "USD").upper()
    url = f"https://v6.exchangerate-api.com/v6/{EXCHANGE_RATE_API_KEY}/latest/{base_currency}"

    try:
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
        rate = data["conversion_rates"]["INR"]
    except Exception as e:
        print(f"❌ Currency conversion failed: {e}")
        state["exchange_rate"] = None
        return state

    state["exchange_rate"] = rate
    state["total_cost"] = state["hotel_cost"] * rate
    print(f"💱 Total Cost in INR: ₹{state['total_cost']}")
    return state

def generate_itinerary(state: TravelState) -> TravelState:
    print("📅 Generating itinerary...")

    attractions = state.get("attractions", [])
    itinerary_lines = []

    for i, place in enumerate(attractions, 1):
        itinerary_lines.append(f"Day {i}: Visit {place}")

    itinerary_text = "\n".join(itinerary_lines) if itinerary_lines else "Itinerary could not be generated due to missing attraction data."

    state["itinerary"] = itinerary_text
    print(f"📅 Generated Itinerary: {state['itinerary']}")
    return state

def generate_summary(state: TravelState) -> TravelState:
    print("📝 Generating summary using LLM...")

    city = state.get("city", "N/A")
    start_date = state.get("start_date", "N/A")
    end_date = state.get("end_date", "N/A")
    weather = state.get("weather", "N/A")
    attractions = state.get("attractions", [])
    hotel_cost = state.get("hotel_cost", 0.0)
    exchange_rate = state.get("exchange_rate", None)
    total_cost = state.get("total_cost", 0.0)
    itinerary = state.get("itinerary", "No itinerary available.")

    input_summary = f"""
    A user is planning a trip to {city} from {start_date} to {end_date}.
    The weather forecast is: {weather}.
    The top 5 attractions include: {', '.join(attractions)}.
    The estimated hotel cost is ${hotel_cost:.2f}.
    The current exchange rate is {exchange_rate} INR per USD.
    The estimated total cost is ₹{total_cost:,.2f}.
    The itinerary is:
    {itinerary}

    Write a short, friendly and informative summary of this trip.
    Only return the final summary. Do not include any reasoning, internal thoughts, or explanations.
    Limit the output to 100-120 words.
    """

    # LLM setup
    prompt = ChatPromptTemplate.from_template("{input}")
    model = ChatGroq(model="deepseek-r1-distill-llama-70b")
    chain = prompt | model

    try:
        llm_response = chain.invoke({"input": input_summary})
        raw_output = llm_response.content.strip()
        # Remove <think>...</think> tags
        summary_text = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()
        state["summary"] = summary_text
        print(f"📝 Generated Summary (LLM):\n{summary_text}")
    except Exception as e:
        print(f"❌ LLM summary generation failed: {e}")
        state["summary"] = "Summary generation failed. Please try again."

    return state


# 5. Define Graph
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

# 6. Compile and run
app = workflow.compile()

# Accept user query in any language
user_query = input("Enter your travel request (any language, any format): ")

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
    "user_query": user_query
}

# Run
final_state = app.invoke(initial_state)

print("\n✅ Final Travel Plan Summary:\n")
print(final_state["summary"])