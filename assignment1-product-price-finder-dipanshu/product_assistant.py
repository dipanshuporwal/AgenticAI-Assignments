import streamlit as st
from dotenv import load_dotenv
from typing import Optional
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

# Load Groq API key
load_dotenv()


# Product Model
class Product(BaseModel):
    product_id: Optional[str] = Field(
        default=None,
        description="Unique idenitfier for the product(product number)",
    )
    product_name: Optional[str] = Field(
        default=None, description="The name of the product"
    )
    description: Optional[str] = Field(
        default=None,
        description="Brief description or key features of the product",
    )
    tentative_price_in_usd: Optional[str] = Field(
        default=None, description="Price of the product in USD"
    )
    category: Optional[str] = Field(
        default=None,
        description="Product category such as electronics, clothing, etc.",
    )
    rating: Optional[float] = Field(
        default=None,
        ge=0,
        le=5,
        description="Average customer rating (0 to 5)",
    )


# Prompt template with system and human messages
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a helpful assistant with deep domain knowledge in"
                " product analysis and pricing. When the user gives about any"
                " product details, provide\n1. The **Product Name** \n2. The"
                " **Tentative Product Price in USD**.\n\nOnly return valid and"
                " structured information."
            ),
        ),
        ("human", "{input}"),
    ]
)

# Streamlit UI
st.set_page_config(page_title="🛍️ Product Assistant", page_icon="🛒")
st.title("🛍️ Product Price Finder Assistant")

st.markdown(
    "Welcome to the **Product Assistant App** powered by **Groq LLMs +"
    " LangChain** 🎯"
)

# Column layout
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 🤖 Select LLM Model")
    model_options = [
        "deepseek-r1-distill-llama-70b",
        "qwen-qwq-32b",
        "llama-3.1-8b-instant",
        "groq-llama-65b-v1",
        "groq-llama-70b-v2",
        "groq-qwen-14b",
        "groq-llama-2-70b-chat",
        "groq-llama-13b-chat",
    ]
    model_choice = st.selectbox("Choose a Groq-hosted model:", model_options)

with col2:
    st.markdown("### ✏️ Enter Product Description")
    product_input = st.text_area(
        "Describe the product below:",
        placeholder=(
            "e.g., A lightweight wireless headphone with noise cancellation"
        ),
    )

# Run prediction
if st.button("🎯 Get Product Info"):
    if product_input and model_choice:
        with st.spinner("Thinking... nn mn nkm6unhjn  🤖"):
            model = ChatGroq(model=model_choice)
            structured_output = model.with_structured_output(Product)
            chain = prompt | structured_output

            try:
                result = chain.invoke({"input": product_input})
                st.success("✅ Prediction Successful!")
                st.markdown(f"### 🧾 Result")
                st.write(f"**Product Name:** {result.product_name}")
                st.write(
                    f"**Estimated Price:** ${result.tentative_price_in_usd}"
                )
            except Exception as e:
                st.error(f"❌ Error: {e}")
    else:
        st.warning("⚠️ Please provide both product description and model.")
