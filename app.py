import streamlit as st
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# LangSmith Tracking 
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_PROJECT"] = "Q&A chatbot Tracking"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user's queries."),
        ("user", "Question: {question}")
    ]
)

def generate_response(question, api_key, model_name, temperature, max_tokens):
    if not api_key:
        return "Please enter your API key in the sidebar."

    llm = ChatOpenAI(
        model=model_name,  
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
        max_tokens=max_tokens,
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question})

st.title("Enhanced Q&A chatbot")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your API key", type="password")

model_name = st.sidebar.selectbox(
    "Select a model",
    ["openai/gpt-4o", "openai/gpt-4-turbo", "openai/gpt-4.1-mini", "openai/gpt-4"]
)

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
max_tokens = st.sidebar.slider("Max tokens", 50, 300, 150, 10)
st.sidebar.markdown(
    "🔑 Get your OpenRouter API key here: "
    "[OpenRouter API Keys](https://openrouter.ai/settings/keys)"
)

st.write("Ask any question:")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, api_key, model_name, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide the query.")
