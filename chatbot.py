from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# Create the chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to the user queries."),
        ("user","Question:{question}")
    ]
)

# Get the llm
model = OllamaLLM(model="llama2")

# Output
output_parser = StrOutputParser()

# Streamlit framework
st.title("Langchain demo with Ollama LLM Llama 2")
input_text = st.text_input("What would you like to search")

# Chains
chains = prompt | model | output_parser

if input_text:
    st.write(chains.invoke({'question':input_text}))