#Research paper : https://arxiv.org/pdf/2502.01584
import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
from datetime import datetime

def initialize_deepseek_client():
    try:
        # Test connection to local LM Studio server
        response = requests.get("http://localhost:1234/v1/models")
        if response.status_code == 200:
            return True
        return False
    except:
        st.error("Could not connect to LM Studio server. Please ensure it's running.")
        return False

def get_model_response(puzzle, system_prompt, max_tokens=8000):
    try:
        response = requests.post(
            "http://localhost:1234/v1/chat/completions",
            json={
                "model": "deepseek-r1-distill-llama-8b",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": puzzle}
                ],
                "max_tokens": max_tokens,
                "temperature": 0.2,
                "top_p": 0.95
            }
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            st.error(f"Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error getting model response: {e}")
        return None

def analyze_response(response, ground_truth):
    if not response or not ground_truth:
        return False
    response = response.lower().strip()
    ground_truth = ground_truth.lower().strip()
    phrases = ground_truth.split(';')
    return all(phrase.strip() in response for phrase in phrases)

# Test cases from the paper
test_cases = {
    "Continent Puzzle": {
        "category": "General Knowledge",
        "challenge": "Think of a well-known category with exactly 7 things in it. Alphabetize the things from their ending letters, and the last letter alphabetically will be E. What is it?",
        "ground_truth": "Continents",
        "difficulty": "Medium",
        "reasoning_required": "Category identification, alphabetical analysis"
    },
    "NONUNION Puzzle": {
        "category": "Word Play",
        "challenge": "The word NONUNION has four N's and no other consonant. What famous American of the past - first and last names, 8 letters in all - has four instances of the same consonant and no other consonant?",
        "ground_truth": "Eli Lilly",
        "difficulty": "Hard",
        "reasoning_required": "Pattern matching, historical knowledge"
    }
    # Add other test cases here
}

# Streamlit UI
st.set_page_config(page_title="Deepseek R1 Reasoning Benchmark", layout="wide")

st.title("Deepseek R1 Reasoning Benchmark")
st.markdown("""
This application tests Deepseek R1's reasoning capabilities using the NPR Sunday Puzzle Challenge benchmark.
The benchmark focuses on puzzles that:
- 🧩 Require general knowledge and reasoning
- ✅ Have verifiable solutions
- 🤔 Test multi-step thinking
""")

# Sidebar configuration
with st.sidebar:
    st.header("Test Configuration")
    
    st.subheader("Model Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 
                          help="Lower values make the output more focused")
    max_tokens = st.slider("Max Tokens", 1000, 8000, 4000,
                          help="Maximum number of tokens in the response")
    
    st.subheader("System Prompt")
    system_prompt = st.text_area(
        "Customize system prompt",
        "You are an expert puzzle solver. Break down the problem step by step and explain your reasoning clearly.",
        help="Sets the context for the model's response"
    )
    
    st.subheader("Model Information")
    st.write("Model: deepseek-r1-distill-llama-8b")
    st.write("Endpoint: http://localhost:1234/v1/chat/completions")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Puzzle Selection")
    
    input_type = st.radio(
        "Choose input type:",
        ["Sample Puzzle", "Custom Puzzle"],
        help="Select a pre-defined puzzle or create your own"
    )
    
    if input_type == "Sample Puzzle":
        selected_puzzle = st.selectbox(
            "Select a puzzle:",
            list(test_cases.keys())
        )
        
        st.info("📝 Challenge")
        st.write(test_cases[selected_puzzle]["challenge"])
        
        st.success("🎯 Required Reasoning")
        st.write(test_cases[selected_puzzle]["reasoning_required"])
        
        puzzle = test_cases[selected_puzzle]["challenge"]
        ground_truth = test_cases[selected_puzzle]["ground_truth"]
        
    else:
        puzzle = st.text_area(
            "Enter your puzzle:",
            height=100,
            help="Write a clear, unambiguous puzzle"
        )
        ground_truth = st.text_input(
            "Enter the correct answer:",
            help="The answer should be verifiable"
        )

with col2:
    st.subheader("Model Response")
    
    if st.button("Test Model"):
        if initialize_deepseek_client():
            with st.spinner("Getting model response..."):
                start_time = datetime.now()
                response = get_model_response(puzzle, system_prompt, max_tokens)
                end_time = datetime.now()
                
                if response:
                    st.markdown("### Model's Reasoning")
                    st.write(response)
                    
                    is_correct = analyze_response(response, ground_truth)
                    
                    st.markdown("### Analysis")
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        status = "✅ Correct" if is_correct else "❌ Incorrect"
                        st.metric("Result", status)
                    with col2:
                        response_time = (end_time - start_time).total_seconds()
                        st.metric("Response Time", f"{response_time:.2f}s")
                    with col3:
                        token_count = len(response.split())
                        st.metric("Output Tokens", token_count)
                    
                    st.markdown("### Ground Truth")
                    st.info(f"Correct Answer: {ground_truth}")

# Footer
st.markdown("---")
st.markdown("*Based on the research paper: PhD Knowledge Not Required: A Reasoning Challenge for Large Language Models*")
