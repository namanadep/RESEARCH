import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import pandas as pd
import plotly.express as px
from datetime import datetime

# Load environment variables
load_dotenv()

def initialize_openai_client():
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        st.error("OpenAI API Key not found in .env file")
        return None
    return OpenAI(api_key=api_key)

def get_model_response(client, puzzle, system_prompt, max_tokens=8000):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": puzzle}
            ],
            max_tokens=max_tokens,
            temperature=0.2,
            top_p=0.95
        )
        return response.choices[0].message.content
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

# Enhanced test cases from the paper
test_cases = {
    "Continent Puzzle": {
        "category": "General Knowledge",
        "challenge": "Think of a well-known category with exactly 7 things in it. Alphabetize the things from their ending letters, and the last letter alphabetically will be E. In other words, no thing in this category ends in a letter after E in the alphabet. It's a category and set of 7 things that everyone knows. What is it?",
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
    },
    "Wild Wild West Movie": {
        "category": "Entertainment",
        "challenge": "The film Wild Wild West had three W's as its initials. What prominent film of 2013 had two W's as its initials?",
        "ground_truth": "The Wolf Of Wall Street",
        "difficulty": "Medium",
        "reasoning_required": "Movie knowledge, pattern recognition"
    },
    "Hidden Planets": {
        "category": "Word Play",
        "challenge": "These four words have a very interesting and unusual property in common - something hidden in them. What is it? NEANDERTHAL EMBARRASS SATURATION CONTEMPTUOUSNESS",
        "ground_truth": "Each word conceals the name of a planet in left to right order (Earth, Mars, Saturn, Neptune)",
        "difficulty": "Hard",
        "reasoning_required": "Pattern recognition, astronomy knowledge"
    },
    "Syllable Change": {
        "category": "Word Play",
        "challenge": "Think of a familiar five-letter word in two syllables. Change the middle letter to the preceding letter of the alphabet, and you'll get a familiar five-letter word in three syllables. What words are these?",
        "ground_truth": "penne and penné",
        "difficulty": "Hard",
        "reasoning_required": "Vocabulary, phonetics"
    },
    "Letter I Challenge": {
        "category": "Word Play",
        "challenge": "Think of an 8-letter word with three syllables that contains the letter 'I' in each syllable but, strangely, doesn't contain a single 'I' sound, either long or short. The answer is not a plural. What word is it?",
        "ground_truth": "Daiquiri",
        "difficulty": "Hard",
        "reasoning_required": "Phonetics, vocabulary"
    },
    "City Letters": {
        "category": "Geography",
        "challenge": "The city UTICA, NEW YORK contains 12 letters, all different. Name a well-known U.S. city that contains 13 different letters when spelled out.",
        "ground_truth": "Casper, Wyoming; Big Flats, New York; Bucksport, Maine; Lynchburg, Texas",
        "difficulty": "Hard",
        "reasoning_required": "Geography knowledge, letter counting"
    },
    "Food Word Transform": {
        "category": "Word Play",
        "challenge": "Name a food item in seven letters. Move the first letter to the fifth position and you'll get two words that are synonyms. What are they?",
        "ground_truth": "Brisket --> risk, bet",
        "difficulty": "Hard",
        "reasoning_required": "Vocabulary, word manipulation"
    }
}

# Streamlit UI
st.set_page_config(page_title="GPT-4o-mini Reasoning Benchmark", layout="wide")

st.markdown("#### PhD Knowledge Not Required: A Reasoning Challenge for Large Language Models")
st.markdown("""
This application tests GPT-4o-mini's reasoning capabilities using the NPR Sunday Puzzle Challenge benchmark. 
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
    
    st.subheader("Filters")
    difficulty = st.multiselect(
        "Difficulty Level",
        ["Easy", "Medium", "Hard"],
        default=["Medium", "Hard"]
    )
    
    category = st.multiselect(
        "Category",
        ["General Knowledge", "Word Play", "Geography", "Entertainment"],
        default=["General Knowledge", "Word Play"]
    )

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
        # Filter puzzles based on sidebar selections
        filtered_puzzles = {k: v for k, v in test_cases.items() 
                          if v['difficulty'] in difficulty 
                          and v['category'] in category}
        
        selected_puzzle = st.selectbox(
            "Select a puzzle:",
            list(filtered_puzzles.keys())
        )
        
        # Display puzzle details
        st.info("📝 Challenge")
        st.write(filtered_puzzles[selected_puzzle]["challenge"])
        
        st.success("🎯 Required Reasoning")
        st.write(filtered_puzzles[selected_puzzle]["reasoning_required"])
        
        puzzle = filtered_puzzles[selected_puzzle]["challenge"]
        ground_truth = filtered_puzzles[selected_puzzle]["ground_truth"]
        
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
    
    if st.button("Test Model", help="Run the model on the selected puzzle"):
        client = initialize_openai_client()
        if client and puzzle:
            with st.spinner("Getting model response..."):
                start_time = datetime.now()
                response = get_model_response(client, puzzle, system_prompt, max_tokens)
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
