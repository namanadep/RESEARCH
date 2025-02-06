import streamlit as st
from langchain.llms import Ollama
from typing import Dict, Any

# Page configuration
st.set_page_config(
    page_title="Ollama Parameter Playground",
    page_icon="🤖",
    layout="wide"
)

def create_ollama_config(params: Dict[str, Any]) -> Dict[str, Any]:
    """Create a valid Ollama configuration dictionary."""
    config = {}
    
    # Only add parameters that have valid values
    if params.get('temperature') is not None:
        config['temperature'] = params['temperature']
    if params.get('num_ctx') is not None:
        config['num_ctx'] = params['num_ctx']
    if params.get('repeat_penalty') is not None:
        config['repeat_penalty'] = params['repeat_penalty']
    if params.get('repeat_last_n') is not None:
        config['repeat_last_n'] = params['repeat_last_n']
    if params.get('top_k') is not None:
        config['top_k'] = params['top_k']
    if params.get('top_p') is not None:
        config['top_p'] = params['top_p']
    if params.get('seed') is not None and params['seed'] != -1:
        config['seed'] = params['seed']
    
    return config

# Main title and description
st.title("🤖 Ollama Parameter Playground")
st.markdown("Experiment with different parameters to see how they affect the model's responses.")

# Sidebar for model selection and core parameters
with st.sidebar:
    st.header("Model Configuration")
    
    model_name = st.selectbox(
        "Select Model",
        ["deepseek-r1:8b"],
        help="Currently using deepseek-r1:8b for demonstration"
    )
    
    st.subheader("Core Parameters")
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Controls randomness in responses"
    )
    
    context_length = st.slider(
        "Context Length (num_ctx)",
        min_value=512,
        max_value=8192,
        value=2048,
        step=512,
        help="Maximum context length for the model"
    )

# Create two columns for parameter groups
col1, col2 = st.columns(2)

# First column for repetition controls
with col1:
    st.header("Repetition Control")
    
    repeat_penalty = st.slider(
        "Repeat Penalty",
        min_value=1.0,
        max_value=2.0,
        value=1.1,
        step=0.1,
        help="Penalizes repeated tokens"
    )
    
    repeat_last_n = st.slider(
        "Repeat Last N",
        min_value=0,
        max_value=512,
        value=64,
        step=32,
        help="Window size for checking repetitions"
    )

# Second column for sampling parameters
with col2:
    st.header("Sampling Parameters")
    
    top_k = st.slider(
        "Top K",
        min_value=1,
        max_value=100,
        value=40,
        step=1,
        help="Limits the number of tokens considered"
    )
    
    top_p = st.slider(
        "Top P",
        min_value=0.0,
        max_value=1.0,
        value=0.95,
        step=0.05,
        help="Cumulative probability threshold"
    )
    
    seed = st.number_input(
        "Random Seed",
        min_value=-1,
        max_value=1000000,
        value=-1,
        help="Set for reproducible outputs. -1 for random"
    )

# Text input area
st.header("Test Your Configuration")
user_prompt = st.text_area(
    "Enter your prompt",
    value="Write a short poem about artificial intelligence",
    height=100
)

# Create Ollama configuration
if st.button("Generate Response", type="primary"):
    try:
        with st.spinner("Generating response..."):
            # Create config dictionary with only valid parameters
            config = create_ollama_config({
                'temperature': temperature,
                'num_ctx': context_length,
                'repeat_penalty': repeat_penalty,
                'repeat_last_n': repeat_last_n,
                'top_k': top_k,
                'top_p': top_p,
                'seed': seed
            })
            
            # Initialize Ollama with the configuration
            llm = Ollama(
                model=model_name,
                **config
            )
            
            # Generate response
            response = llm(user_prompt)
            
            # Display response
            st.markdown("### Generated Response")
            st.markdown(response)
            
            # Display used configuration
            with st.expander("View Configuration"):
                st.json(config)
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
