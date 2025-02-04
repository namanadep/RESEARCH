# Reference : https://huggingface.co/krutrim-ai-labs/Vyakyarth
import gradio as gr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the pre-trained model
model = SentenceTransformer("krutrim-ai-labs/vyakyarth")

def calculate_similarity(sentence1, sentence2):
    # Encode the sentences into embeddings
    embeddings = np.array(model.encode([sentence1, sentence2]))
    # Calculate cosine similarity
    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return f"Similarity Score: {score:.4f}"

# Create Gradio interface with a contrasting color theme
with gr.Blocks(theme=gr.themes.Glass()) as demo:
    gr.Markdown("# # Multilingual Text Similarity Checker - krutrim-ai-labs/vyakyarth")
    gr.Markdown("Enter two sentences to check their similarity.")
    
    with gr.Row():
        sentence1 = gr.Textbox(label="Sentence 1", placeholder="Enter the first sentence here...", lines=2)
        sentence2 = gr.Textbox(label="Sentence 2", placeholder="Enter the second sentence here...", lines=2)
    
    submit_button = gr.Button("Check Similarity")
    
    output = gr.Textbox(label="Output", interactive=False)

    submit_button.click(calculate_similarity, inputs=[sentence1, sentence2], outputs=output)

# Launch the app
demo.launch(share=True)
