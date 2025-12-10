import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

st.set_page_config(page_title="AI Story Generator", page_icon="📚", layout="centered")
st.title("📚 AI Story Generator (BLOOM-1B1 Streaming)")
st.write("Generate stories live as BLOOM-1B1 writes them! Adjust the settings and see output as it comes.")

# --- Load BLOOM-1B1 model once ---
@st.cache_resource(show_spinner=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b1")
    model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-1b1")
    return tokenizer, model

tokenizer, model = load_model()

# --- User inputs ---
title = st.text_input("Story Title", "The Lost Kingdom")
main_character = st.text_input("Main Character Name", "Aria")
genre = st.selectbox("Genre", ["Fantasy", "Sci-Fi", "Horror", "Romance", "Adventure", "Mystery"])
tone = st.selectbox("Tone", ["Lighthearted", "Serious", "Dark", "Funny", "Epic"])
length = st.slider("Story Length (words)", 50, 800, 300)

# --- Generate story ---
if st.button("Generate Story"):
    prompt = f"""
    Write a story titled '{title}'. 
    Main character: {main_character}. 
    Genre: {genre}. 
    Tone: {tone}. 
    Length: about {length} words.
    Make it engaging and creative.
    """

    placeholder = st.empty()
    generated_text = ""

    with st.spinner("Generating story..."):
        # Encode prompt
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        max_new_tokens = length * 2

        # Generate with streaming
        for i in range(0, max_new_tokens, 20):  # generate 20 tokens at a time
            outputs = model.generate(
                input_ids,
                max_new_tokens=i+20,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id
            )
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Only keep new content
            generated_text = decoded
            placeholder.markdown(f"**Your Generated Story:**\n\n{generated_text}")
            # Streamlit will refresh the placeholder each chunk

    st.success("Story generation complete!")
