import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

st.set_page_config(page_title="AI Story Generator", page_icon="📚", layout="centered")
st.title("📚 AI Story Generator (BLOOM-1B1)")
st.write("Create custom stories using BLOOM-1B1! Adjust the settings and generate a unique story.")

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
length = st.slider("Story Length (words)", 50, 800, 300)  # Increased max length for bigger model

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

    with st.spinner("Generating story..."):
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_length=length * 2,
            do_sample=True,
            top_p=0.9,
            temperature=0.8
        )
        story = tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.subheader
