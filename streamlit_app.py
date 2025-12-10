import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

st.set_page_config(page_title="AI Story Generator", page_icon="📚", layout="centered")
st.title("📚 AI Story Generator (BLOOM)")
st.write("Create custom stories using BLOOM! Adjust the settings and generate a unique story.")

# --- Load BLOOM model once ---
@st.cache_resource(show_spinner=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
    model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")
    return tokenizer, model

tokenizer, model = load_model()

# --- User inputs ---
title = st.text_input("Story Title", "The Lost Kingdom")
main_character = st.text_input("Main Character Name", "Aria")
genre = st.selectbox("Genre", ["Fantasy", "Sci-Fi", "Horror", "Romance", "Adventure", "Mystery"])
tone = st.selectbox("Tone", ["Lighthearted", "Serious", "Dark", "Funny", "Epic"])
length = st.slider("Story Length (words)", 50, 500, 200)

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

    st.subheader("Your Generated Story:")
    st.write(story)

        story = response.choices[0].message.content
        st.subheader("Your Generated Story:")
        st.write(story)
