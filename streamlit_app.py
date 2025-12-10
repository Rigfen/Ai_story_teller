import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
from datetime import datetime
import os

st.set_page_config(page_title="AI Story Generator", page_icon="📚", layout="centered")
st.title("📚 AI Story Generator (BLOOM-1B1 Robust)")
st.write("Generate stories live as BLOOM-1B1 writes them! Stories are saved and downloadable.")

# --- File to save stories ---
SAVE_FILE = "generated_stories.csv"

# --- Load BLOOM-1B1 model once ---
@st.cache_resource(show_spinner=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b1")
    model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-1b1")
    return tokenizer, model

tokenizer, model = load_model()

# --- Function to safely save a story ---
def save_story(entry):
    try:
        if os.path.exists(SAVE_FILE):
            df = pd.read_csv(SAVE_FILE)
            df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
        else:
            df = pd.DataFrame([entry])
        df.to_csv(SAVE_FILE, index=False)
    except Exception as e:
        st.error(f"Error saving story: {e}")

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
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        max_new_tokens = length * 2

        # Generate in chunks to show streaming effect
        for i in range(0, max_new_tokens, 20):
            outputs = model.generate(
                input_ids,
                max_new_tokens=i+20,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id
            )
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = decoded
            placeholder.markdown(f"**Your Generated Story:**\n\n{generated_text}")

    st.success("Story generation complete!")

    # --- Save story ---
    story_entry = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Title": title,
        "Main Character": main_character,
        "Genre": genre,
        "Tone": tone,
        "Story": generated_text
    }
    save_story(story_entry)
    st.info(f"Story saved to {SAVE_FILE}")

# --- Show previous stories ---
if st.checkbox("Show Previous Stories"):
    if os.path.exists(SAVE_FILE):
        try:
            df = pd.read_csv(SAVE_FILE)
            st.dataframe(df)
        except Exception as e:
            st.error(f"Cannot read CSV: {e}")
    else:
        st.write("No stories generated yet.")

# --- Download CSV ---
if os.path.exists(SAVE_FILE):
    with open(SAVE_FILE, "rb") as f:
        st.download_button(
            label="Download All Stories as CSV",
            data=f,
            file_name="generated_stories.csv",
            mime="text/csv"
        )
