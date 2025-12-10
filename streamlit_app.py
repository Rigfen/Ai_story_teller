import streamlit as st
import pandas as pd
from datetime import datetime
import os

# --- Page Setup ---
st.set_page_config(page_title="AI Story Generator", page_icon="📚", layout="centered")
st.title("📚 AI Story Generator (BLOOM-1B1 Multi-Chapter)")
st.write("Generate multiple stories or chapters live with BLOOM-1B1! Each story is saved and downloadable.")

SAVE_FILE = "generated_stories.csv"
CACHE_DIR = "./bloom_cache"  # Local cache for model

# --- Load BLOOM-1B1 safely ---
@st.cache_resource(show_spinner=True)
def load_model():
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b1", cache_dir=CACHE_DIR)
        model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-1b1", cache_dir=CACHE_DIR)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading BLOOM-1B1 model: {e}")
        return None, None

tokenizer, model = load_model()
if tokenizer is None or model is None:
    st.stop()  # Stop the app if model didn't load

# --- Safe CSV saving function ---
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

# --- User Inputs ---
title = st.text_input("Story/Book Title", "The Lost Kingdom")
main_character = st.text_input("Main Character Name", "Aria")
genre = st.selectbox("Genre", ["Fantasy", "Sci-Fi", "Horror", "Romance", "Adventure", "Mystery"])
tone = st.selectbox("Tone", ["Lighthearted", "Serious", "Dark", "Funny", "Epic"])
length = st.slider("Story/Chapter Length (words)", 50, 800, 300)
num_chapters = st.slider("Number of Stories / Chapters", 1, 5, 1)

# --- Generate stories ---
if st.button("Generate Stories"):
    for chapter in range(1, num_chapters + 1):
        st.subheader(f"Generating Chapter {chapter}...")
        prompt = f"""
        Write story/chapter {chapter} titled '{title}'. 
        Main character: {main_character}. 
        Genre: {genre}. 
        Tone: {tone}. 
        Length: about {length} words.
        Make it engaging and creative.
        """

        placeholder = st.empty()
        generated_text = ""

        # Encode prompt
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        max_new_tokens = length * 2

        # Streaming generation in chunks
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
            placeholder.markdown(f"**Chapter {chapter}:**\n\n{generated_text}")

        st.success(f"Chapter {chapter} complete!")

        # --- Save chapter to CSV ---
        story_entry = {
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Title": title,
            "Main Character": main_character,
            "Genre": genre,
            "Tone": tone,
            "Chapter": chapter,
            "Story": generated_text
        }
        save_story(story_entry)

# --- Show previous stories ---
if st.checkbox("Show Previous Stories / Chapters"):
    if os.path.exists(SAVE_FILE):
        try:
            df = pd.read_csv(SAVE_FILE)
            st.dataframe(df)
        except Exception as e:
            st.error(f"Cannot read CSV: {e}")
    else:
        st.write("No stories generated yet.")

# --- Download all stories ---
if os.path.exists(SAVE_FILE):
    with open(SAVE_FILE, "rb") as f:
        st.download_button(
            label="Download All Stories / Chapters as CSV",
            data=f,
            file_name="generated_stories.csv",
            mime="text/csv"
        )
