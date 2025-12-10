import streamlit as st
from openai import OpenAI

# Use Streamlit Secrets (Streamlit Cloud only)
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="AI Story Generator", page_icon="📚", layout="centered")

st.title("📚 AI Story Generator")
st.write("Create custom stories using AI! Adjust the settings and generate a unique story.")

# USER INPUTS
title = st.text_input("Story Title", "The Lost Kingdom")
main_character = st.text_input("Main Character Name", "Aria")
genre = st.selectbox("Genre", ["Fantasy", "Sci-Fi", "Horror", "Romance", "Adventure", "Mystery"])
tone = st.selectbox("Tone", ["Lighthearted", "Serious", "Dark", "Funny", "Epic"])
length = st.slider("Story Length (words)", 100, 2000, 500)

if st.button("Generate Story"):
    with st.spinner("Creating your story..."):

        prompt = f"""
        Write a story titled '{title}'. 
        Main character: {main_character}. 
        Genre: {genre}. 
        Tone: {tone}. 
        Length: about {length} words.
        Make it engaging and creative.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=length * 2
        )

        story = response.choices[0].message.content
        st.subheader("Your Generated Story:")
        st.write(story)
