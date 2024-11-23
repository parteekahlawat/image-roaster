import streamlit as st
from PIL import Image
# Use a pipeline as a high-level helper
from transformers import pipeline
from dotenv import find_dotenv, load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import getpass
import os
import time
from google.api_core.exceptions import ResourceExhausted
from langchain_core.prompts import PromptTemplate


load_dotenv(find_dotenv())

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "API_KEY"
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

st.title("Image to ROASTER")
uploaded_file = st.file_uploader("Upload an image file for the story", type=["png", "jpg", "jpeg"])
# uploaded_file = 1
if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Load the image using PIL
    image = Image.open(uploaded_file)

    # image to text
    def convert_img_to_text(url):
        pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

        text = pipe(url)
        print(text)
        converted_text = text[0]["generated_text"]
        return converted_text
    text_got = convert_img_to_text(image)

    # text to generate story
    # def generate_story_from_text(text):

    def safe_invoke(llm, query):
        try:
            template = '''
convert the summary of a photo to a roast line: {content}

give me the savage, burn, spicy roast lines (one for each)
the format should be
Savage:
Burn:
Spicy:
'''
            prompt = PromptTemplate(template = template, input_variables = ["query"])
            formatted_prompt = prompt.format(content=query)  # Format the prompt with the query

            story = llm.invoke(formatted_prompt)
            return story.content
            # return llm.invoke(prompt)
        except ResourceExhausted as e:
            st.write(f"Rate limit exceeded: {e}. Retrying after 60 seconds...")
            print(f"Rate limit exceeded: {e}. Retrying after 60 seconds...")
            time.sleep(15)
            return safe_invoke(llm, query) 
        
    res = safe_invoke(llm, text_got)
    print(res)
    st.write(res)