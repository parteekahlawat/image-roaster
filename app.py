import streamlit as st
from PIL import Image
# Use a pipeline as a high-level helper
from dotenv import find_dotenv, load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import getpass
import os
import time
from google.api_core.exceptions import ResourceExhausted
from langchain_core.prompts import PromptTemplate
import requests
import tempfile

load_dotenv(find_dotenv())

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "API_KEY"
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

st.title("GET ROASTED")
uploaded_file = st.file_uploader("Upload an image file for the story", type=["png", "jpg", "jpeg"])
# uploaded_file = 1
if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Load the image using PIL
    image = Image.open(uploaded_file)

    # image to text
    def convert_img_to_text(url):
        # pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

        # text = pipe(url)
        # print(text)
        API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
        h_token = os.getenv("HUGGINGFACE_TOKEN")
        headers = {"Authorization": f"Bearer {h_token}"}
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file_path = temp_file.name
            image.save(temp_file_path, format="JPEG")

        try:
            # Read the saved file for the API
            with open(temp_file_path, "rb") as f:
                data = f.read()
            response = requests.post(API_URL, headers=headers, data=data)

            # Validate response
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and "generated_text" in result[0]:
                    return result[0]["generated_text"]
                else:
                    raise ValueError(f"Unexpected API response format: {result}")
            else:
                raise RuntimeError(f"Failed to get a response: {response.status_code}, {response.text}")
        finally:
            # Ensure the temporary file is deleted
            os.remove(temp_file_path)

        # converted_text = text[0]["generated_text"]
        # return converted_text
    text_got = convert_img_to_text(image)
    print(text_got)
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