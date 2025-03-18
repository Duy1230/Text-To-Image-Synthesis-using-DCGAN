import torch
import streamlit as st
from sentence_transformers import SentenceTransformer
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from model import generator

# Set page title
st.set_page_config(page_title="Text to Image Generator")

# Initialize models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model = SentenceTransformer("all-mpnet-base-v2").to(device)


def encode_captions(caption):
    return torch.tensor(bert_model.encode(caption))


# Create the main interface
st.title("Text to Image Generator")
st.write("Enter your text description below and see the generated image!")

# Text input
text_input = st.text_input("Enter your description:",
                           "a beautiful sunset over the ocean")

# Generate button
if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        # Encode the text
        text_embedding = encode_captions(text_input).unsqueeze(0).to(device)

        # Generate random noise
        noise = torch.randn(1, 100).to(device)

        # Generate the image
        with torch.no_grad():
            generated_img = generator(noise, text_embedding)

        # Convert the tensor to image
        generated_img = generated_img.squeeze(0).cpu()
        # Convert from [-1,1] to [0,1] range
        generated_img = (generated_img + 1) / 2
        generated_img = generated_img.clamp(0, 1)
        # Convert to PIL Image
        generated_img = transforms.ToPILImage()(generated_img)

        # Display the generated image
        st.image(generated_img, caption="Generated Image",
                 use_column_width=True)
