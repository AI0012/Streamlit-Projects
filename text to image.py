import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

# Stable Diffusion
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Streamlit app
st.title("Image Generation Application")
prompt = st.text_input("Enter the prompt", "a photo of an astronaut riding a horse on mars")

if st.button('Generate Image'):
    image = pipe(prompt).images[0]  
    st.image(image, caption='Generated Image', use_column_width=True)
#image.save("astronaut_rides_horse.png")
