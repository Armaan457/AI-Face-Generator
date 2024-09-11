import streamlit as st
import torch
from diffusers import StableDiffusionPipeline

@st.cache_resource
def load_model():
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    try:
        pipe.to('cuda')
    except:
        pipe.to('cpu') 
    return pipe

pipe = load_model()

# Streamlit app 
st.title("AI-Generated Face Based on Characteristics")

# Input fields 
st.subheader("Define the characteristics of the face you want to generate:")
gender = st.selectbox("Select gender", ["Male", "Female"])
age_group = st.selectbox("Select age group", ["Child", "Teenager", "Adult", "Elderly"])
hair_color = st.selectbox("Select hair color", ["Black", "Brown", "Blonde", "Red", "Gray", "No"])
expression = st.selectbox("Select facial expression", ["Happy", "Sad", "Angry", "Neutral", "Surprised"])
nationality = st.text_input("Country of birth (e.g., India, America, France):", "")
additional_traits = st.text_input("Add any additional traits (e.g., beard, glasses, freckles):", "")

# Slider to control resolution and inference steps
st.sidebar.subheader("Advanced Settings")
resolution = st.sidebar.slider("Image resolution (Lower is faster but compromises quality)", 256, 768, 512)  
steps = st.sidebar.slider("Number of inference steps (Lower is faster but compromises quality)", 10, 50, 20) 

# Build the text prompt
def build_prompt(gender, age_group, hair_color, expression, nationality, additional_traits):
    prompt = f"A {age_group.lower()} {gender.lower()} human from {nationality.lower()} with {hair_color.lower()} hair, looking {expression.lower()}"
    if additional_traits:
        prompt += f" having {additional_traits.lower()}."
    else:
        prompt += "."
    #print(prompt)
    return prompt

# Generate face 
if st.button("Generate Face"):
    prompt = build_prompt(gender, age_group, hair_color, expression, nationality, additional_traits)    
    with st.spinner("Generating face..."):
        image = pipe(prompt, num_inference_steps=steps, height=resolution, width=resolution).images[0]
        st.image(image, caption="Generated Face", use_column_width=True)

