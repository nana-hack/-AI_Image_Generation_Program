# Import necessary libraries: torch for tensor operations and diffusers for Stable Diffusion pipeline
import torch
from diffusers import StableDiffusion3Pipeline

# Specify the model to be used, in this case, Stable Diffusion 3.5 large version from Stability AI
large_model = "stabilityai/stable-diffusion-3.5-large"

# Load the Stable Diffusion model with the specified precision (bfloat16) for optimized memory usage
pipe = StableDiffusion3Pipeline.from_pretrained(large_model, torch_dtype=torch.bfloat16)

# Enable attention slicing to reduce memory usage and allow larger image generation on limited GPU memory
pipe.enable_attention_slicing()

# Move the pipeline to GPU for faster inference; use "cpu" instead of "cuda" if an NVIDIA GPU is not available
pipe = pipe.to("cuda")

# Define a prompt for generating an image
prompt = "An astronaut relaxing on Mars, gazing at Earth in the starry night sky."

# Generate the image using the pipeline with specified settings
results = pipe(
    prompt,
    num_inference_steps=20,     # Number of steps for refining the image
    guidance_scale=3.5,         # Strength of prompt guidance to control creativity
    height=512,                 # Height of the generated image
    width=512                   # Width of the generated image
)

# Retrieve the generated image from the results
images = results.images

# Save each generated image to a file
for i, img in enumerate(images):
    img.save(f"image_{i}.png")  # Save each image as "image_0.png", "image_1.png", etc.

