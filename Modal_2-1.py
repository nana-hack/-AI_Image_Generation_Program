import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# Specify the model to be used, in this case, Stable Diffusion version 2.1 from Stability AI
small_model = "stabilityai/stable-diffusion-2-1"

# Load the Stable Diffusion model with the specified precision (bfloat16) for reduced memory usage
pipe = StableDiffusionPipeline.from_pretrained(small_model, torch_dtype=torch.bfloat16)

# Replace the default scheduler with DPMSolverMultistepScheduler for more efficient and stable results
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Enable attention slicing to save memory and allow larger batch sizes on limited GPU memory
pipe.enable_attention_slicing()

# Move the pipeline to GPU (if available) to speed up inference

pipe = pipe.to("cuda") # Change "cuda" to "cpu" if you are not using an NVIDIA graphics card

# Define a list of prompts for generating different images
prompts = [
    "a medieval knight exploring a futuristic city",
    "A surreal forest with bioluminescent trees and floating crystals at twilight.",
    "A serene beach with a giant whale in the sky, sunset colors reflecting on the water.",
    "An astronaut relaxing on Mars, gazing at Earth in the starry night sky."
]
# Generate images using the pipeline with specified inference steps, guidance scale, and image dimensions
results = pipe(
    prompts,
    num_inference_steps=50,     # Number of denoising steps to perform
    guidance_scale=3.5,         # Strength of prompt guidance
    height=512,                 # Height of the generated images
    width=512                   # Width of the generated images
)

# Retrieve the generated images from the results
images = results.images

# Save each generated image to a file
for i, img in enumerate(images):
    img.save(f"image_{i}.png")  # Save each image as "image_0.png", "image_1.png", etc.
