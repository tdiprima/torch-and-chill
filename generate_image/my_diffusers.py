# Needs torch>=2.4.0
# No matching distribution found for torch>=2.4.0

import torch
# Diffusers: Generate an image from text using Stable Diffusion
from diffusers import DiffusionPipeline

# Load pre-trained Stable Diffusion model (requires torch)
pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe = pipe.to("cpu")  # Use "cuda" for GPU if available

prompt = "A futuristic cityscape at sunset"
image = pipe(prompt).images[0]  # Generate image
image.save("generated_image.png")  # Save to file
print("Image generated and saved as 'generated_image.png'!")
