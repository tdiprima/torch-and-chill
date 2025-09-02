import torch
from diffusers import DiffusionPipeline

# Load the pipeline (Stable Diffusion v1-4)
pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

# Move to CPU (use "cuda" if you have a GPU and CUDA-enabled PyTorch)
pipe = pipe.to("cpu")

# Optional: Enable CPU offloading to reduce memory usage (helps on low-RAM systems)
pipe.enable_sequential_cpu_offload()

# Your prompt
prompt = "A futuristic cityscape at sunset"

# Generate the image
try:
    image = pipe(prompt).images[0]
    image.save("generated_image.png")
    print("Image generated and saved as 'generated_image.png'!")
except Exception as e:
    print(f"An error occurred: {e}")
    print("Try checking your dependencies or available memory.")
