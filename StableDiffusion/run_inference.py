import torch
from diffusers import StableDiffusionPipeline

#Change the model path here
model_path = "/project/sz457/ms3537/DS677/diffusers/examples/text_to_image/sd_flickr30k_model10k"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda")

my_prompt = "A father holding a toddler’s hand while pointing at a carouse"
image = pipe(prompt= my_prompt).images[0]
image.save(f"generated_sd_flickr30k_model10k/{my_prompt}.png")#Change path here 
# image.save(f"generated/{my_prompt}.png")