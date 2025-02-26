import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL
from torch import autocast
import os
from pathlib import Path
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.optimization import get_scheduler
from PIL import Image
import numpy as np
import glob

def train_style(
    style_name,
    image_paths,
    placeholder_token="<new-style>",
    initializer_token="art",
    num_training_steps=3000,
    learning_rate=1e-4,
    train_batch_size=1,
    gradient_accumulation_steps=4,
    output_dir="style_embeddings"
):
    accelerator = Accelerator()
    model_id = "runwayml/stable-diffusion-v1-5"
    
    # Load the tokenizer and text encoder
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    
    # Add the placeholder token to the tokenizer
    num_added_tokens = tokenizer.add_tokens(placeholder_token)
    token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
    initializer_token_id = token_ids[0]
    placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
    
    # Resize the token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]
    
    # Create the optimizer
    optimizer = torch.optim.AdamW(
        [token_embeds[placeholder_token_id]],
        lr=learning_rate
    )
    
    # Load and process training images
    train_images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert('RGB')
        image = image.resize((512, 512))
        train_images.append(np.array(image))
    train_images = torch.tensor(np.stack(train_images)).permute(0, 3, 1, 2) / 255.0
    
    # Training loop
    for step in range(num_training_steps):
        with accelerator.accumulate():
            loss = train_batch(text_encoder, train_images, placeholder_token, tokenizer)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
        
        if step % 100 == 0:
            print(f"Step {step}: Loss {loss.item()}")
    
    # Save the trained embeddings
    learned_embeds = token_embeds[placeholder_token_id].detach().cpu()
    learned_embeds_dict = {placeholder_token: learned_embeds}
    
    output_path = Path(output_dir) / f"{style_name}.bin"
    torch.save(learned_embeds_dict, output_path)
    print(f"Saved style embedding to {output_path}")

def train_batch(text_encoder, images, placeholder_token, tokenizer):
    # Implement your training logic here
    # This is a simplified version - you'll need to adapt this based on your needs
    text_input = tokenizer(
        [f"a photo in {placeholder_token} style"] * len(images),
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    
    encoder_hidden_states = text_encoder(text_input.input_ids)[0]
    # Add your loss computation here
    loss = torch.mean((encoder_hidden_states - images.mean()) ** 2)
    return loss

# Example usage
if __name__ == "__main__":
    styles_to_train = [
        {
            "name": "dhoni",
            "images": ["training_images/dhoni/*.jpg"]
        },
        {
            "name": "mickey_mouse",
            "images": ["training_images/mickey_mouse/*.jpg"]
        },
        {
            "name": "balloon",
            "images": ["training_images/balloon/*.jpg"]
        },
        {
            "name": "lion_king",
            "images": ["training_images/lion_king/*.jpg"]
        },
        {
            "name": "rose_flower",
            "images": ["training_images/rose_flower/*.jpg"]
        }
    ]
    
    for style in styles_to_train:
        print(f"\nTraining {style['name']} style...")
        train_style(
            style_name=style["name"],
            image_paths=glob.glob(style["images"][0]),
            placeholder_token=f"<{style['name']}-style>",
            num_training_steps=3000
        )