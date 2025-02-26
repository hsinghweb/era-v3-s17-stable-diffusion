import torch
from diffusers import StableDiffusionPipeline
from torch import autocast
import streamlit as st
from huggingface_hub import hf_hub_download
import os
from pathlib import Path
import traceback
import glob
from PIL import Image

# Reuse the same load_learned_embed_in_clip and Distance_loss functions
def load_learned_embed_in_clip(learned_embeds_path, text_encoder, tokenizer, token=None):
    loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")
    trained_token = list(loaded_learned_embeds.keys())[0]
    embeds = loaded_learned_embeds[trained_token]

    # Get the expected dimension from the text encoder
    expected_dim = text_encoder.get_input_embeddings().weight.shape[1]
    current_dim = embeds.shape[0]

    # Resize embeddings if dimensions don't match
    if current_dim != expected_dim:
        print(f"Resizing embedding from {current_dim} to {expected_dim}")
        # Option 1: Truncate or pad with zeros
        if current_dim > expected_dim:
            embeds = embeds[:expected_dim]
        else:
            embeds = torch.cat([embeds, torch.zeros(expected_dim - current_dim)], dim=0)
        
    # Reshape to match expected dimensions
    embeds = embeds.unsqueeze(0)  # Add batch dimension
    
    # Cast to dtype of text_encoder
    dtype = text_encoder.get_input_embeddings().weight.dtype
    embeds = embeds.to(dtype)

    # Add the token in tokenizer
    token = token if token is not None else trained_token
    num_added_tokens = tokenizer.add_tokens(token)
    
    # Resize the token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))
    
    # Get the id for the token and assign the embeds
    token_id = tokenizer.convert_tokens_to_ids(token)
    text_encoder.get_input_embeddings().weight.data[token_id] = embeds[0]
    return token

def Distance_loss(images):
    # Ensure we're working with gradients
    if not images.requires_grad:
        images = images.detach().requires_grad_(True)
    
    # Convert to float32 and normalize
    images = images.float() / 2 + 0.5
    
    # Get RGB channels
    red = images[:,0:1]
    green = images[:,1:2]
    blue = images[:,2:3]
    
    # Calculate color distances using L2 norm
    rg_distance = ((red - green) ** 2).mean()
    rb_distance = ((red - blue) ** 2).mean()
    gb_distance = ((green - blue) ** 2).mean()
    
    return (rg_distance + rb_distance + gb_distance) * 100  # Scale up the loss

class StyleGenerator:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self.pipe = None
        self.style_tokens = []
        self.styles = [
            "dhoni",
            "mickey_mouse",
            "balloon",
            "lion_king",
            "rose_flower"
        ]
        self.style_names = [
            "Dhoni Style",
            "Mickey Mouse Style",
            "Balloon Style",
            "Lion King Style",
            "Rose Flower Style"
        ]
        self.is_initialized = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            print("NVIDIA GPU not found. Running on CPU (this will be slower)")

    def initialize_model(self):
        if self.is_initialized:
            return
            
        try:
            print("Initializing Stable Diffusion model...")
            model_id = "runwayml/stable-diffusion-v1-5"
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None
            )
            self.pipe = self.pipe.to(self.device)
            
            # Load style embeddings from current directory
            current_dir = Path(__file__).parent
            
            for style, style_name in zip(self.styles, self.style_names):
                style_path = current_dir / f"{style}.bin"
                if not style_path.exists():
                    raise FileNotFoundError(f"Style embedding not found: {style_path}")
                
                print(f"Loading style: {style_name}")
                token = load_learned_embed_in_clip(str(style_path), self.pipe.text_encoder, self.pipe.tokenizer)
                self.style_tokens.append(token)
                print(f"✓ Loaded style: {style_name}")
            
            self.is_initialized = True
            print(f"Model initialization complete! Using device: {self.device}")
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            print(traceback.format_exc())
            raise

    def generate_single_style(self, prompt, selected_style):
        try:
            # Find the index of the selected style
            style_idx = self.style_names.index(selected_style)
            
            # Generate single image with selected style
            styled_prompt = f"{prompt}, {self.style_tokens[style_idx]}"
            
            # Set seed for reproducibility
            generator_seed = 42
            torch.manual_seed(generator_seed)
            if self.device == "cuda":
                torch.cuda.manual_seed(generator_seed)
            
            # Generate base image
            with autocast(self.device):
                base_image = self.pipe(
                    styled_prompt,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    generator=torch.Generator(self.device).manual_seed(generator_seed)
                ).images[0]
            
            # Generate same image with loss
            with autocast(self.device):
                loss_image = self.pipe(
                    styled_prompt,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    callback=self.callback_fn,
                    callback_steps=5,
                    generator=torch.Generator(self.device).manual_seed(generator_seed)
                ).images[0]
            
            return base_image, loss_image
            
        except Exception as e:
            print(f"Error in generate_single_style: {e}")
            raise

    def callback_fn(self, i, t, latents):
        if i % 5 == 0:  # Apply loss every 5 steps
            try:
                # Create a copy that requires gradients
                latents_copy = latents.detach().clone()
                latents_copy.requires_grad_(True)
                
                # Compute loss
                loss = Distance_loss(latents_copy)
                
                # Compute gradients
                if loss.requires_grad:
                    grads = torch.autograd.grad(
                        outputs=loss,
                        inputs=latents_copy,
                        allow_unused=True,
                        retain_graph=False
                    )[0]
                    
                    if grads is not None:
                        # Apply gradients to original latents
                        return latents - 0.1 * grads.detach()
            
            except Exception as e:
                print(f"Error in callback: {e}")
            
        return latents

# Set page config
st.set_page_config(
    page_title="AI Style Transfer Studio",
    page_icon="🎨",
    layout="wide"
)

# Apply custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #1f2937;
    }
    .stMarkdown {
        color: #f3f4f6;
    }
    .stButton > button {
        background-color: #6366F1;
        color: white;
    }
    .stButton > button:hover {
        background-color: #4F46E5;
    }
    .dark-theme {
        background-color: #111827;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #374151;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="dark-theme" style="text-align: center;">
    <h1>🎨 AI Style Transfer Studio</h1>
    <h3>Transform your ideas into artistic masterpieces</h3>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'generator' not in st.session_state:
    st.session_state.generator = StyleGenerator.get_instance()
    if not st.session_state.generator.is_initialized:
        st.session_state.generator.initialize_model()

# Sidebar for controls
with st.sidebar:
    st.markdown("## 🎯 Controls")
    
    prompt = st.text_area(
        "What would you like to create?",
        placeholder="e.g., a soccer player celebrating a goal",
        height=100
    )
    
    selected_style = st.radio(
        "Choose Your Style",
        st.session_state.generator.style_names,
        index=0
    )
    
    if st.button("🚀 Generate Artwork", use_container_width=True):
        if prompt:
            try:
                with st.spinner("Generating your artwork..."):
                    base_image, loss_image = st.session_state.generator.generate_single_style(prompt, selected_style)
                    
                    # Store images in session state
                    st.session_state.base_image = base_image
                    st.session_state.loss_image = loss_image
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a prompt first!")

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Original Style")
    if 'base_image' in st.session_state:
        st.image(st.session_state.base_image, use_column_width=True)

with col2:
    st.markdown("### Color Enhanced")
    if 'loss_image' in st.session_state:
        st.image(st.session_state.loss_image, use_column_width=True)

# Example Gallery
st.markdown("""
<div class="dark-theme">
    <h2>🎆 Example Gallery</h2>
    <p>Compare original and enhanced versions for each style:</p>
</div>
""", unsafe_allow_html=True)

# Load and display example images
try:
    output_dir = Path("Outputs")
    original_dir = output_dir
    enhanced_dir = output_dir / "Color_Enhanced"

    if enhanced_dir.exists():
        original_images = {
            Path(f).stem.split('_example')[0]: f 
            for f in original_dir.glob("*.webp") 
            if '_example' in f.name
        }
        enhanced_images = {
            Path(f).stem.split('_example')[0]: f 
            for f in enhanced_dir.glob("*.webp") 
            if '_example' in f.name
        }

        styles = [
            ("ronaldo", "Ronaldo Style"),
            ("canna_lily", "Canna Lily"),
            ("three_stooges", "Three Stooges"),
            ("pop_art", "Pop Art"),
            ("bird_style", "Bird Style")
        ]

        for style_key, style_name in styles:
            if style_key in original_images and style_key in enhanced_images:
                st.markdown(f"### {style_name}")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(
                        str(original_images[style_key]),
                        caption="Original",
                        use_column_width=True
                    )
                with col2:
                    st.image(
                        str(enhanced_images[style_key]),
                        caption="Color Enhanced",
                        use_column_width=True
                    )
                st.markdown("<hr>", unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error loading example gallery: {str(e)}")

# Info sections
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="dark-theme">
        <h2>🎨 Style Guide</h2>
        <table>
            <tr>
                <th>Style</th>
                <th>Best For</th>
            </tr>
            <tr>
                <td><strong>Ronaldo Style</strong></td>
                <td>Dynamic sports scenes, action shots, celebrations</td>
            </tr>
            <tr>
                <td><strong>Canna Lily</strong></td>
                <td>Natural scenes, floral compositions, garden imagery</td>
            </tr>
            <tr>
                <td><strong>Three Stooges</strong></td>
                <td>Comedy, humor, expressive character portraits</td>
            </tr>
            <tr>
                <td><strong>Pop Art</strong></td>
                <td>Vibrant artwork, bold colors, stylized designs</td>
            </tr>
            <tr>
                <td><strong>Bird Style</strong></td>
                <td>Wildlife, nature scenes, peaceful landscapes</td>
            </tr>
        </table>
        <em>Choose the style that best matches your creative vision</em>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="dark-theme">
        <h2>🔍 Color Enhancement Technology</h2>
        
        <p>Our advanced color processing uses distance loss to