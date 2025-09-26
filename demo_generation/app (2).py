"""
Hugging-Face Spaces – ZeroGPU version
"""
import gradio as gr
import torch
from diffusers import DiffusionPipeline
import spaces  # <--- decorator source
import numpy as np
from PIL import Image
import os

# --------------------------------------------------
# 1. Config
# --------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")            # optional
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
LORA_ID  = "danie94-lml/sdxl-lora-cracks"

# --------------------------------------------------
# 2. Prompt templates (MODIFIED AND SIMPLIFIED)
# --------------------------------------------------
# Simplified choices for the UI dropdowns
MATERIAL_CHOICES = ["concrete", "asphalt", "masonry", "ceramic"]

CRACK_PATTERN_CHOICES = [
    "linear crack",
    "branching crack",
    "map/alligator crack pattern",
    "spider web crack pattern",
    "hairline crack network",
    "interconnected crack network"
]

SEVERITY_CHOICES = [
    "fine hairline",
    "medium-width",
    "wide",
    "severe and crumbling"
]

# --------------------------------------------------
# 3. Generation core  (ZeroGPU decorated)
# --------------------------------------------------
@spaces.GPU 
def generate_crack_image(
    material_type,
    crack_pattern,
    severity,
    custom_prompt,
    use_custom,
    negative_prompt,
    num_inference_steps,
    guidance_scale,
    seed,
    width,
    height):
    
    # Load the pipeline and LoRA weights
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        safety_checker=None,
        use_safensors=True,
        variant="fp16"
    )
    pipe.load_lora_weights(LORA_ID)
    pipe = pipe.to("cuda")

    # Set up the generator for reproducible results if a seed is provided
    generator = None
    if seed != -1:
        generator = torch.Generator(device="cuda").manual_seed(int(seed))

    # --- PROMPT LOGIC MODIFIED ---
    if use_custom and custom_prompt:
        prompt = custom_prompt
    else:
        # New simplified prompt structure based on metadata analysis
        prompt = (
            f"A photorealistic, high-resolution image of a {severity} {crack_pattern} in a {material_type} surface, "
            f"detailed photograph, sharp focus, detailed texture."
        )

    # --- NEGATIVE PROMPT MODIFIED ---
    full_negative = (
        f"{negative_prompt}, cartoon, 3d, painting, illustration, anime, "
        "blurry, pixelated, unrealistic, oversaturated, "
        "underexposed, overexposed, grain"
    )

    try:
        result = pipe(
            prompt=prompt,
            negative_prompt=full_negative,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=guidance_scale,
            generator=generator,
            width=int(width),
            height=int(height)
        )
        image = result.images[0]
        info = f"✅ Generated | Steps: {num_inference_steps} | Guidance: {guidance_scale}"
        return image, info, prompt
    except Exception as e:
        return None, f"❌ Generation failed: {str(e)}", prompt

# --------------------------------------------------
# 4. Comparison grid helper (MODIFIED for new prompts)
# --------------------------------------------------
def create_comparison_grid(material_type, severity):
    images = []
    # Using a fixed set of patterns for a consistent grid
    comparison_patterns = [
        "linear crack",
        "branching crack",
        "map/alligator crack pattern",
        "spider web crack pattern"
    ]

    for pattern in comparison_patterns:
        img, _, _ = generate_crack_image(
            material_type=material_type,
            crack_pattern=pattern,
            severity=severity,
            custom_prompt="",
            use_custom=False,
            negative_prompt="worst quality, low quality",
            num_inference_steps=30,
            guidance_scale=7.5,
            seed=42, # Use a fixed seed for consistency
            width=512,
            height=512
        )
        if img:
            images.append(img)

    if not images or len(images) < 4:
        # Return a placeholder or error image if generation fails
        return None

    # Create a 2x2 grid
    grid_img = Image.new("RGB", (1024, 1024))
    positions = [(0, 0), (512, 0), (0, 512), (512, 512)]
    for idx, img in enumerate(images):
        img = img.resize((512, 512))
        grid_img.paste(img, positions[idx])

    return grid_img

# --------------------------------------------------
# 5. UI helpers (MODIFIED for new prompts)
# --------------------------------------------------
def update_prompt_preview(material, pattern, severity):
    # New simplified preview logic
    return (
        f"A photorealistic, high-resolution image of a {severity} {pattern} in a {material} surface, "
        f"detailed photograph, sharp focus, detailed texture."
    )

def toggle_custom_prompt(use_custom):
    # This function remains the same, it just toggles visibility
    return (
        gr.update(visible=not use_custom),
        gr.update(visible=not use_custom),
        gr.update(visible=not use_custom),
        gr.update(visible=use_custom),
        gr.update(visible=not use_custom) # Hides prompt preview when custom is active
    )

# --------------------------------------------------
# 6. Gradio interface
# --------------------------------------------------
def create_interface():
    with gr.Blocks(title="SDXL LoRA – Infrastructure Crack Generation", theme=gr.themes.Soft()) as demo:
        # Header HTML remains the same
        gr.HTML("""
        <div style='text-align:center;padding:20px;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);border-radius:10px;margin-bottom:20px;'>
          <h1 style='color:white;font-size:2.5em;margin-bottom:10px;'>🎨 SDXL LoRA for Infrastructure Cracks</h1>
          <p style='color:rgba(255,255,255,0.9);font-size:1.1em;'>Fine-tuned Stable Diffusion XL for synthetic crack pattern generation</p>
          <div style='display:flex;justify-content:center;gap:15px;margin-top:15px;'>
            <a href='https://huggingface.co/danie94-lml/sdxl-lora-cracks' style='background:white;color:#667eea;padding:8px 16px;border-radius:5px;text-decoration:none;font-weight:bold;'>🤗 Model</a>
            <a href='https://huggingface.co/datasets/danie94-lml/crack_dataset' style='background:white;color:#667eea;padding:8px 16px;border-radius:5px;text-decoration:none;font-weight:bold;'>📊 Dataset</a>
          </div>
        </div>
        """)

        with gr.Tabs():
            with gr.Tab("🖼️ Generate Single Image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Generation Parameters")
                        # --- UI CHOICES MODIFIED ---
                        material_type = gr.Dropdown(label="🏗️ Material Type", choices=MATERIAL_CHOICES, value="concrete")
                        crack_pattern = gr.Dropdown(label="🔍 Crack Pattern", choices=CRACK_PATTERN_CHOICES, value="linear crack")
                        severity = gr.Radio(label="⚠️ Severity / Width", choices=SEVERITY_CHOICES, value="medium-width")

                        with gr.Accordion("🎨 Custom Prompt", open=False):
                            use_custom = gr.Checkbox(label="Use custom prompt instead of template", value=False)
                            custom_prompt = gr.Textbox(label="Custom Prompt", lines=3, visible=False)

                        with gr.Accordion("⚙️ Advanced Settings", open=False):
                            # --- NEGATIVE PROMPT MODIFIED ---
                            negative_prompt = gr.Textbox(label="Negative Prompt", value="worst quality, low quality, lowres, bad anatomy", lines=2)
                            with gr.Row():
                                width = gr.Slider(label="Width", minimum=512, maximum=1024, value=1024, step=128)
                                height = gr.Slider(label="Height", minimum=512, maximum=1024, value=1024, step=128)
                            with gr.Row():
                                num_inference_steps = gr.Slider(label="Inference Steps", minimum=20, maximum=100, value=50, step=5)
                                guidance_scale = gr.Slider(label="Guidance Scale", minimum=1, maximum=20, value=7.5, step=0.5)
                            seed = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)

                        generate_btn = gr.Button("🚀 Generate Image", variant="primary", size="lg")
                        prompt_preview = gr.Textbox(label="📝 Prompt Preview", lines=3, interactive=False)

                    with gr.Column(scale=1):
                        output_image = gr.Image(label="Generated Crack Pattern", type="pil")
                        status_text = gr.Textbox(label="Status", value="Ready to generate…", interactive=False)

            with gr.Tab("🔲 Pattern Comparison"):
                gr.Markdown("### Generate Multiple Crack Patterns for Comparison")
                with gr.Row():
                    comp_material = gr.Dropdown(label="Material Type", choices=MATERIAL_CHOICES, value="concrete")
                    comp_severity = gr.Radio(label="Severity / Width", choices=SEVERITY_CHOICES, value="medium-width")
                    comp_generate = gr.Button("Generate Comparison Grid", variant="primary")
                comp_output = gr.Image(label="Crack Pattern Comparison Grid", type="pil")

            with gr.Tab("📚 Technical Details"):
                gr.Markdown("""
                ## Model Architecture
                This is a LoRA fine-tune of the `stabilityai/stable-diffusion-xl-base-1.0` model, specifically trained to generate high-fidelity images of infrastructure cracks.
                - **Base Model**: SDXL 1.0
                - **LoRA Rank**: 32
                - **Training Dataset**: [danie94-lml/crack_dataset](https://huggingface.co/datasets/danie94-lml/crack_dataset) (1,114 images with detailed text descriptions)
                - **Resolution**: Trained on 1024×1024 images
                - **Backend**: This Space runs on a ZeroGPU backend, which means it uses an on-demand GPU that spins up for generation and then shuts down.
                """)

        # Event wiring
        use_custom.change(
            toggle_custom_prompt,
            inputs=[use_custom],
            outputs=[material_type, crack_pattern, severity, custom_prompt, prompt_preview]
        )
        
        # Update preview whenever a template dropdown changes
        for component in [material_type, crack_pattern, severity]:
            component.change(
                update_prompt_preview,
                inputs=[material_type, crack_pattern, severity],
                outputs=[prompt_preview]
            )

        generate_btn.click(
            generate_crack_image,
            inputs=[
                material_type, crack_pattern, severity, custom_prompt, use_custom,
                negative_prompt, num_inference_steps, guidance_scale, seed, width, height
            ],
            outputs=[output_image, status_text, prompt_preview]
        )
        
        comp_generate.click(
            create_comparison_grid,
            inputs=[comp_material, comp_severity],
            outputs=[comp_output]
        )

        # Footer HTML remains the same
        gr.HTML("""
        <div style='text-align:center;padding:20px;margin-top:40px;border-top:1px solid #e0e0e0;'>
          <p style='color:#666;'>Created by Roberto Daniel Verdugo Siqueiros | <a href='mailto:verdugo.rds@gmail.com' style='color:#667eea;'>Contact</a></p>
        </div>
        """)

    return demo

# --------------------------------------------------
# 7. Launch
# --------------------------------------------------
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False)