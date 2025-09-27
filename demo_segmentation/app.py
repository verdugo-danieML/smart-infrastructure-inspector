# app.py  (Florence-2 + SAM-2 – ZeroGPU, no supervision for masks)
import os
from pathlib import Path
import spaces
import gradio as gr
import supervision as sv
import torch
import numpy as np
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor
import cv2

# ------------------------------------------------------------------ #
# 1. Environment Setup for ZeroGPU
# ------------------------------------------------------------------ #
os.environ['SAM2_BUILD_CUDA'] = '0'
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0;8.6'

# ------------------------------------------------------------------ #
# 2. Globals
# ------------------------------------------------------------------ #
PEFT_MODEL_PATH = "florence2-lora"
BASE_MODEL_PATH = "microsoft/Florence-2-base-ft"
REVISION = 'refs/pr/6'
TASK = "<OD>"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------ #
# 3. Florence-2 model
# ------------------------------------------------------------------ #
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH, trust_remote_code=True, revision=REVISION
).to(DEVICE)

processor = AutoProcessor.from_pretrained(
    BASE_MODEL_PATH, trust_remote_code=True, revision=REVISION
)

florence_model = PeftModel.from_pretrained(base_model, PEFT_MODEL_PATH, device_map=DEVICE)
florence_model = florence_model.merge_and_unload()

# ------------------------------------------------------------------ #
# 4. SAM-2 config paths
# ------------------------------------------------------------------ #
SAM2_CONFIG = "sam2_hiera_s.yaml"
SAM2_CHECKPOINT = "sam2_finetune/sam2_hiera_small.pt"
SAM2_WEIGHTS = "sam2_finetune/sam2_finetuned_best.pt"

sam2_predictor = None
_model_initialized = False

# ------------------------------------------------------------------ #
# 5. SAM-2 lazy init
# ------------------------------------------------------------------ #
@spaces.GPU
def initialize_sam2():
    global sam2_predictor, _model_initialized
    if _model_initialized:
        return sam2_predictor

    print("Initializing SAM-2 model...")
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    try:
        sam2_model = build_sam2(
            config_file=SAM2_CONFIG,
            ckpt_path=SAM2_CHECKPOINT,
            device=DEVICE,
            apply_postprocessing=False
        )
        sam2_predictor = SAM2ImagePredictor(sam2_model)
        checkpoint = torch.load(SAM2_WEIGHTS, map_location=DEVICE)
        sam2_predictor.model.load_state_dict(checkpoint["model_state_dict"])
        sam2_predictor.model.eval()
        _model_initialized = True
        print("SAM-2 model initialized successfully!")
    except Exception as e:
        print("Error initializing SAM-2:", e)
        raise
    return sam2_predictor

# ------------------------------------------------------------------ #
# 6. Box annotators (supervision still used only for boxes)
# ------------------------------------------------------------------ #
RED = sv.ColorPalette.from_hex(["#FF0000"])
box_ann = sv.BoxAnnotator(color=RED, color_lookup=sv.ColorLookup.INDEX)
label_ann = sv.LabelAnnotator(color=RED, color_lookup=sv.ColorLookup.INDEX)

# ------------------------------------------------------------------ #
# 7. Mask overlay helpers (no supervision)
# ------------------------------------------------------------------ #
def blend_masks(image, masks, alpha=0.6, borders=True):
    """Blend boolean masks onto PIL image -> PIL Image."""
    img = np.asarray(image).copy()
    for m in masks:
        m = m.astype(np.uint8)
        color = np.array([30, 144, 255], dtype=np.uint8)
        overlay = m[..., None] * color  # (H,W,3)
        if borders:
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(c, 0.01, True) for c in contours]
            overlay = cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
        img = cv2.addWeighted(img, 1-alpha, overlay, alpha, 0)
    return Image.fromarray(img)

# ------------------------------------------------------------------ #
# 8. Inference functions
# ------------------------------------------------------------------ #
@spaces.GPU
def detect_cracks(pil_img: Image.Image | None):
    if pil_img is None:
        return None, None
    inputs = processor(text=TASK, images=pil_img, return_tensors="pt").to(DEVICE)
    with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
        gen_ids = florence_model.generate(**inputs, max_new_tokens=1024, num_beams=3)
    gen_text = processor.batch_decode(gen_ids, skip_special_tokens=False)[0]
    response = processor.post_process_generation(
        gen_text, task=TASK, image_size=(pil_img.width, pil_img.height)
    )
    detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, response, resolution_wh=pil_img.size)
    annotated = pil_img.copy()
    annotated = box_ann.annotate(annotated, detections)
    annotated = label_ann.annotate(annotated, detections)
    return annotated, detections

@spaces.GPU(duration=120)
def segment_cracks(pil_img: Image.Image | None, detections):
    if pil_img is None or detections is None or len(detections) == 0:
        return pil_img
    global sam2_predictor
    if sam2_predictor is None:
        sam2_predictor = initialize_sam2()

    image_np = np.array(pil_img.convert("RGB"))
    sam2_predictor.set_image(image_np)
    boxes = detections.xyxy

    with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
        masks, scores, _ = sam2_predictor.predict(
            point_coords=None, point_labels=None, box=boxes, multimask_output=False
        )
    masks = masks.squeeze(1) if masks.ndim == 4 else masks
    masks = masks = masks.astype(bool)
    return blend_masks(pil_img, masks, alpha=0.6, borders=True)

@spaces.GPU(duration=120)
def detect_and_segment(pil_img: Image.Image | None):
    if pil_img is None:
        return None, None
    boxes_annotated, detections = detect_cracks(pil_img)
    if len(detections) == 0:
        return boxes_annotated, pil_img
    global sam2_predictor
    if sam2_predictor is None:
        sam2_predictor = initialize_sam2()

    image_np = np.array(pil_img.convert("RGB"))
    sam2_predictor.set_image(image_np)
    boxes = detections.xyxy

    with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
        masks, scores, _ = sam2_predictor.predict(
            point_coords=None, point_labels=None, box=boxes, multimask_output=False
        )
    masks = masks.squeeze(1) if masks.ndim == 4 else masks
    masks = masks = masks.astype(bool)
    masks_annotated = blend_masks(pil_img, masks, alpha=0.6, borders=True)
    return boxes_annotated, masks_annotated

# ------------------------------------------------------------------ #
# 9.  Gradio UI  –  NEW SKIN ONLY
# ------------------------------------------------------------------ #
examples = sorted([str(p) for p in Path("examples").glob("*")
                  if p.suffix.lower() in (".png", ".jpg", ".jpeg")])

css = """
/* ----------  layout ---------- */
.cont {max-width: 1200px; margin: auto; padding: 2rem}
/* ----------  card ---------- */
.card {
    background: #ffffff;
    border-radius: 12px;
    box-shadow: 0 4px 16px rgba(0,0,0,.08);
    padding: 1.2rem;
    margin-bottom: 1rem;
}
.dark .card {background: #1e1e1e}
/* ----------  buttons ---------- */
.btn-row {display: flex; gap: .75rem; margin-top: .5rem}
/* ----------  tiny status ---------- */
.status {font-size: .85rem; color: #666; margin-top: .5rem}
"""

with gr.Blocks(title="Crack Detector – Florence-2 + SAM-2 (ZeroGPU)",
               css=css, theme=gr.themes.Soft()) as demo:

    gr.HTML('<div class="cont">')

    gr.Markdown("## 🔍 Crack Detection & Segmentation with Florence-2 + SAM-2")
    gr.Markdown("Upload an image and detect cracks using Florence-2, then segment them with SAM-2.  **Note:** First segmentation may take longer due to model initialization.")

    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(label="Input", type="pil", sources=["upload", "webcam"],
                          elem_classes="card")

            

            examples_gal = gr.Gallery(value=examples, columns=3, height=240,
                                     object_fit="cover", label="Examples",
                                     elem_classes="card")


        with gr.Column(scale=2):
            boxes_out = gr.Image(label="Florence-2 boxes", type="pil", elem_classes="card")
            masks_out = gr.Image(label="SAM-2 masks",   type="pil", elem_classes="card")

    with gr.Row(elem_classes="btn-row"):
        #detect_btn   = gr.Button("Detect cracks", variant="primary")
        #segment_btn  = gr.Button("Segment cracks", variant="primary")
        combined_btn = gr.Button("Detect + Segment (faster)", variant="secondary")

    detections_state = gr.State(value=None)

    # ------------- existing back-end bindings ------------- #
    #detect_btn.click(detect_cracks, inputs=inp,
    #                outputs=[boxes_out, detections_state])
    #segment_btn.click(segment_cracks, inputs=[inp, detections_state],
    #                 outputs=masks_out)
    combined_btn.click(detect_and_segment, inputs=inp,
                      outputs=[boxes_out, masks_out])


    gr.Markdown("### Technical Details\n"
               "- **Florence-2**: Fine-tuned for crack detection  \n"
               "- **SAM-2**: Fine-tuned for crack segmentation  \n"
               "- **ZeroGPU**: Dynamic GPU allocation for efficient inference  \n"
               "- First run initializes models (may take ~30 seconds)")

    gr.HTML('</div>')  # close cont

demo.queue(max_size=20).launch()