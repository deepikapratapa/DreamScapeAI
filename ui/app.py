"""
Gradio interface for DreamScape AI (Deliverable 3)

- Uses multimodal_generation_d3.py from the notebooks/ folder
- Skips CLIP scoring for faster runtime
- Defaults to Fast Mode = ON
"""

from pathlib import Path
import sys

import gradio as gr

# --------------------------------------------------------------------
# 1. Resolve project paths and import generation functions
# --------------------------------------------------------------------

# ui/app.py -> project root is two levels up
ROOT_DIR = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
RESULTS_DIR = NOTEBOOKS_DIR / "results"

print("📂 ROOT       :", ROOT_DIR)
print("📂 NOTEBOOKS  :", NOTEBOOKS_DIR)

# Make notebooks/ importable
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

# Import the generation utilities from the exported notebook script
import multimodal_generation_d3 as d3
from multimodal_generation_d3 import run_all_with_audio

# --------------------------------------------------------------------
# 2. Skip CLIP scoring for faster runtime
#    (monkey-patch compute_clip_score to a cheap no-op)
# --------------------------------------------------------------------

def _no_clip(prompt, image):
    """Disable CLIP scoring (returns None instead of running the model)."""
    return None

# Monkey-patch inside the module so all callers see the new behavior
if hasattr(d3, "compute_clip_score"):
    d3.compute_clip_score = _no_clip
    print("⚙️  CLIP scoring disabled for UI (compute_clip_score -> None).")
else:
    print("⚠️  compute_clip_score not found in multimodal_generation_d3. Nothing patched.")

# --------------------------------------------------------------------
# 3. Build Gradio Blocks interface
# --------------------------------------------------------------------

with gr.Blocks(title="DreamScape AI — Multimodal Dream Generator") as demo:
    gr.Markdown(
        """
        # 🌌 DreamScape AI — Multimodal Dream Generator (D3)

        Enter or record a dream, and the system will generate:
        - 🎨 A surreal image (Stable Diffusion Turbo)  
        - 🎵 Ambient audio (MusicGen or fallback)  
        - 🧩 A symbol / motif graph (NER + lexicon)  
        - 🎨 A 6-tile moodboard  
        - 📊 Toxicity score & runtime (CLIP alignment disabled in UI to reduce latency)  

        This prototype runs fully on local hardware (CPU / MPS / GPU).
        """
    )

    with gr.Row():
        # ---------------- Left column: inputs ----------------
        with gr.Column(scale=1):
            tbox = gr.Textbox(
                label="Describe your dream (text input)",
                lines=4,
                value=(
                    "I was walking through a museum that slowly turned into a forest. "
                    "The paintings on the wall became windows, and through them I could "
                    "see the night sky folding into itself like paper."
                ),
                placeholder=(
                    "Example: My reflection in the mirror started breathing, "
                    "then turned into a bird flying through a burning city..."
                ),
            )
            aud = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Optional: record/upload your dream (audio)",
            )
            prefer_audio = gr.Checkbox(
                value=True,
                label="Use audio transcript if audio is provided",
            )
            make_mood = gr.Checkbox(
                value=True,
                label="Generate moodboard",
            )
            make_motif = gr.Checkbox(
                value=True,
                label="Generate motif graph",
            )
            fast = gr.Checkbox(
                value=True,  # ✅ Default Fast Mode ON
                label="Fast mode (smaller image, fewer steps)",
            )
            run = gr.Button("Generate Multimodal Dream", variant="primary")

        # ---------------- Right column: outputs ----------------
        with gr.Column(scale=1):
            with gr.Tab("Image & Audio"):
                out_img = gr.Image(label="Generated Image")
                out_aud = gr.Audio(label="Generated Audio")

            with gr.Tab("Moodboard & Motifs"):
                out_mb = gr.Image(label="Moodboard (6 styles)")
                out_g = gr.Image(label="Motif Graph")

            with gr.Tab("Analysis"):
                out_t = gr.Textbox(
                    label="Toxicity & runtime (CLIP disabled in UI)",
                    lines=4,
                )
                out_asr = gr.Textbox(
                    label="Transcribed Text (if audio used)",
                    lines=4,
                )

    # Wire button to generation function from multimodal_generation_d3.py
    run.click(
        fn=run_all_with_audio,
        inputs=[tbox, aud, prefer_audio, make_mood, make_motif, fast],
        outputs=[out_img, out_aud, out_mb, out_g, out_t, out_asr],
    )

# --------------------------------------------------------------------
# 4. Launch app
#    allowed_paths lets Gradio serve files from notebooks/results
# --------------------------------------------------------------------

demo.launch(
    debug=True,
    allowed_paths=[str(RESULTS_DIR)],
)