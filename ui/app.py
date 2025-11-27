"""
Gradio interface for DreamScape AI (Deliverable 3)

- Uses multimodal_generation_d3.py from the src/ folder
- Skips CLIP scoring for faster runtime
- Defaults to Fast Mode = ON
"""

from pathlib import Path
import sys
import gradio as gr

# --------------------------------------------------------------------
# 1. Resolve project paths and import generation functions
# --------------------------------------------------------------------

# ui/app.py → repo root = parent directory
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results"
DATA_DIR = REPO_ROOT / "data"

print("📂 Repo Root     :", REPO_ROOT)
print("📂 Src Folder    :", SRC_DIR)
print("📂 Results Folder:", RESULTS_DIR)
print("📂 Data Folder   :", DATA_DIR)

# Make src/ importable
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import DreamScape D3 multimodal generator
import multimodal_generation_d3 as d3
from multimodal_generation_d3 import run_all_with_audio

# --------------------------------------------------------------------
# 2. Skip CLIP scoring for UI (reduces latency)
# --------------------------------------------------------------------

def _no_clip(prompt, image):
    return None

if hasattr(d3, "compute_clip_score"):
    d3.compute_clip_score = _no_clip
    print("⚙️  CLIP scoring disabled for UI.")
else:
    print("⚠️  compute_clip_score not found. Nothing patched.")

# --------------------------------------------------------------------
# 3. Build Gradio Blocks Interface
# --------------------------------------------------------------------

with gr.Blocks(title="DreamScape AI — Multimodal Dream Generator") as demo:
    gr.Markdown(
        """
        # 🌌 DreamScape AI — Multimodal Dream Generator (Deliverable 3)

        Enter or record a dream, and the system will generate:

        - 🎨 Surreal dream **image** (Stable Diffusion Turbo)  
        - 🎵 Ambient **audio** (MusicGen or fallback)  
        - 🎨 Six-style **moodboard**  
        - 🕸️ Symbolic **motif graph** (NER + lexicon)  
        - 📊 **Toxicity & runtime** report  
        
        *(CLIP alignment disabled in this UI to improve runtime)*

        Runs entirely on **local CPU / MPS / GPU**.
        """
    )

    with gr.Row():
        # ----------------------------------------------------------
        # Left Column — Inputs
        # ----------------------------------------------------------
        with gr.Column(scale=1):
            tbox = gr.Textbox(
                label="Describe your dream (text input)",
                lines=4,
                value=(
                    "I was walking through a museum that slowly turned into a forest. "
                    "The paintings became windows into a folding night sky."
                ),
                placeholder=(
                    "Example: My reflection in the mirror started breathing and turned "
                    "into a bird flying through a burning city..."
                ),
            )

            aud = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Optionally: record/upload your dream (audio)",
            )

            prefer_audio = gr.Checkbox(
                value=True,
                label="Use audio transcript if available",
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
                value=True,
                label="Fast mode (smaller image, fewer diffusion steps)",
            )

            run = gr.Button("Generate Multimodal Dream", variant="primary")

        # ----------------------------------------------------------
        # Right Column — Outputs
        # ----------------------------------------------------------
        with gr.Column(scale=1):
            with gr.Tab("Image & Audio"):
                out_img = gr.Image(label="Generated Dream Image")
                out_aud = gr.Audio(label="Generated Ambient Audio")

            with gr.Tab("Moodboard & Motifs"):
                out_mb = gr.Image(label="Moodboard (6 styles)")
                out_g = gr.Image(label="Motif Graph")

            with gr.Tab("Analysis"):
                out_t = gr.Textbox(
                    label="Toxicity & Runtime Report",
                    lines=4,
                )
                out_asr = gr.Textbox(
                    label="Transcribed Text (if audio provided)",
                    lines=4,
                )

    # Connect UI → model
    run.click(
        fn=run_all_with_audio,
        inputs=[tbox, aud, prefer_audio, make_mood, make_motif, fast],
        outputs=[out_img, out_aud, out_mb, out_g, out_t, out_asr],
    )

# --------------------------------------------------------------------
# 4. Launch app
# --------------------------------------------------------------------

demo.launch(
    debug=True,
    allowed_paths=[str(RESULTS_DIR)],   # let Gradio serve generated images/audio
)