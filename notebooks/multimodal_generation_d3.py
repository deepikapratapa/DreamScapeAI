#!/usr/bin/env python
# coding: utf-8

# # 🎨 DreamScape AI — Multimodal Generation (Deliverable 3)
# 
# This notebook generates **dream-inspired multimodal outputs** — combining *text, audio, and visuals* using large generative models.  
# It integrates:
# - Stable Diffusion (SD-Turbo) for **illustrations**
# - Faster-Whisper for **speech transcription**
# - MusicGen (via 🤗 Transformers) for **ambient sound**
# - Detoxify for **safety filtering**
# - NetworkX for **motif graphing**
# - CLIP (openai/clip-vit-base-patch32) for **text–image semantic alignment**
# 
# All assets and metadata are saved under `results/` for easy inspection and reporting.

# In[1]:


# (Optional) Only run this if you are missing dependencies
# %pip -q install --upgrade diffusers accelerate pillow gradio soundfile librosa \
#     faster-whisper detoxify torch torchvision torchaudio transformers \
#     networkx matplotlib scipy


# In[2]:


import os, gc, io, math, json, random, warnings, tempfile, re, itertools, time
from pathlib import Path
from datetime import datetime

import numpy as np
import soundfile as sf
import librosa
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import torch
import matplotlib.pyplot as plt
import networkx as nx

from detoxify import Detoxify
from transformers import pipeline

import gradio as gr

warnings.filterwarnings("ignore")

# Output directory
BASE = Path("..").resolve() if Path(".").name == "notebooks" else Path(".").resolve()
OUT_DIR = BASE / "results"
OUT_DIR.mkdir(exist_ok=True)

print("BASE:", BASE)
print("OUT_DIR:", OUT_DIR)


# In[3]:


from diffusers import AutoPipelineForText2Image

# Safety checker imports (optional)
SafetyChecker = None
ClipProcessor = None
try:
    from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker as SafetyChecker
except Exception:
    try:
        from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker as SafetyChecker
    except Exception:
        SafetyChecker = None

try:
    from transformers import CLIPImageProcessor as ClipProcessor
except Exception:
    try:
        from transformers import AutoImageProcessor as ClipProcessor
    except Exception:
        ClipProcessor = None

TXT2IMG_ID = "stabilityai/sd-turbo"

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
dtype = torch.float16 if device == "cuda" else torch.float32

safety_checker = None
feature_extractor = None
if SafetyChecker and ClipProcessor:
    try:
        safety_checker = SafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker"
        )
        feature_extractor = ClipProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
    except Exception as e:
        warnings.warn(f"Safety checker unavailable: {e}")

pipe_kwargs = {}
if safety_checker and feature_extractor:
    pipe_kwargs["safety_checker"] = safety_checker
    pipe_kwargs["feature_extractor"] = feature_extractor

TXT2IMG_PIPE = AutoPipelineForText2Image.from_pretrained(TXT2IMG_ID, **pipe_kwargs).to(device)
try:
    TXT2IMG_PIPE.enable_attention_slicing()
except Exception:
    pass

print(
    f"Loaded text→image: {TXT2IMG_ID} on {device} (dtype={dtype}); "
    f"safety={'ON' if safety_checker else 'OFF (blur fallback)'}"
)


# In[4]:


def set_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def sanitize_prompt(prompt: str):
    """
    Filter toxic or unsafe language before generation using Detoxify.
    If toxicity > 0.5, prepend a softening prefix.
    """
    try:
        scores = Detoxify("original").predict(prompt)
        tox_score = float(scores.get("toxicity", 0.0))
    except Exception:
        tox_score = 0.0
    cleaned = prompt.strip()
    softened = False
    if tox_score > 0.5:
        cleaned = "A calm, imaginative reinterpretation of " + prompt
        softened = True
    return cleaned, tox_score, softened

def save_image(img: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    return path

def now_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# In[5]:


import scipy.io.wavfile as wavfile
from transformers import pipeline, AutoProcessor, MusicgenForConditionalGeneration

MUSICGEN_MODEL_ID = "facebook/musicgen-small"
TTA_DEVICE = -1  # CPU

TTA_PIPE = None
TTA_DIRECT = None
AUDIO_ENGINE = "AmbientFallback"

def _print_versions():
    import transformers, torch, scipy, huggingface_hub
    print(
        "versions =>",
        "transformers", transformers.__version__,
        "| torch", torch.__version__,
        "| scipy", scipy.__version__,
        "| hf-hub", huggingface_hub.__version__,
    )

_print_versions()

# 1) Try the Transformers pipeline("text-to-audio")
try:
    TTA_PIPE = pipeline(
        task="text-to-audio",
        model=MUSICGEN_MODEL_ID,
        device=TTA_DEVICE,
    )
    AUDIO_ENGINE = f"MusicGen/Transformers({MUSICGEN_MODEL_ID})"
    print(f"🎵 Loaded MusicGen via pipeline: {MUSICGEN_MODEL_ID}")
except Exception as e:
    print("⚠️ pipeline(text-to-audio) failed:", e)
    TTA_PIPE = None

# 2) If pipeline failed, try direct model+processor
if TTA_PIPE is None:
    try:
        proc = AutoProcessor.from_pretrained(MUSICGEN_MODEL_ID)
        mdl = MusicgenForConditionalGeneration.from_pretrained(MUSICGEN_MODEL_ID)
        mdl = mdl.to("cpu")
        TTA_DIRECT = (proc, mdl)
        AUDIO_ENGINE = f"MusicGen/Direct({MUSICGEN_MODEL_ID})"
        print(f"🎵 Loaded MusicGen via direct model: {MUSICGEN_MODEL_ID}")
    except Exception as e:
        print("⚠️ Direct MusicGen load failed:", e)
        TTA_DIRECT = None
        AUDIO_ENGINE = "AmbientFallback"
        print("➡️ Will use ambient fallback for audio.")

def text_to_audio(prompt: str, path: Path, seconds: int = 8) -> Path:
    """
    Prefer MusicGen (pipeline). If unavailable, try direct model.
    If both fail, write a quiet sine tone (ambient fallback).
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # A) pipeline route
    if TTA_PIPE is not None:
        try:
            result = TTA_PIPE(
                prompt,
                forward_params={"do_sample": True, "max_new_tokens": int(seconds * 50)},
            )
            sr = int(result["sampling_rate"])
            audio = result["audio"]
            wavfile.write(path, rate=sr, data=(audio * 32767).astype(np.int16))
            return path
        except Exception as e:
            print("⚠️ MusicGen pipeline failed, trying direct:", e)

    # B) direct route
    if TTA_DIRECT is not None:
        try:
            proc, mdl = TTA_DIRECT
            inputs = proc(text=[prompt], padding=True, return_tensors="pt")
            with torch.no_grad():
                audio_values = mdl.generate(
                    **inputs, max_new_tokens=int(seconds * 50)
                )
            sr = mdl.config.audio_encoder.sampling_rate
            audio = audio_values[0, 0].cpu().numpy()
            wavfile.write(path, rate=sr, data=(audio * 32767).astype(np.int16))
            return path
        except Exception as e:
            print("⚠️ MusicGen direct failed, falling back:", e)

    # C) fallback tone
    sr = 22050
    t = np.linspace(0, 5, int(5 * sr), endpoint=False)
    tone = 0.02 * np.sin(2 * np.pi * 220 * t)
    wavfile.write(path, sr, (tone * 32767).astype(np.int16))
    return path

print(f"🔊 AUDIO_ENGINE active: {AUDIO_ENGINE}")


# In[6]:


def text_to_image(
    prompt: str,
    negative: str = "low quality, blurry, watermark, text, logo",
    steps: int = 4,
    guidance: float = 0.0,
    height: int = 512,
    width: int = 512,
    seed: int | None = 1234,
) -> Image.Image:
    set_seed(seed)
    try:
        out = TXT2IMG_PIPE(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=height,
            width=width,
        )
        img = out.images[0]
        flagged = False
        if hasattr(out, "nsfw_content_detected") and out.nsfw_content_detected:
            flagged = bool(out.nsfw_content_detected[0])
        if flagged:
            img = img.filter(ImageFilter.GaussianBlur(radius=24))
        return img
    except Exception as e:
        print("⚠️ Generation retry on CPU:", e)
        pipe_cpu = TXT2IMG_PIPE.to("cpu")
        out = pipe_cpu(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=max(steps, 8),
            guidance_scale=max(guidance, 1.0),
            height=height,
            width=width,
        )
        img = out.images[0]
        return img

def _square_thumb(im: Image.Image, size=384):
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    return im.resize((size, size), Image.LANCZOS)

def compose_grid(images, cols=3, pad=8, bg=(18, 18, 18)):
    tiles = [_square_thumb(img) for img in images]
    w, h = tiles[0].size
    rows = math.ceil(len(tiles) / cols)
    W, H = cols * w + (cols + 1) * pad, rows * h + (rows + 1) * pad
    canvas = Image.new("RGB", (W, H), bg)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas.paste(t, (pad + c * (w + pad), pad + r * (h + pad)))
    return canvas

def text_to_moodboard(text: str, n_images=6, cols=3):
    base_prompt, _, _ = sanitize_prompt(text)
    styles = [
        "surreal cinematic volumetric light",
        "dreamlike watercolor pastel",
        "digital art neon synthwave",
        "oil painting baroque lighting",
        "minimalist muted palette",
        "storybook ink & wash",
    ]
    imgs = []
    for i in range(n_images):
        styl = styles[i % len(styles)]
        prompt_i = f"{base_prompt}. Style: {styl}"
        img = text_to_image(prompt_i, seed=1234 + i)
        imgs.append(img)
    return compose_grid(imgs, cols=cols)


# In[7]:


# NER-backed motif graph
import re
import networkx as nx
import matplotlib.pyplot as plt

NER_PIPE = None
try:
    from transformers import pipeline as hf_pipeline

    NER_PIPE = hf_pipeline(
        task="token-classification",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple",
        device=-1,
    )
    print("🧩 Motif NER: dslim/bert-base-NER loaded.")
except Exception as e:
    print("⚠️ Motif NER unavailable, using lexicon only.", e)
    NER_PIPE = None

MOTIF_LEXICON = {
    "forest","tree","river","sea","ocean","wave","water","rain","storm","cloud","sky","moon","sun","star","night","light","shadow",
    "mirror","glass","window","door","stairs","bridge","room","house","city","desert","mountain","valley","garden","flower",
    "bird","cat","dog","fish","horse","person","child","friend","stranger",
    "clock","time","book","key","phone","car","train","boat","plane",
    "color","red","blue","green","gold","violet","purple","black","white",
    "fire","ice","snow","fog","mist","sand"
}

_token = re.compile(r"[A-Za-z']+")

def _heuristic_motifs(text: str):
    toks = [w.lower() for w in _token.findall(text)]
    return [w for w in toks if w in MOTIF_LEXICON]

def _ner_motifs(text: str):
    if NER_PIPE is None:
        return []
    ents = NER_PIPE(text)
    motifs = []
    for e in ents:
        span = e.get("word", "") or e.get("entity_group", "")
        span = span.strip()
        if not span:
            continue
        parts = [w.lower() for w in _token.findall(span)]
        if span and len(span) <= 24:
            motifs.append(span.lower())
        motifs.extend(parts)
    return [m for m in motifs if len(m) >= 2]

def extract_motifs(text: str, top_k: int = 20):
    ner = _ner_motifs(text)
    lex = _heuristic_motifs(text)
    combined = []
    seen = set()
    for w in ner + lex:
        if w not in seen:
            combined.append(w)
            seen.add(w)
        if len(combined) >= top_k:
            break
    return combined

def build_motif_graph(text: str, window: int = 4, min_weight: int = 1):
    motifs = extract_motifs(text, top_k=40)
    G = nx.Graph()
    for m in motifs:
        G.add_node(m)

    for i in range(len(motifs)):
        for j in range(i + 1, min(i + window, len(motifs))):
            if motifs[i] != motifs[j]:
                a, b = sorted((motifs[i], motifs[j]))
                w = G.get_edge_data(a, b, default={"weight": 0})["weight"] + 1
                G.add_edge(a, b, weight=w)

    if min_weight > 1:
        G.remove_edges_from(
            [(u, v) for u, v, d in G.edges(data=True) if d["weight"] < min_weight]
        )

    return G

def draw_motif_graph(G: nx.Graph, out_path=None):
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No motifs found", ha="center", va="center")
        ax.axis("off")
    else:
        if G.number_of_edges() == 0:
            pos = nx.circular_layout(G)
            nx.draw_networkx_nodes(G, pos, node_color="#91b4ff", node_size=900, ax=ax)
            nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)
            ax.set_title("Motifs (no strong co-occurrence found)")
            ax.axis("off")
        else:
            pos = nx.spring_layout(G, seed=42)
            widths = [1 + d["weight"] for *_u, _v, d in G.edges(data=True)]
            nx.draw(
                G,
                pos,
                with_labels=True,
                node_color="#91b4ff",
                node_size=900,
                width=widths,
                edge_color="#8aa0d6",
                ax=ax,
            )
            ax.axis("off")
    if out_path:
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_path


# In[8]:


from transformers import CLIPProcessor as HFCLIPProcessor, CLIPModel

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

try:
    clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    CLIP_MODEL = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(clip_device)
    CLIP_PROCESSOR = HFCLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    print(f"✅ Loaded CLIP model for semantic alignment: {CLIP_MODEL_ID} on {clip_device}")
except Exception as e:
    CLIP_MODEL = None
    CLIP_PROCESSOR = None
    print("⚠️ CLIP model not available:", e)

def compute_clip_score(prompt: str, image: Image.Image) -> float | None:
    """
    Compute cosine similarity between text and image embeddings using CLIP.
    CLIP only supports sequences up to 77 tokens, so we truncate the text
    to a safe length before encoding.
    """
    if CLIP_MODEL is None or CLIP_PROCESSOR is None:
        return None

    # --- Truncate prompt to avoid >77 token issues ---
    # keep first ~60 words to stay comfortably under the 77-token cap
    truncated_prompt = " ".join(prompt.split()[:60])

    try:
        inputs = CLIP_PROCESSOR(
            text=[truncated_prompt],
            images=[image],
            return_tensors="pt",
            padding=True,
        ).to(clip_device)

        with torch.no_grad():
            out = CLIP_MODEL(**inputs)
            img_emb = out.image_embeds[0]
            txt_emb = out.text_embeds[0]

        img_emb = img_emb / img_emb.norm(p=2)
        txt_emb = txt_emb / txt_emb.norm(p=2)
        score = float((img_emb * txt_emb).sum().cpu().item())
        return score
    except Exception as e:
        print("CLIP score failed:", e)
        return None


# In[9]:


def multimodal_from_text(
    user_text: str,
    project_tag: str = "dream",
    fast: bool = False,
):
    """
    Core multimodal generation:
    - prompt sanitization
    - SD-Turbo image
    - MusicGen / fallback audio
    - CLIP semantic alignment score
    - runtime measurement
    """
    start_time = time.perf_counter()

    clean_prompt, tox_score, softened = sanitize_prompt(user_text)
    stamp = now_stamp()
    base = OUT_DIR / f"{project_tag}_{stamp}"
    base.parent.mkdir(parents=True, exist_ok=True)

    art_prompt = (
        "Dreamlike surreal illustration, cinematic light, ethereal mood, 35mm film look. "
        f"Abstract symbols: {clean_prompt[:500]}"
    )
    img = text_to_image(
        art_prompt,
        steps=3 if fast else 4,
        height=448 if fast else 512,
        width=448 if fast else 512,
    )
    img_path = save_image(img, base.with_suffix(".png"))

    audio_prompt = f"Ambient calm pads, shimmering tones, inspired by: {clean_prompt[:150]}"
    wav_path = text_to_audio(audio_prompt, base.with_suffix(".wav"))

    # CLIP text–image semantic alignment
    clip_score = compute_clip_score(clean_prompt, img)

    end_time = time.perf_counter()
    runtime_sec = end_time - start_time

    report = {
        "timestamp": stamp,
        "input_text": user_text,
        "used_prompt": clean_prompt,
        "toxicity_score": tox_score,
        "softened_prompt": bool(softened),
        "image_model": TXT2IMG_ID,
        "audio_engine": AUDIO_ENGINE,
        "paths": {"image": str(img_path), "audio": str(wav_path)},
        "clip_score": clip_score,
        "runtime_seconds": runtime_sec,
    }
    with open(base.with_suffix(".json"), "w") as f:
        json.dump(report, f, indent=2)
    return report, img


# In[10]:


def multimodal_with_extras(
    user_text: str,
    project_tag: str = "dream",
    mood: bool = True,
    motifs: bool = True,
    fast: bool = False,
):
    report, base_img = multimodal_from_text(user_text, project_tag=project_tag, fast=fast)
    mb_path = None
    g_path = None

    if mood:
        mb = text_to_moodboard(user_text)
        mb_path = OUT_DIR / f"{project_tag}_{now_stamp()}_moodboard.png"
        mb.save(mb_path)

    if motifs:
        G = build_motif_graph(user_text)
        g_path = OUT_DIR / f"{project_tag}_{now_stamp()}_motifs.png"
        draw_motif_graph(G, out_path=g_path)

    report["extras"] = {
        "moodboard": str(mb_path) if mb_path else None,
        "motif_graph": str(g_path) if g_path else None,
    }

    # Persist enriched report next to the main image
    json_path = Path(report["paths"]["image"]).with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


# In[11]:


import faster_whisper
from typing import Optional

WHISPER_DEVICE = "cpu"
WHISPER_MODEL_SIZE = "small"
WHISPER_COMPUTE = "int8"

try:
    ASR_MODEL = faster_whisper.WhisperModel(
        WHISPER_MODEL_SIZE,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE,
    )
    print(f"🎤 Loaded Faster-Whisper: {WHISPER_MODEL_SIZE} on {WHISPER_DEVICE} ({WHISPER_COMPUTE})")
except Exception as e:
    ASR_MODEL = None
    print("Faster-Whisper not available:", e)

def transcribe_audio(audio_path: Optional[str]) -> str:
    """
    Transcribe an audio file path to text using Faster-Whisper.
    Returns an empty string if ASR is unavailable or no audio given.
    """
    if not audio_path or ASR_MODEL is None:
        return ""
    try:
        segments, info = ASR_MODEL.transcribe(
            audio_path,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            beam_size=1,
            best_of=1,
            language=None,  # auto-detect
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text
    except Exception as e:
        print("ASR failed:", e)
        return ""


# In[12]:


def run_all_with_audio(
    text_input,
    audio_input,
    prefer_audio,
    make_mood=True,
    make_motif=True,
    fast=False,
):
    """
    - If 'prefer_audio' is True and a microphone/upload file is provided, use ASR transcript.
    - Otherwise use the text box as usual.
    Returns:
      image_path, audio_path, moodboard_path, motif_path, analysis_text, transcript
    """
    transcript = ""
    if prefer_audio and audio_input:
        transcript = transcribe_audio(audio_input)
        if transcript:
            text_input = transcript

    text_input = (text_input or "").strip()
    if not text_input:
        return (
            None,
            None,
            None,
            None,
            "Please enter or record a dream.",
            transcript,
        )

    meta = multimodal_with_extras(
        text_input,
        project_tag="dream",
        mood=make_mood,
        motifs=make_motif,
        fast=fast,
    )

    tox = meta.get("toxicity_score", 0.0)
    clip_s = meta.get("clip_score", None)
    rt = meta.get("runtime_seconds", None)

    analysis_lines = [f"Toxicity: {tox:.2f}"]
    if clip_s is not None:
        analysis_lines.append(f"CLIP text–image alignment: {clip_s:.3f}")
    else:
        analysis_lines.append("CLIP text–image alignment: N/A")
    if rt is not None:
        analysis_lines.append(f"End-to-end runtime: {rt:.1f} s")

    analysis = "\n".join(analysis_lines)

    return (
        meta["paths"]["image"],
        meta["paths"]["audio"],
        meta["extras"]["moodboard"],
        meta["extras"]["motif_graph"],
        analysis,
        transcript,
    )

