# 🌌 DreamScape AI — From Pitch to Prototype  
### Deliverable 1 · Foundation and Design Blueprint  
**Author:** Deepika Sarala Pratapa  |  **Course:** EEE 6778 – Applied Machine Learning II (University of Florida)

---

## 🪄 Project Overview
DreamScape AI transforms dream diary text into **immersive multimodal representations** —  
visual moodboards, ambient soundscapes, and motif graphs — blending psychology, art, and machine learning.  
It allows both researchers and individuals to explore subconscious themes through AI-generated imagery and sound.

---

## 🎯 Objectives
- Load and preprocess the **DreamBank Annotated** dataset from Hugging Face.  
- Analyze recurring **emotions, themes, and symbols** in dream narratives.  
- Prepare cleaned datasets and visual statistics for model development.  
- Establish the architecture connecting **NLP → Diffusion → Text-to-Music → UI**.  

---

## 🧱 Repository Structure
```
dreamscape-ai/
│
├── data/
│   ├── raw/                         # Original dream data (from DreamBank Annotated)
│   │    └── README.md               # Note dataset source + license info
│   └── processed/
│        └── dreambank_clean.json    # Cleaned dream dataset after EDA
│
├── notebooks/
│   ├── setup.ipynb      #(dataset load + plot + verification)
│ 
│   ├── exploratory_data_analysis.ipynb  # Loads DreamBank, cleans text, plots stats
│                                   
│   ├── nlp_motif_extraction.ipynb   # Transformer-based motif/emotion extraction
│                                   
│   ├── multimodal_generation.ipynb  # Text→Image & Text→Audio generation
│                                   
│   └── visualization_summary.ipynb   # Final plots, heatmaps, summaries
│                                  
│
├── src/
│   ├── preprocess.py                # Helper functions for cleaning, tokenizing
│   ├── nlp_utils.py                 # Keyword & emotion extraction helpers
│   └── visualization_utils.py       # Plotting utilities (themes, emotions, motifs)
│
├── results/
│   ├── eda_plots.png                # Barplots from EDA
│   ├── emotion_distribution.png     # Emotion frequency chart
│   ├── motif_network.html           # Interactive motif graph (optional)
│   └── summary_tables.csv           # Statistical summaries
│
├── docs/
│   ├── diagram.png                  # Data→Model→Visualization architecture diagram
│   ├── technical_blueprint.pdf      # 3–5-page design document (Canvas upload)
│   └── ui_wireframe.png             # static mockup of dashboard layout
│
├── requirements.txt                 # Verified library list (for pip install)
├── environment.yml                  # Optional conda version of dependencies
├── README.md                        # Full project overview + setup guide
└── LICENSE                          # MIT or CC-BY license for your code
```
---

## 🧠 Dataset
**Name:** [DreamBank Annotated – Gustave Cortal (2023)](https://huggingface.co/datasets/gustavecortal/DreamBank-annotated)  
**Type:** Text dataset of dream reports with annotations for characters and emotions  
**Size:** ~28,000 dreams across 400+ individuals  

**Data Fields:**
| Column | Description |
|--------|--------------|
| `id` | Dreamer identifier |
| `name` | Source / dreamer label |
| `number` | Dream index |
| `time`, `date` | Period or timestamp |
| `gender`, `age` | Demographic metadata |
| `report` | Full dream text |
| `character`, `emotion` | HVdC character and affect tags |

**Preprocessing Summary**
- Removed null or short dream texts (< 50 chars)  
- Normalized column names (`report → text`)  
- Parsed HVdC-style character codes (e.g., `1MSA`, `2FSA`)  
- Extracted affect tags (e.g., `AN`, `SD`, `JO`)  
- Computed text length, token counts, and summary statistics  
- Saved cleaned output as `data/processed/dreambank_clean.json`

---

## ⚙️ Installation & Setup

### 1️⃣ Create Environment (Recommended: Conda)
```bash
conda create -n dreamscape python=3.11 -y
conda activate dreamscape
pip install -r requirements.txt
```

2️⃣ (Optional) Register as Jupyter Kernel
```
pip install jupyterlab ipykernel
python -m ipykernel install --user --name dreamscape --display-name "DreamScape AI"
```

3️⃣ Verify Installation
Open Jupyter and run:
```
import pandas, datasets, matplotlib, seaborn, transformers
print("Environment ready!")
```
▶️ How to Run
	1.	Launch Jupyter Lab or Notebook:
  ```
  jupyter lab
  jupyter notebook
```
	2.	Open notebooks/exploratory_data_analysis.ipynb
	3.	Run all cells top to bottom:
	•	Loads DreamBank Annotated dataset from Hugging Face
	•	Cleans and processes dream texts
	•	Generates exploratory plots (themes, emotions, dream length)
	•	Saves processed JSON to data/processed/

Example output:
	•	dreambank_clean.json – structured dataset for NLP modeling
	•	eda_plots.png – top themes/emotions
	•	summary_tables.csv – descriptive stats

  🧩 Planned System Architecture
  Flow Overview:
  Dream Text → NLP Extraction (Transformers)
           → Visual Generation (Diffusion Models)
           → Audio Generation (Text-to-Music)
           → Visualization (EDA + Motif Graphs)

| **Stage** | **Method** | **Output** |
|------------|-------------|-------------|
| **Text Understanding** | BERT / DistilBERT | Motifs, emotions, symbolic elements |
| **Visual Synthesis** | Stable Diffusion / FLUX | Surreal moodboard images |
| **Audio Generation** | MusicGen / AudioLDM | Ambient dream soundscapes |
| **Visualization** | matplotlib / seaborn / networkx | Thematic & motif graphs |

💡 Innovation & Significance

DreamScape AI bridges psychology and AI by translating dream narratives into tangible, multimodal artifacts.

Key Contributions
	•	Introduces a computational approach to analyzing subconscious patterns.
	•	Integrates NLP, computer vision, and generative audio pipelines.
	•	Serves dual audiences: researchers studying dream themes and individuals exploring self-reflection through art.

📆 **Implementation Timeline (Oct – Dec 2025)**

| **Week** | **Focus** | **Expected Output** |
|-----------|------------|---------------------|
| **Oct 20 – 26** | Data cleaning, baseline setup | Working data loader and first notebook |
| **Oct 27 – Nov 2** | Baseline training, simple UI prototype | Model trains end-to-end, early interface running |
| **Nov 3 – 16** | Model tuning and interpretability | Improved accuracy and analytical insights |
| **Nov 17 – 30** | Interface integration and refinement | Interactive and stable user interface |
| **Dec 1 – 11** | Demo and final report | Completed project, poster and presentation ready |

⚖️ Responsible AI Reflection
	•	Data Ethics: DreamBank dataset is public (CC BY 4.0) and anonymized.
	•	Transparency: All generative outputs labeled as AI-created.
	•	Sustainability: Pipeline optimized for CPU fallback to reduce compute footprint.
	•	Interpretability: Visual and textual analyses included for transparency in theme clustering.

  📚 References
	•	Hall, C., & Van de Castle, R. (1966). The Content Analysis of Dreams. Appleton-Century-Crofts.
	•	Cortal, G. (2023). DreamBank Annotated [Dataset]. Hugging Face.
	•	Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models.
	•	Copet et al. (2023). MusicGen: Simple and Controllable Music Generation.
  
🧾 License

All original code © 2025 Deepika Sarala Pratapa — released under the MIT License.
DreamBank Annotated dataset © Gustave Cortal (2023) — CC BY 4.0.


DreamScape AI doesn’t just analyze dreams — it brings them to life. 🌠


## 🧑‍💻 Author

**Deepika Sarala Pratapa**  
MS Applied Data Science @ UF
 [deepikapratapa27@gmail.com](mailto:dpratapa@ufl.edu)  

