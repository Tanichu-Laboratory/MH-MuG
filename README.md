# MH-MuG: Collaborative Music Generation Game between AI Agents towards Emergent Musical Creativity

[![Paper](https://img.shields.io/badge/Paper-IEEE%20Access-blue)](https://ieeexplore.ieee.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the official implementation of **"MH-MuG: Collaborative Music Generation Game between AI Agents towards Emergent Musical Creativity"** published in IEEE Access.

## 📖 Overview

MH-MuG (Metropolis-Hastings Music Generation Game) is a novel multi-agent framework for collaborative symbolic music generation that models the creative process through interactions between AI agents with different musical knowledge. The framework is grounded in the **Systems Model of Creativity** and formulated as a **decentralized Bayesian inference process**.

### Key Features

- 🎵 **Multi-Agent Collaborative Generation**: Two AI agents (trained on Classical and Jazz) collaborate to create novel music
- 🔄 **MCMC-Based Interaction**: Uses Metropolis-Hastings algorithm for accept/reject decisions
- 🎼 **Symbolic Music Generation**: Employs Latent Diffusion Models (LDMs) for high-quality music generation
- 🧠 **Emergent Communication**: Models music creation as a generative emergent communication process
- 🎯 **Two Operational Modes**:
  - **w/o fine-tuning**: Fixed-knowledge collaborative game
  - **w/ fine-tuning**: Mutual adaptation through LoRA

## 🏗️ Architecture

The system consists of:

1. **Latent Diffusion Models (LDMs)**: Based on Diffusion Transformers (DiT-XL/8) with VAE
2. **ProbVLM**: Probabilistic Vision-Language Model for accept/reject judgment
3. **CLIP**: Text and image encoding for conditional generation
4. **LoRA**: Parameter-efficient fine-tuning for adaptation (w/ f.t. variant)

### Graphical Model

```
      w (musical work)
     / \
    /   \
  z^A   z^B  (latent representations)
   |     |
  o^A   o^B  (text observations)
```
## 📈 Evaluation

### Quantitative Metrics

```bash
# Evaluate generated music
python evaluate.py \
  --generated_dir generated_samples \
  --reference_dir data/test \
  --metrics pche gps diversity
```

**Metrics**:
- **PCHE** (Pitch Class Histogram Entropy): Measures tonal stability
- **GPS** (Grooving Pattern Similarity): Assesses rhythmic stability
- **Diversity**: Measures stylistic variety

### Results Summary

| Model | PCHE | GPS | Diversity ↑ |
|-------|------|-----|------------|
| Original Dataset | 2.929 | 0.610 | 5.587 |
| LDM (Classical + Jazz) | 2.889 | 0.614 | 5.743 |
| **MH-MuG (w/o f.t.) Agent A** | **2.995** | **0.589** | **4.700** |
| **MH-MuG (w/o f.t.) Agent B** | **2.988** | **0.593** | **4.917** |
| MH-MuG (w/ f.t.) Agent A | 2.960 | 0.502 | 3.988 |
| MH-MuG (w/ f.t.) Agent B | 2.845 | 0.506 | 4.983 |

## 🎧 Audio Examples

Generated music samples are available in the `audio_examples/` directory:

- `audio_examples/classical/` - Baseline Classical LDM outputs
- `audio_examples/jazz/` - Baseline Jazz LDM outputs
- `audio_examples/mhmug_wo_ft/` - MH-MuG without fine-tuning
- `audio_examples/mhmug_w_ft/` - MH-MuG with fine-tuning

## 📁 Project Structure

```
MH-MuG/
├── configs/              # Configuration files
│   ├── vae_config.yaml
│   ├── ldm_classical.yaml
│   ├── ldm_jazz.yaml
│   └── probvlm_config.yaml
├── data/                 # Dataset directory
│   ├── classical/
│   └── jazz/
├── models/               # Model implementations
│   ├── vae.py
│   ├── dit.py            # Diffusion Transformer
│   ├── ldm.py            # Latent Diffusion Model
│   └── probvlm.py        # Probabilistic VLM
├── scripts/              # Utility scripts
│   ├── prepare_data.py
│   └── evaluate.py
├── train_vae.py          # VAE training script
├── train_ldm.py          # LDM training script
├── train_probvlm.py      # ProbVLM training script
├── run_mhmug.py          # Main MH-MuG execution
├── generate.py           # Generation script
├── evaluate.py           # Evaluation script
└── README.md
```

## 🔬 Algorithm Details

### MH-MuG Process

1. **Role Assignment**: Assign composer and listener roles to agents
2. **Composition**: Composer generates music conditioned on text
3. **Appreciation**: Listener evaluates using acceptance probability:
   ```
   r = min(1, exp(L_co - L_li))
   ```
4. **Decision**: Accept or reject based on Metropolis-Hastings criterion
5. **Learning** (w/ f.t. only): Fine-tune listener's model using LoRA
6. **Role Switch**: Swap composer and listener roles

### Acceptance Probability Calculation

The log-likelihood L* is computed using ProbVLM's Generalized Normal distribution:

```
L* = (μ_w* - z_c / α_w*)^β_w* - log(β_w* / α_w*) + log Γ(1 / β_w*)
```

## 🙏 Acknowledgments

- This work was supported by JSPS KAKENHI Grant Numbers JP23H04835 and JP21H04904
- Figure 1 was generated using an AI image generation tool
- Manuscript editing was assisted by Gemini 2.5 Pro (Google)
- Data sourced from [MuseScore](https://musescore.com/)
- Text generation using [MU-LLaMA](https://github.com/mumax/mu-llama)

## 👥 Authors

- **Koki Sakurai** - Ritsumeikan University
- **Haruto Uenoyama** - Kyoto University
- **Akira Taniguchi** - Ritsumeikan University (IEEE Member)
- **Tadahiro Taniguchi** - Kyoto University (IEEE Member)

**Corresponding author**: Tadahiro Taniguchi (taniguchi@i.kyoto-u.ac.jp)

## ⚠️ Known Limitations

1. **Dataset Size**: Limited to 100 pieces per genre (proof-of-concept)
2. **Textual Prompts**: "Classical" label is ambiguous, leading to less distinctive evaluation
3. **Closed-Loop Interaction**: Fine-tuning variant shows "siloing" effect with reduced diversity
4. **Embodiment**: Agents lack interoceptive grounding in emotion

## 🔮 Future Work

- Incorporate interoceptive and affective grounding
- Extend to System 0/1/2/3 multi-timescale agent framework
- Explore open-ended interactions with external environment
- Scale to larger and more diverse musical datasets
- Real-time collaborative generation with human musicians

---

**Last Updated**: 10 November 2024  

