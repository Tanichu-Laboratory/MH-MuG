# MH-MuG: Collaborative Music Generation Game between AI Agents towards Emergent Musical Creativity

[![Paper](https://img.shields.io/badge/Paper-IEEE%20Access-blue)](https://ieeexplore.ieee.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the official implementation of **"MH-MuG: Collaborative Music Generation Game between AI Agents towards Emergent Musical Creativity"** which is now under review.

## Overview
MH-MuG (Metropolis-Hastings Music Generation Game) is a novel multi-agent framework for collaborative symbolic music generation that models the creative process through interactions between AI agents with different musical knowledge. The framework is grounded in the **Systems Model of Creativity** and formulated as a **decentralized Bayesian inference process**.

### Key Features

- **Multi-Agent Collaborative Generation**: Two AI agents (trained on Classical and Jazz) collaborate to create novel music
- **MCMC-Based Interaction**: Uses Metropolis-Hastings algorithm for accept/reject decisions
- **Symbolic Music Generation**: Employs Latent Diffusion Models (LDMs) for high-quality music generation
- **Emergent Communication**: Models music creation as a generative emergent communication process
- **Two Operational Modes**:
  - **w/o fine-tuning**: Fixed-knowledge collaborative game
  - **w/ fine-tuning**: Mutual adaptation through LoRA

## Architecture

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

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU support)
- NVIDIA GPU with at least 24GB VRAM (for training)

### Setup

```bash
# Clone the repository
git clone https://github.com/Tanichu-Laboratory/MH-MuG.git
cd MH-MuG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
diffusers>=0.21.0
pretty_midi
mido
numpy
scipy
librosa
```

## Dataset

We use piano-solo pieces from [MuseScore](https://musescore.com/) in two genres:

All datasets are stored in `dataset/midi`.

### Data Preprocessing

You can preprocess the dataset by running `dataset/midi/midi2npy.ipynb`.

## Training

### Pre-training Individual Agents

```bash
# Move to folder
cd ./train

# Train VAE
python train_vae.py

# Train LDM
python train_diffusion.py

# Train CLIP and ProbVLM
python train_clip_probvlm.py 
```
The YAML files for pre-training each model are located in `./train/config/train/`. Please modify the parameters in each path within the dataset path to match your dataset.

### MH-MuG Collaborative Generation

```bash
# Move to folder
cd ./train

# Train MH-MuG
python train_MH.py
```
You can adjust MH-MuG parameters in `./train/config/train/train_MH.yaml`.

## Generation

Set the model path of the LDM trained in MH-MuG to `pretrained_model_path` in `./train/config/diffusion/diffusion.yaml`, then execute the following command:
```bash
cd ./train
python sampling_diffusion.py
```

After execution, run `dataset/midi/convert2midi.ipynb` on the inference results.

## Evaluation

### Quantitative Metrics

You can evaluate using the metrics in `./dataset/evaluate_audio_metric.ipynb` and `./dataset/evaluate_midi_metric.ipynb`.

**Metrics**:
- **PCHE** (Pitch Class Histogram Entropy): Measures tonal stability
- **GPS** (Grooving Pattern Similarity): Assesses rhythmic stability
- **Diversity**: Measures stylistic variety

### Results Summary

| Model | PCHE | GPS | Diversity ↑ |
|-------|------|-----|------------|
| Original Dataset | 2.929 | 0.610 | 5.587 |
| LDM (Classical + Jazz) | 2.889 | 0.614 | 5.743 |
| MH-MuG (w/o f.t.) Agent A | 2.995 | 0.589 | 4.700 |
| MH-MuG (w/o f.t.) Agent B | 2.988 | 0.593 | 4.917 |
| MH-MuG (w/ f.t.) Agent A | 2.960 | 0.502 | 3.988 |
| MH-MuG (w/ f.t.) Agent B | 2.845 | 0.506 | 4.983 |
| MH-MuG (all acceptance) Agent A | 1.760 | 0.837 | 7.645 |
| MH-MuG (all acceptance) Agent B | 2.953 | 0.606 | 6.162 |

## Audio Examples

You can listen to sample audio clips in https://tanichu-laboratory.github.io/MH-MuG/.

## Algorithm Details

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

## Contributing

We welcome contributions! Please feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This work was supported by JSPS KAKENHI Grant Numbers JP23H04835 and JP21H04904
- Figure 1 was generated using an AI image generation tool
- Manuscript editing was assisted by Gemini 2.5 Pro (Google)
- Data sourced from [MuseScore](https://musescore.com/)
- Text generation using [MU-LLaMA](https://github.com/mumax/mu-llama)

## Authors

- **Koki Sakurai** - Ritsumeikan University
- **Haruto Uenoyama** - Kyoto University
- **Akira Taniguchi** - Ritsumeikan University (IEEE Member)
- **Tadahiro Taniguchi** - Kyoto University (IEEE Member)

**Corresponding author**: Tadahiro Taniguchi (taniguchi@i.kyoto-u.ac.jp)

## Links

- [Project Page](https://tanichu-laboratory.github.io/MH-MuG/)
- [Audio Examples](https://github.com/Tanichu-Laboratory/MH-MuG/tree/main/audio_examples)

## Known Limitations

1. **Dataset Size**: Limited to 100 pieces per genre (proof-of-concept)
2. **Textual Prompts**: "Classical" label is ambiguous, leading to less distinctive evaluation
3. **Closed-Loop Interaction**: Fine-tuning variant shows "siloing" effect with reduced diversity
4. **Embodiment**: Agents lack interoceptive grounding in emotion

## Future Work

- Incorporate interoceptive and affective grounding
- Extend to System 0/1/2/3 multi-timescale agent framework
- Explore open-ended interactions with external environment
- Scale to larger and more diverse musical datasets
- Real-time collaborative generation with human musicians

---

**Last Updated**: November 2024  
**Status**: Research Code - Active Development
