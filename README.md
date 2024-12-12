# DIP_FinalProject

## Overview

### Motivation
When photographing objects behind glass, ambient light or other interference often reflects on the glass surface, which negatively affects image quality and visibility. Removing these reflections, or dereflection, enhances image clarity, enabling more accurate object recognition and improved background segmentation for various applications.

### Method
Inspired by the [Cold-Diffusion](https://github.com/arpitbansal297/Cold-Diffusion-Models) framework, which replaces noise-based destruction in diffusion models with deterministic transformations, we adapt this approach to better suit the requirements of dereflection.

### Project Directory 

- **`reflection-diffusion-pytorch/`**: Our custom implementation for reflection removal.
- **`Cold-Diffusion-Models-main/`**: Original Cold-Diffusion code.
- **`ReflectionSeparation-master/`**: Baseline code for reflection separation.


## Environment
### 1. Use conda (Optimal)
```
conda env create -f requirements.yml 
```

### 2. Use pip
```
pip install -r environment.txt
```

## Usage
Reflection removal is a crucial task in various computer vision and image processing applications. 
Some common applications of reflection removal include:

* **Autonomous driving** : Removing reflections from windshield or window images enhances the clarity and accuracy of scene understanding, ensuring safer navigation.
* **Medical image analysis** : Eliminating reflections from imaging devices or glass barriers improves the visibility of critical details, facilitating more accurate diagnoses and evaluations.
  
## Hyperparameters

### Train

### Test


## Experiment results
