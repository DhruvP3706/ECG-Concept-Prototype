# ECG-Concept-Prototype

Research project on concept-grounded prototype learning for
interpretable 12-lead ECG diagnosis.

## Research Goal

The objective is to develop an interpretable ECG diagnosis framework
that combines:

1. A pretrained LightECGNetV2 ECG representation
2. Clinically meaningful electrophysiological concepts
3. Prototype-based reasoning
4. Faithful concept-level and case-based explanations

## Planned Pipeline

Raw 12-lead ECG
        ↓
Pretrained LightECGNetV2
        ↓
ECG representation
        ↓
Clinical concept layer
        ↓
Concept vector
        ↓
Prototype reasoning
        ↓
Diagnosis + explanation

## Dataset

Development dataset:

A Large-Scale 12-Lead Electrocardiogram Database for Arrhythmia Study

The raw dataset is stored locally and is not included in this repository.

## Pretrained Models

The project uses pretrained LightECGNetV2 checkpoints as the initial
ECG backbone.

The checkpoint files are stored locally and are not included in this
repository.

## Repository Structure

- `data/` — datasets and processed data
- `models/` — pretrained and future trained models
- `src/` — reusable implementation
- `scripts/` — experiment scripts
- `results/` — experiment outputs
- `notebooks/` — exploratory notebooks

## Status

Initial repository setup.
Implementation begins from Step 2.