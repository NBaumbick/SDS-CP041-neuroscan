# NeuroScan: Brain Tumor Classifier

A production-deployed deep learning classifier that analyzes MRI scans and predicts the presence or absence of a brain tumor, returning a classification with a calibrated confidence score.

**Live demo:** [huggingface.co/spaces/NBaumbick/Neuroscan](https://huggingface.co/spaces/NBaumbick/Neuroscan)

---

## Overview

NeuroScan uses a custom convolutional neural network trained on MRI brain scan images to perform binary classification: **Tumor Detected** or **No Tumor Detected**. The model is served via a Dockerized application hosted on Hugging Face Spaces, with a simple upload interface that returns a prediction and confidence level in real time.

This project covers the full ML lifecycle: data preprocessing, custom model architecture design, training and evaluation, containerization, and production deployment.

---

## Demo

1. Navigate to the [live Space](https://huggingface.co/spaces/NBaumbick/Neuroscan)
2. Upload an MRI brain scan image
3. Receive a binary classification (Tumor / No Tumor) with a confidence score

---

## Model

**Architecture:** Custom CNN built with Keras  
**Framework:** TensorFlow / Keras  
**Task:** Binary image classification  
**Input:** MRI brain scan image  
**Output:** Predicted class (Tumor / No Tumor) + confidence score  

The architecture was designed from scratch rather than relying on a pretrained backbone, with convolutional blocks tuned specifically for the structural patterns present in MRI scan data.

### Evaluation Results (held-out test set, n=41)

| Metric | Score |
|---|---|
| Accuracy | 95.1% (39/41 correct) |
| Precision | 96.3% (1 false positive) |
| Recall | 96.3% (caught 26 of 27 tumors) |
| Specificity | 92.9% (13 of 14 healthy scans correctly identified) |

The model is optimized to minimize false negatives given the clinical context. Missing a tumor (false negative) carries a higher cost than a false positive, and the recall score reflects that priority.

---

## Project Structure

```
neuroscan/
├── app.py                  # Gradio application entry point
├── model/
│   └── neuroscan_model.h5  # Trained Keras model weights
├── src/
│   ├── preprocess.py       # Image preprocessing pipeline
│   └── predict.py          # Inference logic
├── notebooks/
│   └── training.ipynb      # Model training and evaluation notebook
├── Dockerfile              # Container definition for HF Spaces deployment
└── requirements.txt
```

---

## Deployment

The application is containerized with Docker and deployed on Hugging Face Spaces.

To run locally:

```bash
docker pull nbaumbick/neuroscan
docker run -p 7860:7860 nbaumbick/neuroscan
```

Then open `http://localhost:7860` in your browser.

---

## Tech Stack

- Python, TensorFlow, Keras
- Gradio (inference UI)
- Docker
- Hugging Face Spaces

## Acknowledgments

Built as part of the SuperDataScience Community Project (SDS-CP041). Extended with custom architecture design, Docker containerization, and production deployment on Hugging Face Spaces.
