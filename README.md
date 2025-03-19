# Vision and Text Transformers with TensorFlow

This project implements Transformer models for both Natural Language Processing (NLP) and Computer Vision tasks using TensorFlow and the Hugging Face transformers library. It covers training a DistilBERT model for sentiment analysis on the IMDB dataset and a Vision Transformer (ViT) for image classification on the CIFAR-10 dataset.

## Installation

Ensure you have Python 3.6+ and the following libraries installed:

pip install tensorflow transformers numpy tqdm scikit-learn matplotlib seaborn

## Usage

Run the script to train the models. The models will be automatically saved after training:

python train_transformers.py

## Overview

### IMDB Sentiment Analysis

- **Model:** DistilBERT
- **Dataset:** IMDB movie reviews
- **Tokenization:** Converts each review into token IDs using the DistilBERT tokenizer.

### CIFAR-10 Image Classification

- **Model:** Vision Transformer (ViT)
- **Dataset:** CIFAR-10 image dataset
- **Patch Encoding:** Each image is divided into patches, which are then linearly embedded.

## Key Differences in Text vs. Image Processing

### Text (NLP):

- **Tokenization:** Text is split into tokens or words.
- **Sequence Handling:** Ordered sequences with possible truncation or padding.
- **Positional Embedding:** Necessary to maintain input sequence order.

### Images (Vision):

- **Patch Division:** Images are divided into non-overlapping patches.
- **Flattening:** Patches are flattened and treated as tokens.
- **Embedding and Attention:** Similar to NLP but adapted for 2D spatial contexts.

## Self-Attention Adaptation

The self-attention mechanism adapts to different data modalities by adjusting the way it captures relationships:

- **For Text:** Self-attention captures semantic relationships and long-distance dependencies between words.
- **For Images:** Self-attention focuses on spatial relationships between patches, allowing the model to capture features relevant to both local and global contexts in the image.

## Limitations and Mitigations

### Limitations:

- **Compute Requirements:** Transformers, particularly vision models, can be computationally intensive.
- **Data Efficiency:** Large amounts of data are typically required for effective training.
- **Lack of Inductive Biases:** Unlike CNNs, ViTs lack certain spatial inductive biases, which can make training data-intensive.

### Mitigations:

- **Data Augmentation:** In the vision domain, augmenting training data increases robustness.
- **Pre-training:** Leveraging pre-trained models or transfer learning can reduce the amount of task-specific data needed.
- **Hybrid Models:** Combining CNNs with transformers can offer the benefits of both architectures.

---

## Results

- **IMDB Sentiment Analysis:** The DistilBERT model achieved satisfactory accuracy, demonstrating its ability to process sequential linguistic data effectively.

![Capture d’écran 2025-03-19 113142](https://github.com/user-attachments/assets/5a15a161-11c2-4250-b1fc-d42e19aa34c7)

- **CIFAR-10 Image Classification:** The ViT model achieved competitive accuracy, showcasing the potential of transformers in image classification tasks.

![image](https://github.com/user-attachments/assets/67bdd3d7-f92e-4c9f-b112-d85742798e82)

---

## Contributions

Contributions and improvements to the project are welcome! If you encounter any issues or have suggestions, please open an issue or submit a pull request.
