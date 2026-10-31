# Dogs vs Cats Image Classification using a CNN

A convolutional neural network built from scratch in TensorFlow/Keras that classifies images as **cat** or **dog**. The project also walks through how a CNN works: convolution, padding, pooling, flattening, fully connected layers and backpropagation.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/<repo-name>/blob/main/CNN_mini_project.ipynb)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-orange)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [CNN Workflow](#cnn-workflow)
3. [Core Concepts](#core-concepts)
   - [Convolution](#1-convolution)
   - [Padding](#2-padding)
   - [Pooling](#3-pooling)
   - [Flatten and Fully Connected Layers](#4-flatten-and-fully-connected-layers)
   - [Output Layer](#5-output-layer)
   - [Backpropagation in a CNN](#6-backpropagation-in-a-cnn)
4. [Model Architecture](#model-architecture)
5. [Results](#results)
6. [Project Structure](#project-structure)
7. [How to Run](#how-to-run)
8. [Possible Improvements](#possible-improvements)

---

## Project Overview

| Item | Details |
|---|---|
| Task | Binary image classification (cat vs dog) |
| Dataset | [Dogs vs Cats on Kaggle](https://www.kaggle.com/datasets/salader/dogsvscats) (`salader/dogsvscats`) |
| Data split | 20,000 training images, 5,000 validation images |
| Input | 256 × 256 RGB images, pixel values scaled to [0, 1] |
| Model | 4 convolution blocks + 5 dense layers, 17.2M parameters |
| Loss / Optimizer | Binary cross-entropy / Adam |
| Validation accuracy (20 epochs) | **90.06%** |

---

## CNN Workflow

![CNN workflow](assets/cnn_workflow.png)

```
Input -> Convolution -> MaxPooling -> Flatten -> Fully Connected Layers -> Output
```

| Stage | What it does | In this project |
|---|---|---|
| **Input** | Receives the raw image as a 3D array (height × width × colour channels) | 256 × 256 × 3 |
| **Convolution** | Slides small filters over the image to detect local patterns (edges, textures, shapes) and produces feature maps | 4 `Conv2D` layers, 3×3 filters, 32 → 64 → 128 → 256 filters, ReLU |
| **MaxPooling** | Down-samples each feature map, keeping the strongest response in every window | 2×2 window, stride 2, after every convolution |
| **Flatten** | Converts the final 3D feature maps into a 1D vector | 16 × 16 × 256 → 65,536 values |
| **Fully Connected** | Combines the extracted features to make a decision | Dense 256 → 128 → 64 → 32 → 16 |
| **Output** | Produces the final prediction | 1 neuron with sigmoid: probability the image is a dog |

The first half (Convolution + Pooling) is **feature extraction**. The second half (Flatten + Dense + Output) is **classification**.

---

## Core Concepts

### 1. Convolution

A convolution layer holds a set of small learnable matrices called **filters** (or kernels). Each filter slides across the image; at every position it multiplies its values with the image patch underneath and sums the result into a single number. Repeating this over the whole image builds a **feature map**.

![Convolution and pooling example](assets/conv_pool_example.png)

In the example above, the 3×3 filter is a vertical-edge detector. Its top-left output is computed as:

```
window = [[1, 2, 0],      filter = [[ 1, 0, -1],
          [0, 1, 3],                [ 1, 0, -1],
          [2, 1, 0]]                [ 1, 0, -1]]

(1·1 + 2·0 + 0·-1) + (0·1 + 1·0 + 3·-1) + (2·1 + 1·0 + 0·-1) = 1 - 3 + 2 = 0
```

Key ideas:

- **Weight sharing:** the same filter is reused at every position, so a conv layer needs far fewer parameters than a dense layer on the same image. The first layer here has only `3·3·3·32 + 32 = 896` parameters.
- **Local connectivity:** each output value depends only on a small neighbourhood of the input.
- **Depth of filters:** the number of filters sets how many feature maps a layer produces (32, 64, 128, 256 here). Early layers learn simple patterns such as edges; deeper layers combine them into more complex shapes.
- **Activation (ReLU):** after the convolution, `ReLU(x) = max(0, x)` adds non-linearity, which lets the network learn complex patterns.
- **Stride:** how many pixels the filter moves each step (stride 1 is used in every conv layer).

Output size of a convolution:

$$O = \left\lfloor \frac{N + 2P - F}{S} \right\rfloor + 1$$

where `N` is input size, `F` filter size, `P` padding and `S` stride.

### 2. Padding

Without padding, a convolution shrinks the feature map and the border pixels are used less often than the centre ones. **Padding** adds a border of zeros around the input to control this.

| Padding | Keras option | Behaviour | Example (N=256, F=3, S=1) |
|---|---|---|---|
| Valid | `padding='valid'` | No padding; output shrinks | (256 − 3) + 1 = **254** |
| Same | `padding='same'` | Zeros added so output size = input size (for stride 1) | **256** |

In this project most convolutions use `'same'` padding so spatial detail is preserved until pooling reduces it. The second convolution uses `'valid'` (feature map 128 → 126). An earlier version with `'valid'` in every layer gave worse results.

### 3. Pooling

A pooling layer down-samples each feature map independently. **Max pooling** takes the largest value in each window, which:

- cuts the size of the feature maps (less computation, fewer parameters downstream),
- keeps the strongest activation in each region,
- makes the network more tolerant to small shifts of an object in the image,
- has **no learnable parameters**.

With a 2×2 window and stride 2, each spatial dimension is halved. This project uses `padding='same'` in pooling, so odd sizes are rounded up (63 → 32).

| Layer | Feature map after Conv | After MaxPool |
|---|---|---|
| Block 1 | 256 × 256 × 32 | 128 × 128 × 32 |
| Block 2 | 126 × 126 × 64 | 63 × 63 × 64 |
| Block 3 | 63 × 63 × 128 | 32 × 32 × 128 |
| Block 4 | 32 × 32 × 256 | 16 × 16 × 256 |

### 4. Flatten and Fully Connected Layers

**Flatten** reshapes the last pooled output (16 × 16 × 256) into a single vector of 65,536 values. No computation happens here, only a change of shape.

The **fully connected (Dense)** layers then take this vector of learned features and combine them. Every neuron is connected to every neuron in the previous layer: `output = activation(W · input + b)`. The widths shrink gradually (256 → 128 → 64 → 32 → 16) to compress the features into a final decision. Batch normalization after most layers stabilises and speeds up training.

### 5. Output Layer

A single neuron with a **sigmoid** activation outputs a probability between 0 and 1:

- `p >= 0.5` → **Dog**
- `p < 0.5` → **Cat**

The loss is **binary cross-entropy**:

$$L = -\big[\,y\log(p) + (1-y)\log(1-p)\,\big]$$

where `y` is the true label (1 = dog, 0 = cat) and `p` the predicted probability.

### 6. Backpropagation in a CNN

Training repeats two passes for every batch:

1. **Forward pass:** the batch flows through Conv → Pool → Flatten → Dense → Output, and the loss is computed.
2. **Backward pass (backpropagation):** the gradient of the loss with respect to every weight is computed using the chain rule, moving from the output back to the input. The optimizer (Adam) then updates the weights to reduce the loss.

Weight update (plain gradient descent form): `w ← w − η · ∂L/∂w`, where `η` is the learning rate. Adam adapts this step size per parameter.

How the gradient travels backward through each layer type:

| Layer | Backward behaviour |
|---|---|
| **Output (sigmoid + BCE)** | The gradient at the output pre-activation simplifies to `p − y` |
| **Dense** | `∂L/∂W = aᵀ · δ` (input activations × incoming gradient); the gradient passed back is `δ · Wᵀ` |
| **Flatten** | The 1D gradient is simply reshaped back to 16 × 16 × 256 |
| **MaxPooling** | The gradient is sent **only to the position that held the maximum** in each window; all other positions get 0 |
| **ReLU** | The gradient passes where the input was positive and is blocked (0) where it was negative |
| **Convolution** | **Filter gradient:** the input patches are correlated with the incoming gradient, and contributions from all positions are **summed** (because the filter is shared). **Input gradient:** the incoming gradient is convolved with the 180°-rotated filter, passing the error to the previous layer. Bias gradient is the sum of the incoming gradient |

In short: the error is computed at the output, reshaped through Flatten, routed through the max positions in Pooling, masked by ReLU, and distributed through the filters in Convolution, giving every filter weight the gradient it needs to improve. Keras performs all of this automatically inside `model.fit()`.

---

## Model Architecture

| Layer | Output Shape | Parameters |
|---|---|---|
| Input | 256 × 256 × 3 | 0 |
| Conv2D (32, 3×3, same) + BatchNorm | 256 × 256 × 32 | 1,024 |
| MaxPooling2D (2×2) | 128 × 128 × 32 | 0 |
| Conv2D (64, 3×3, valid) + BatchNorm | 126 × 126 × 64 | 18,752 |
| MaxPooling2D (2×2) | 63 × 63 × 64 | 0 |
| Conv2D (128, 3×3, same) + BatchNorm | 63 × 63 × 128 | 74,368 |
| MaxPooling2D (2×2) | 32 × 32 × 128 | 0 |
| Conv2D (256, 3×3, same) + BatchNorm | 32 × 32 × 256 | 296,192 |
| MaxPooling2D (2×2) | 16 × 16 × 256 | 0 |
| Flatten | 65,536 | 0 |
| Dense 256 + BatchNorm | 256 | 16,778,496 |
| Dense 128 + BatchNorm | 128 | 33,408 |
| Dense 64 + BatchNorm | 64 | 8,512 |
| Dense 32 + BatchNorm | 32 | 2,208 |
| Dense 16 | 16 | 528 |
| Dense 1 (sigmoid) | 1 | 17 |
| **Total** | | **17,213,505** |

About 97% of the parameters sit in the first dense layer after Flatten, which is a common source of overfitting in simple CNNs.

---

## Results

Trained for 20 epochs (batch size 32) on a Colab T4 GPU, about 65 seconds per epoch.

| Metric | Train | Validation |
|---|---|---|
| Accuracy (epoch 20) | 99.40% | **90.06%** |
| Loss (epoch 20) | 0.0186 | 0.4366 |

![Training curves](assets/training_curves.png)

**Observations**

- Validation accuracy rises from 69.2% (epoch 1) to 90.1% (epoch 20) but is noisy, with dips such as 83.6% at epoch 16.
- Training accuracy approaches 100% while validation loss stops improving after epoch 13 (lowest value 0.3748), so the model is **overfitting**. Early stopping around epochs 13–15 would give a lower validation loss.

**Sample prediction**

![Sample prediction](assets/sample_prediction.png)

---

## Project Structure

```
.
├── CNN_mini_project.ipynb   # Full notebook: data, model, training, prediction
├── README.md
├── requirements.txt
├── .gitignore
└── assets/                  # Images used in this README
```

---

## How to Run

### Option 1: Google Colab (recommended, GPU available)

1. Click the **Open in Colab** badge at the top and set the runtime to **GPU**.
2. Create a Kaggle API token (Kaggle → Settings → API → *Create New Token*) to get `kaggle.json`.
3. Run the notebook cells in order and upload `kaggle.json` when prompted. The dataset is downloaded automatically.

### Option 2: Local machine

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt

# Download and extract the dataset (kaggle.json must be in ~/.kaggle/)
kaggle datasets download -d salader/dogsvscats
unzip dogsvscats.zip
```

The notebook's data-setup cells use Colab paths (`/content/...`) and `google.colab`; replace them with local paths when running outside Colab. Then open `CNN_mini_project.ipynb` in Jupyter.

> Never commit `kaggle.json`. It is already listed in `.gitignore`.

---

## Possible Improvements

- **Data augmentation** (random flips, rotations, zoom) to reduce overfitting
- **Dropout** in the dense layers and **early stopping** on validation loss
- Replace `Flatten` + large dense layer with **GlobalAveragePooling2D** to cut parameters sharply
- **Transfer learning** with a pretrained backbone (e.g. MobileNetV2, ResNet50)
- Evaluate with a confusion matrix, precision and recall, and keep a separate test set
- Use `padding='same'` consistently across all convolution layers
