# Style Transfer Implementation

## Overview

This project implements **Neural Style Transfer** using a VGG19 convolutional neural network. Style transfer is a technique that takes a content image and a style image, then generates a new image that combines the content of the first image with the artistic style of the second image.

**Reference Paper:** [Image Style Transfer Using Convolutional Neural Networks](https://arxiv.org/abs/1508.06576)

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture](#architecture)
3. [Loss Functions](#loss-functions)
4. [Implementation Details](#implementation-details)
5. [Usage](#usage)
6. [Results](#results)
7. [Technical Details](#technical-details)

---

## Introduction

Neural Style Transfer works by:
1. Using a pre-trained CNN (VGG19) to extract features from images
2. Defining two types of losses:
   - **Content Loss**: Preserves the structure and content of the content image
   - **Style Loss**: Captures the artistic style from the style image
3. Optimizing a generated image to minimize both losses simultaneously

The key insight is that different layers of a CNN capture different information:
- **Lower layers** (early conv layers) capture texture, color, and style information
- **Higher layers** (deeper conv layers) capture semantic content and object structure

---

## Architecture

### Model Used
- **VGG19** pre-trained on ImageNet (from `torchvision.models`)
- Only the **feature extraction layers** are used (not the classification head)
- Model is set to evaluation mode with gradients disabled for all parameters

### Layer Selection
- **Content Layers**: `conv_4` (4th convolutional layer)
  - This layer captures high-level content and structure
  
- **Style Layers**: `conv_1`, `conv_2`, `conv_3`, `conv_4`, `conv_5` (first 5 convolutional layers)
  - These layers capture texture, patterns, and artistic style at different scales
  - Style loss is computed as the sum of losses from all these layers

---

## Loss Functions

### 1. Content Loss

The content loss measures how different the feature representations of the generated image are from the content image.

**Formula:**
```
L_content = ||F^C - F^G||²
```

Where:
- `F^C` = Feature map of the content image at a specific layer
- `F^G` = Feature map of the generated image at the same layer
- `||·||²` = Squared L2 norm (Mean Squared Error)

**Implementation:**
- Computed at the `conv_4` layer
- Uses `F.mse_loss()` to calculate the mean squared error between feature maps

### 2. Style Loss

The style loss captures the artistic style by comparing **Gram matrices** of feature representations.

#### Gram Matrix

The Gram matrix captures correlations between different feature channels, which represents texture and style patterns.

**Calculation:**
1. Reshape the feature map from `(1, C, H, W)` to `(C, N)` where `N = H × W`
2. Compute: `G = F · F^T` (matrix multiplication)
3. Normalize by dividing by `(C × H × W)` to prevent bias from layer size

**Formula:**
```
G_ij = (1/(C×H×W)) × Σ(F_i · F_j)
```

**Style Loss:**
```
L_style = ||G^S - G^G||²
```

Where:
- `G^S` = Gram matrix of the style image
- `G^G` = Gram matrix of the generated image
- Computed for each style layer and summed

**Implementation:**
- Computed for layers: `conv_1`, `conv_2`, `conv_3`, `conv_4`, `conv_5`
- Each layer's style loss is summed together
- Uses `F.mse_loss()` between Gram matrices

### 3. Total Loss

The total loss is a weighted combination of content and style losses:

```
L_total = α × L_content + β × L_style
```

Where:
- `α` (content_weight) = 1 (default)
- `β` (style_weight) = 1e4 (default, 10,000)

**Why different weights?**
- Style loss values are typically much smaller than content loss values
- The style weight needs to be much larger to balance their contributions
- This ensures both content and style are preserved in the final image

---

## Implementation Details

### Image Preprocessing

1. **Resize**: Images are resized to a maximum dimension of 512 pixels (maintaining aspect ratio)
2. **Convert to Tensor**: Images are converted to PyTorch tensors
3. **Normalize**: Images are normalized to [0, 1] range
4. **Device**: All tensors are moved to GPU if available

### Optimization

#### L-BFGS Optimizer

The implementation uses **L-BFGS** (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) optimizer, which is a quasi-Newton optimization algorithm.

**Why L-BFGS?**
- More efficient for this type of optimization problem
- Requires fewer iterations than gradient descent
- Better convergence properties for image optimization

**Closure Function:**
L-BFGS needs to evaluate the loss function multiple times per step (for line search). The closure function:
1. Zeros gradients
2. Computes forward pass
3. Computes loss (content + style)
4. Performs backward pass
5. Returns the loss value

**Implementation:**
```python
def closure():
    optimizer.zero_grad()
    loss, c_loss, s_loss = total_loss(...)
    loss.backward()
    return loss

for _ in range(num_steps):
    optimizer.step(closure)
```

#### Constraints

After each optimization step:
- Image pixel values are clamped to [0, 1] range using `target_img.clamp_(0, 1)`
- This ensures the generated image remains a valid image

### Training Parameters

- **Content Weight**: 1.0
- **Style Weight**: 1e4 (10,000)
- **Number of Steps**: 300 iterations
- **Max Image Size**: 512 pixels

### Feature Extraction

The `get_features()` function:
1. Iterates through VGG19 layers
2. Tracks convolutional layers (numbered as `conv_1`, `conv_2`, etc.)
3. Extracts feature maps at specified layers
4. Returns a dictionary of layer names and their feature maps

---

## Usage

### Basic Usage

```python
python style_transfer.py
```

### Input Structure

```
style_transfer/
├── data/
│   ├── art/          # Style images
│   │   ├── art1.jpg
│   │   └── art2.jpg
│   ├── photo/        # Content images
│   │   ├── photo1.jpeg
│   │   └── photo2.jpg
│   └── output/       # Generated images
│       ├── photo1_art1_styled.jpg
│       ├── photo1_art2_styled.jpg
│       ├── photo2_art1_styled.jpg
│       └── photo2_art2_styled.jpg
```

### How It Works

1. **Loads VGG19 model** pre-trained on ImageNet
2. **Processes each photo** with each art style:
   - For each content image (photo)
   - For each style image (art)
   - Performs style transfer
   - Saves result to `data/output/`
3. **Output naming**: `{photo_name}_{art_name}_styled.jpg`

### Custom Usage

To use with your own images:

1. Place content images in `data/photo/`
2. Place style images in `data/art/`
3. Run the script
4. Results will be saved in `data/output/`

---

## Results

The script processes all combinations of photos and art styles, generating styled images. Each output:
- Preserves the content and structure of the original photo
- Applies the artistic style from the art image
- Shows the style transfer effect through texture, color, and pattern matching

### Example Outputs

- `photo1_art1_styled.jpg`: Photo 1 with Art 1 style
- `photo1_art2_styled.jpg`: Photo 1 with Art 2 style
- `photo2_art1_styled.jpg`: Photo 2 with Art 1 style
- `photo2_art2_styled.jpg`: Photo 2 with Art 2 style

---

## Technical Details

### Why VGG19?

- Well-established architecture for style transfer
- Clear layer structure makes feature extraction straightforward
- Pre-trained weights provide good feature representations
- Original paper used VGG19

### Why These Specific Layers?

**Content Layer (`conv_4`):**
- Deep enough to capture semantic content
- Not too deep to lose spatial information
- Standard choice in style transfer literature

**Style Layers (first 5 conv layers):**
- Capture style at multiple scales
- Lower layers: fine textures and colors
- Higher layers: larger patterns and structures
- Summing losses from multiple layers captures comprehensive style

### Gram Matrix Intuition

The Gram matrix measures:
- **Which features activate together** (correlations)
- **Spatial patterns** in the feature maps
- **Texture and style** rather than exact pixel locations

This is why it works for style transfer - style is about patterns and textures, not exact spatial locations.

### Loss Balancing

The style weight (1e4) is much larger than content weight (1) because:
- Style loss values are typically 10^-5 to 10^-3
- Content loss values are typically 10^-2 to 10^-1
- Without balancing, content would dominate and style would be lost

### Optimization Process

1. **Initialize**: Start with the content image (or random noise)
2. **Forward Pass**: Extract features from content, style, and generated images
3. **Compute Loss**: Calculate content and style losses
4. **Backward Pass**: Compute gradients
5. **Update**: L-BFGS optimizer updates the image
6. **Clamp**: Ensure pixel values stay in [0, 1]
7. **Repeat**: For specified number of iterations

### Comparison: L-BFGS vs Adam

**L-BFGS:**
-  Faster convergence (fewer iterations needed)
-  Better quality results
-  More memory efficient for this problem
-  Requires closure function (more complex)

**Adam:**
-  Simpler implementation
-  More iterations may be needed (3000+)
-  May not converge as well
-  Results may be less optimal

The implementation uses L-BFGS as recommended in the original paper.

---

## Hyperparameter Tuning

### Weight Configurations

Different weight combinations produce different results:

1. **High Content Weight, Low Style Weight**:
   - Output looks very similar to content image
   - Minimal style transfer effect

2. **Low Content Weight, High Style Weight**:
   - Output looks very similar to style image
   - Content structure may be lost

3. **Balanced Weights** (default):
   - Good balance between content and style
   - Both are visible in the final image

### Experimenting with Weights

You can modify these in the code:
```python
content_weight = 1      # Try: 0.1, 1, 5, 10
style_weight = 1e4      # Try: 1e3, 1e4, 1e5, 1e6
```

### Number of Iterations

- **300 steps**: Good balance between quality and time
- **More steps**: Better quality but longer training
- **Fewer steps**: Faster but may not converge fully

---

## File Structure

```
style_transfer/
├── style_transfer.py      # Main implementation script
├── 3.ipynb               # Original notebook with code
├── README.MD             # This file
└── data/
    ├── art/              # Style images
    ├── photo/            # Content images
    └── output/           # Generated styled images
```

---

## Dependencies

- `torch` - PyTorch deep learning framework
- `torchvision` - Pre-trained models and transforms
- `PIL` (Pillow) - Image processing
- `matplotlib` - Visualization (optional)
- `numpy` - Numerical operations

---

## Key Concepts Explained

### What is Style Transfer?

Style transfer is the process of taking two images:
- **Content Image**: The subject/structure you want to keep
- **Style Image**: The artistic style you want to apply

And creating a new image that has:
- The **content** (objects, structure) from the first image
- The **style** (texture, colors, patterns) from the second image

### Why Does This Work?

1. **CNNs learn hierarchical features**:
   - Early layers: edges, colors, textures
   - Later layers: objects, scenes, semantic content

2. **Content is captured in deeper layers**:
   - High-level features represent "what" is in the image

3. **Style is captured in correlations**:
   - Gram matrices capture "how" features relate to each other
   - This represents texture and artistic style

4. **Optimization finds the balance**:
   - Minimizing both losses finds an image that satisfies both constraints

### Why Gram Matrices?

- **Spatial invariance**: Style doesn't depend on exact pixel locations
- **Feature correlations**: Style is about which features appear together
- **Texture representation**: Gram matrices naturally capture texture patterns

## Code Structure

### Main Functions

1. **`pre_process(image_path, max_size=512)`**
   - Loads and preprocesses images
   - Resizes to max dimension
   - Converts to tensor

2. **`content_loss(content, target)`**
   - Computes MSE between content features

3. **`gram_matrix(input)`**
   - Computes Gram matrix from feature map
   - Normalizes by spatial dimensions

4. **`style_loss(style, target)`**
   - Computes MSE between Gram matrices

5. **`get_features(model, x, layers)`**
   - Extracts feature maps from specified VGG layers

6. **`total_loss(...)`**
   - Computes weighted sum of content and style losses

7. **`style_transfer(...)`**
   - Main optimization loop
   - Uses L-BFGS optimizer
   - Returns optimized image and loss history

8. **`save_image(tensor, output_path)`**
   - Saves generated image to file

### Main Execution

The `main()` function:
1. Sets up directory paths
2. Finds all art and photo images
3. Loads VGG19 model
4. Processes each combination
5. Saves results

---

