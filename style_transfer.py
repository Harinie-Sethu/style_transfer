import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def pre_process(image_path, max_size=512):
    image = Image.open(image_path).convert("RGB")
    size = min(max_size, max(image.size))
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)
    return image.to(device)

def display(tensor):
    image = tensor.cpu().clone().detach().squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    plt.axis("off")
    plt.show()

def save_image(tensor, output_path):
    image = tensor.cpu().clone().detach().squeeze(0)
    image = transforms.ToPILImage()(image)
    image.save(output_path)
    print(f"Saved output to: {output_path}")

def content_loss(content, target):
    return F.mse_loss(content, target)

def gram_matrix(input):
    b, c, h, w = input.size()
    features = input.view(c, h * w)
    gram = torch.mm(features, features.t())
    return gram / (c * h * w)

def style_loss(style, target):
    return F.mse_loss(gram_matrix(style), gram_matrix(target))

def get_features(model, x, layers):
    features = {}
    i = 0

    for layer in model.children():
        x = layer(x)

        if isinstance(layer, nn.Conv2d):
            i += 1
            name = "conv_" + str(i)
            if name in layers:
                features[name] = x

    return features

def total_loss(target_img, content_features, style_grams, model, c_layers, s_layers, c_weight, s_weight):
    layers = c_layers + s_layers
    target_features = get_features(model, target_img, layers)

    c_loss = sum(content_loss(target_features[layer], content_features[layer]) for layer in c_layers)
    s_loss = sum(style_loss(target_features[layer], style_grams[layer]) for layer in s_layers)

    return c_weight * c_loss + s_weight * s_loss, c_loss, s_loss

def style_transfer(content_img, style_img, vgg, content_layers, style_layers, content_weight, style_weight, num_steps):
    content_img = content_img.to(device)
    style_img = style_img.to(device)
    target_img = content_img.clone().requires_grad_(True).to(device)

    content_features = get_features(vgg, content_img, content_layers)
    style_features = get_features(vgg, style_img, style_layers)
    style_features_list = {layer: style_features[layer] for layer in style_layers}

    optimizer = optim.LBFGS([target_img])
    content_losses, style_losses, total_losses = [], [], []
    step = 0

    def closure():
        nonlocal step
        optimizer.zero_grad()
        loss, c_loss, s_loss = total_loss(target_img, content_features, style_features_list, vgg, content_layers, style_layers, content_weight, style_weight)
        loss.backward()

        content_losses.append(c_loss.item())
        style_losses.append(s_loss.item())
        total_losses.append(loss.item())

        step += 1

        if step % 50 == 0:
            print(f'Step {step}: Content Loss: {c_loss.item():.6f}, Style Loss: {s_loss.item():.6f}, Total Loss: {loss.item():.6f}')

        return loss

    for _ in range(num_steps):
        optimizer.step(closure)
        with torch.no_grad():
            target_img.clamp_(0, 1)

    return target_img, content_losses, style_losses, total_losses

def main():
    # Setup paths
    base_dir = Path(__file__).parent
    art_dir = base_dir / "data" / "art"
    photo_dir = base_dir / "data" / "photo"
    output_dir = base_dir / "data" / "output"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all art and photo files
    art_files = sorted(list(art_dir.glob("*.jpg")) + list(art_dir.glob("*.jpeg")))
    photo_files = sorted(list(photo_dir.glob("*.jpg")) + list(photo_dir.glob("*.jpeg")))
    
    print(f"Found {len(art_files)} art images: {[f.name for f in art_files]}")
    print(f"Found {len(photo_files)} photo images: {[f.name for f in photo_files]}")
    
    # Load VGG model
    print("\nLoading VGG19 model...")
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
    for param in vgg.parameters():
        param.requires_grad = False
    print("Model loaded successfully!")
    
    # Style transfer parameters
    content_layers = ["conv_4"]
    style_layers = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]
    content_weight = 1
    style_weight = 1e4
    num_steps = 300
    
    # Process each photo with each art style
    for photo_file in photo_files:
        print(f"\n{'='*60}")
        print(f"Processing photo: {photo_file.name}")
        print(f"{'='*60}")
        
        content_img = pre_process(str(photo_file))
        
        for art_file in art_files:
            print(f"\nApplying style from: {art_file.name}")
            print("-" * 60)
            
            style_img = pre_process(str(art_file))
            
            # Perform style transfer
            target_img, content_losses, style_losses, total_losses = style_transfer(
                content_img, style_img, vgg, content_layers, style_layers, 
                content_weight, style_weight, num_steps
            )
            
            # Generate output filename
            photo_name = photo_file.stem
            art_name = art_file.stem
            output_filename = f"{photo_name}_{art_name}_styled.jpg"
            output_path = output_dir / output_filename
            
            # Save the result
            save_image(target_img, str(output_path))
            
            print(f"Final - Content Loss: {content_losses[-1]:.6f}, Style Loss: {style_losses[-1]:.6f}, Total Loss: {total_losses[-1]:.6f}")
    
    print(f"\n{'='*60}")
    print("Style transfer complete! All outputs saved to data/output/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

