# 🎨 Image Colorization Using Autoencoder

This project takes black-and-white images and automatically adds color to them using a deep learning model.
It uses the CIFAR-10 dataset and trains an autoencoder to learn how to convert grayscale images into color versions.

---

## 📌 Project Overview

This project demonstrates how a neural network can learn to colorize images.

- The project demonstrates the process of:
- Preprocessing CIFAR-10 images
- Converting RGB images to grayscale
- Training a convolutional autoencoder
- Generating colorized predictions
- Comparing grayscale, predicted color, and original images
- Saving the trained model for future use

The goal is to show how deep learning can understand color patterns and apply them to black-and-white images.

---

## 🚀 Features

- End-to-end autoencoder for image colorization
- Visualization of:
  - Grayscale input
  - Model-generated color output
  - Original RGB image
- Random test image colorization
- Model saved in .keras format 

---

## 🧩 Dataset

The dataset contains:

- CIFAR-10 (60,000 images, 32×32 resolution)
- Used only for image reconstruction (labels are not required)

---

## 🚀 Future Upgrades

- Use LAB color space for better color realism
- Replace autoencoder with U-Net or GAN for sharper outputs
- Train on larger datasets like ImageNet/Places365
- Add a web app for real-image uploads
- Improve performance with tuning and augmentation

---
