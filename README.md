# CNN from Scratch using NumPy 🧠

This project is a **Convolutional Neural Network (CNN) implemented fully from scratch in NumPy** — without using high-level deep learning libraries such as TensorFlow or PyTorch.  

It is inspired by **Week 1 labs of the Convolutional Neural Networks course** from [DeepLearning.AI](https://www.deeplearning.ai/) on Coursera.  
Instead of only implementing forward and backward functions as in the course, I went further and created a **complete working CNN model**.


# Table of Contents

1. [🚀 Features](#features)
2. [⚠️ Limitations](#limitations)
3. [📂 Project Structure](#project-structure)
4. [📖 Inspiration](#inspiration)
5. [📜 License](#license)
6. [🎯 Motivation](#motivation)
7. [👤 Author](#author)

---

## 🚀 Features <a name="features"></a>
- Implemented **Conv2D**, **MaxPooling**, **Flatten**, and **Fully Connected layers** from scratch.  
- **Forward and Backward propagation** for all layers.  
- Basic **cross-entropy loss** and gradient computation.  
- A simple **training loop** that runs on CPU only.  
- Example notebook (`cnn-from-scratch.ipynb`) demonstrating training for 2 epochs.  

---

## ⚠️ Limitations <a name="limitations"></a>
- This project is **CPU-only** and does not use GPU acceleration.  
- Due to resource limitations, I only trained it for **2 epochs** on small data samples.  
- Not intended for production use — rather as an **educational project** to understand the inner workings of CNNs.  

---

## 📂 Project Structure <a name="project-structure"></a>
<img width="987" height="447" alt="image" src="https://github.com/user-attachments/assets/88b2ab3b-c66a-4fa7-b14d-3007399c360e" />
---

## 📖 Inspiration <a name="inspiration"></a>

This project is inspired by the DeepLearning.AI Convolutional Neural Networks course on Coursera.
I decided to push the idea further by building a whole CNN model, not just standalone forward/backward functions.

---

## 📜 License <a name="license"></a>

This project is licensed under the MIT License — see the LICENSE file for details.

---

## 🎯 Motivation <a name="motivation"></a>

Most people use libraries like TensorFlow or PyTorch without knowing what happens under the hood.
This project forces you to implement the building blocks manually to truly understand CNNs at the mathematical and algorithmic level.

---
## 👤 Author <a name="author"></a>

Built with ❤️ by Youssef Ahmed El Demerdash.
