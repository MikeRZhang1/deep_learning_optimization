# CPSC 485: Augmenting Deep Learning Optimizers with Ideas from Line Search, Trust-Region Methods, and Non-linear Least Squares

**Author:** Mike Zhang  

## Project Overview

Optimization algorithms serve as the bedrock for state-of-the-art model training in deep learning. This project investigates whether standard deep learning optimizers in PyTorch can be enhanced using classical optimization techniques that are less common in modern neural network training. 

Specifically, this project explores:
1.  **Line Search:** Augmenting popular first-order methods (Momentum SGD, Adam) with a Strong Wolfe line search to improve step-size selection.
2.  **Trust-Region Methods:** Implementing a Trust-Region algorithm with Cauchy point updates in PyTorch.
3.  **Non-linear Least Squares:** Adapting the Levenberg-Marquardt algorithm for neural network regression tasks.

The results indicate that Strong Wolfe line search consistently improves convergence speed and out-of-sample performance for standard classification and regression tasks. While the second-order methods (Trust-Region and Levenberg-Marquardt) proved more complex to implement, they demonstrated remarkable stability and offer promising avenues for future research in robust model training.

---

## File Descriptions

The project is organized into the following modules and notebooks:

### Source Code
* **`models.py`** Contains the PyTorch neural network architectures used for experimentation, including the Convolutional Neural Networks (CNNs) for image classification and the Multi-Layer Perceptron (MLP) for regression.

* **`new_optimizers.py`** The core implementation file containing the custom optimizer classes. This includes:
    * `Adam_Strong_Wolfe`
    * `MomentumSGD_Strong_Wolfe`
    * `TrustRegion_Cauchy`
    * `LevenbergMarquardt`

### Experiments
* **`img_classify.ipynb`** The primary notebook for the image classification task on the **CIFAR-100** dataset. This file contains the training loops, validation logic, and code to generate comparative plots (loss curves and accuracy metrics) for the different optimizers.

* **`wine_regression.ipynb`** The primary notebook for the regression task on the **Wine Quality** dataset. It demonstrates the application of the Levenberg-Marquardt algorithm alongside standard baselines, including data preprocessing, training execution, and performance visualization.