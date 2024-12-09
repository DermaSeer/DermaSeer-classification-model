# Getting Started

An acne classification system is a machine learning-based technology designed to analyze and categorize different types of acne based on their visual characteristics. It uses algorithms to detect patterns, features, and distinctive attributes in images of acne, allowing the system to identify and classify new, unseen images into predefined categories. The system is trained on a dataset of labeled acne images, learning to distinguish between five types of acne: Nodules, Fulminans, Papules, Fungal, and Pustules. Once the model is trained, it can accurately classify new images into the appropriate acne type, providing valuable insights for skincare professionals and individuals seeking treatment.

## Table of Contents
- [Getting Started](#getting-started)
- [Installation](#install-the-required-dependencies)
- [Fine-tuning the Model](#fine-tuning-the-model)
- [Custom Layers](#custom-layers)
- [Model Compilation](#model-compilation)
- [Training and Evaluation](#training-and-evaluation)
- [Pretrained Model](#pretrained-model)

## Contributor

| Name                        | Bangkit-ID     | Github-Profile                                       | LinkedIn                                          |
|-----------------------------|----------------|-----------------------------------------------------|--------------------------------------------------|
| Chanan Artamma    | M183B4KY0916   | [@Chanan1](https://github.com/Chanan1)         | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/chanan-artamma-586412313/) |
| Reza Putri Angga             | M296B4KX3788   | [@rrezaputria](https://github.com/rrezaputria)             | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rrezaputriaa/) |
| Muhammad Aryasatya Nugroho          | M296B4KY2749   | [@arryasatya](https://github.com/arryasatya)             | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/arryasatya/) |



## Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Fine-tuning the Model

To fine-tune the model, we use the **MobileNet** architecture, which is pretrained on **ImageNet**. The base model is used as the starting point, and we perform **transfer learning** by modifying the model to fit our acne type classification task. Fine-tuning begins from the 50th layer onwards. All layers before the 100th layer are frozen to retain the features learned from ImageNet, while layers after the 50th layer are made trainable to adapt to the acne classification task.

## Custom Layers
To modify the model for acne type classification, we add the following custom layers on top of the **MobileNet** base model:

- **Global Average Pooling**: This layer reduces the spatial dimensions of the feature maps and outputs a single value per feature map. It helps reduce overfitting by lowering the number of parameters.
  
- **Batch Normalization**: This layer normalizes the outputs from the previous layer, improving model convergence and making the training process more stable.

- **Dropout (0.4 rate)**: The dropout layer randomly deactivates 40% of the neurons during training. This regularization technique helps prevent the model from overfitting and improves its generalization ability.

- **Dense Layer (128 units, ReLU activation)**: A fully connected layer with 128 units and ReLU activation is added to help the model learn more complex representations of the data.

- **Output Layer (Softmax activation)**: The output layer uses **softmax activation**, which is suitable for multi-class classification problems. The number of output nodes corresponds to the number of acne types (classes). The softmax function outputs probabilities for each class.

## Model Compilation
The model is compiled with the following parameters:

- **Optimizer**: We use **RMSprop** optimizer with a learning rate of 0.0001. RMSprop is an adaptive optimizer that adjusts the learning rate based on a moving average of the squared gradients. This helps the model converge more efficiently, especially in cases where the learning rate needs to be adjusted dynamically throughout training.

- **Loss Function**: **Categorical cross-entropy** is used as the loss function, which is ideal for multi-class classification tasks. It calculates the difference between the predicted class probabilities and the actual class labels.

- **Evaluation Metric**: The model’s performance is evaluated using **accuracy** during training and validation to track how well it is learning.

## Training and Evaluation
During training, the model is fed preprocessed images, and the parameters are updated based on the computed loss. The following strategies are employed:

- **Early Stopping**: This callback monitors the validation loss during training. If the validation loss does not improve for 10 epochs, training will stop early to prevent overfitting, and the best model weights will be restored.

- **Learning Rate Reduction**: If the validation loss does not improve for 5 epochs, the learning rate is reduced by half to encourage the model to continue learning effectively.

The model is trained for a maximum of **50 epochs**, using the **train_generator** for training and **val_generator** for validation. The model’s progress is tracked, and its performance is evaluated on unseen data using the **test_generator**.

## Pretrained Model
The base model, **MobileNet**, is pretrained on **ImageNet**, a large-scale image dataset. This provides a strong foundation for transfer learning, where the model leverages features such as edges, textures, and patterns learned from the ImageNet dataset to perform well on the acne type classification task. By using a pretrained model, we can save computational resources and training time, as the model has already learned general features that can be fine-tuned for the specific acne classification task.

---
This fine-tuning approach enables the model to learn to classify acne types effectively, leveraging the power of **MobileNet** while adapting the model to the specific needs of the task at hand.


