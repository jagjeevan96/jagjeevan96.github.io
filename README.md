### Currently formatting and organising projects for upload. *More to come.*

# Portfolio by Jagjeevan Bhamra

PhD Student @ [Imperial College London](https://www.imperial.ac.uk/people/j.bhamra19)

[LinkedIn](https://www.linkedin.com/in/jagjeevan/)

[Medium](https://medium.com/@j.bhamra96)

This portfolio is a compilation of notebooks and Python programs subdivided into separate categories relating to mostly data science.

## Deep Learning from the Foundations

**Implemented Training Loop from Scratch**

[Notebook](https://github.com/jagjeevan96/jagjeevan96.github.io/blob/main/notebooks/training-loop.ipynb)
- Implemented matrix multiplication, forward pass, backward propogation, activation/loss functions, and training loop from scratch
- Comparisons are made with the PyTorch implementation

## Stable Diffusion with Diffusers
[Notebook](https://github.com/jagjeevan96/jagjeevan96.github.io/blob/main/notebooks/stable-diffusion.ipynb)
- Explore diffusion models and examined main components
- Used the Hugging Face [diffusers library](https://github.com/huggingface/diffusers)
- Some code adapted from the [fast.ai course](https://www.fast.ai)
- *Incomplete notebook. Work is in progress.*

## Comparing Three Unique Classification Algorithms

NOTE: Minimal feature engineering and models were not optimised for performance.

**Linear Model from Scratch**

[Notebook](https://github.com/jagjeevan96/jagjeevan96.github.io/blob/main/notebooks/titanic/linear_from_scratch.ipynb)
    
- A simple linear model built from scratch using matrix multiplication and gradient descent on an implemented loss function as the absolute value of the difference between the prediction and the dependent variable
- Accuracy comparisons are then made with a sigmoid activation function, which generally improves the loss
- Note: a random seed is not set, and so accuracy/loss metrics are not always going to be the same. This was done purposely to see how the accuracy/loss changes with different random states (i.e. different runs of the model)
- This code can be taken further by computing more matrix multiplications (more hidden layers) and creating a deep learning model

**Deep Learning using fastai**

[Notebook](https://github.com/jagjeevan96/jagjeevan96.github.io/blob/main/notebooks/titanic/fastai_tabular.ipynb)
    
- [fastai](https://github.com/fastai/fastai) is a deep learning library which provides practitioners with high-level components that can quickly and easily provide state-of-the-art results in standard deep learning domains
- fastai allows for more time for building and testing the model since it automates several data processing capabilities

**Quick XGBoost Model as a Baseline**

[Notebook](https://github.com/jagjeevan96/jagjeevan96.github.io/blob/main/notebooks/titanic/XGBoost.ipynb)
    
- At the time of running this model, the previous two models performed better
- Minimal feature engineering

## Fraud Detection
[Notebook](https://github.com/jagjeevan96/jagjeevan96.github.io/blob/main/notebooks/fraud-detection.ipynb)
- Highest classification was 99% accuracy using `BaseEstimator` to build classifier to classify everything as not fraud (data is highly unbalanced)
- Showed how `cross_val_score` is a poor metric for this classification task
- `xgboost` had the best `roc_auc_score` with `1.00` on the training set (validation) and `0.98` on new data (test set)

## Computer Vision with fastai
[Notebook](https://github.com/jagjeevan96/jagjeevan96.github.io/blob/main/notebooks/image-classifier.ipynb)
- Built an image classifier to classify if an image was either a cucumber or zucchini
- Model had a large validation loss with the `resnet18` architecture and a somewhat high error rate but was still able to correctly identify a cucumber with very high probability
- With the `resnet152` architecture, the training loss and error rate were lower than with `resnet18`, however, the model was severely overfitting with 10 epochs as the validation loss began to increase with training

## Comparing Metrics and Classifiers on MNIST
[Notebook](https://github.com/jagjeevan96/jagjeevan96.github.io/blob/main/notebooks/mnist.ipynb)
- Highest classification accuracy at the time of running ~95%
- Accuracy could have been improved by:
    - Training the entire data set (50% was trained to save on computational time)
    - Optimise hyperparameters with `GridSearchCV`
- Models included:
    1. Simple Binary Classifier
    2. One Vs One Classifier
    3. Random Forest Classifier
    4. AdaBoost Classifier with a Decision Tree Classifier as the base
    5. Gradient Boosting through an XGBoost Classifier
- Models were evaluated through their `roc_auc_score` and `cross_val_score`
- Dimensionality reduction and compression with `PCA`

## Text Summarisation with NLP
[Notebook](https://github.com/jagjeevan96/jagjeevan96.github.io/blob/main/notebooks/nlp-text-summarisation.ipynb)
- A notebook to demonstrate implementation of a text summarisation algorithm (TextRank)
- Compared implementation with other algorithms from the `sumy` library

## Image Classification with Keras
[Notebook](https://github.com/jagjeevan96/jagjeevan96.github.io/blob/main/notebooks/fashion-mnist.ipynb)
- Explored *Fashion MNIST*
- Employed Sequential API (TensorFlow/Keras)
- Fine-tuned hyperparameters, implemented early stopping, used variety of activation functions, optimisers etc.
- Linear models achieve ~83% accuracy, best model achieved ~94% accuracy

## Visualise Merge Sort
[Code](https://github.com/jagjeevan96/merge_sort)
- Simple Python script that shows a visualisation of merge sort using pygame
