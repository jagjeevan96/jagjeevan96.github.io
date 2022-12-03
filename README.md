### jagjeevan96.github.io

# Portfolio by Jagjeevan Bhamra

PhD Student @ [Imperial College London](https://www.imperial.ac.uk/people/j.bhamra19)

[LinkedIn](https://www.linkedin.com/in/jagjeevan/)

[Medium](https://medium.com/@j.bhamra96)

This portfolio is a compilation of notebooks relating to all things data science, machine learning, and deep learning. Each project is subdived into a separate category.

## Publication(s)
[Interfacial Bonding Controls Friction in Diamond–Rock Contacts](https://pubs.acs.org/doi/10.1021/acs.jpcc.1c02857)

## Deep dive into MNIST
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
    5. Gradient Boosting through a XGBoost Classifier
- Models were evaluated through their `roc_auc_score` and `cross_val_score`
- Dimensionality reduction and compression with `PCA`

## Titanic: Machine Learning from Disaster

A binary classification problem.

Introductory [kaggle](https://www.kaggle.com/c/titanic) competition on predicting whether passengers on the Titanic survived based on a set of features.

I was already familiar with machine learning and did minimal data exploration. Also, I did not try and optimise the model. But I did test a few models to see which was best with minimal feature engineering...

1. Linear Model from Scratch

    [Notebook](https://github.com/jagjeevan96/jagjeevan96.github.io/blob/main/notebooks/titanic/linear_from_scratch.ipynb)
    
- A simple linear model built from scratch using matrix multiplication and gradient descent on an implemented loss function as the absolute value of the difference between the prediction and the dependent variable
- Accuracy comparissons are then made with a sigmoid activation function, which generally improves the loss
- Note: a random seed is not set, and so accuracy/loss metrics are always going to be the same. This was done purposely to see how the accuracy/loss changes with different random states (i.e. different runs of the model)
- This code can be taken further by computing more matrix multiplications (more hidden layers) and creating a deep learning model

2. Deep Learning using fastai

    [Notebook](https://github.com/jagjeevan96/jagjeevan96.github.io/blob/main/notebooks/titanic/fastai_tabular.ipynb)
    
- [fastai](https://github.com/fastai/fastai) is a deep learning library which provides practitioners with high-level components that can quickly and easily provide state-of-the-art results in standard deep learning domains
- fastai allows for more time for building and testing the model since it automates several data processing capabilities

3. Quick XGBoost Model as a Baseline

    [Notebook](https://github.com/jagjeevan96/jagjeevan96.github.io/blob/main/notebooks/titanic/XGBoost.ipynb)
    
- At the time of running this model, the previous two models performed better
- Minimal feature engineering

## Computer Vision with fastai
[Notebook](https://github.com/jagjeevan96/jagjeevan96.github.io/blob/main/notebooks/image-classifier.ipynb)
- Built an image classify to classify if an image was either a cucumber or zucchini
- Model had a large validation loss with the `resnet18` architecture and a somewhat high error rate but was still able to correctly identify a cucumber with very high probability
- With the `resnet152` architecture, the training loss and error rate were lower than with `resnet18`, however, the model was severely overfitting with 10 epochs as the validation loss began to increase with training

## Text Summarisation with NLP
[Notebook](https://github.com/jagjeevan96/jagjeevan96.github.io/blob/main/notebooks/nlp-text-summarisation.ipynb)
- A notebook to demonstrate implementation of a text summarisation algorithm (TextRank)
- Compared implementation with other algorithms from the `sumy` library
