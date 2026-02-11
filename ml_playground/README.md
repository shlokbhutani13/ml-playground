# ML3 — Machine Learning Playground (From Scratch)

This repo is my “from-scratch” ML playground (ML3). I implemented core machine learning algorithms using only Python + NumPy (no scikit-learn training APIs) to understand how the math maps to code.

The goal is conceptual clarity + correctness, not production performance.


Supervised Learning
-Linear Regression (Gradient Descent)
  Predicts a continuous value by minimizing mean squared error (MSE).  
  Optional L2 regularization (Ridge) supported.

-Logistic Regression (Gradient Descent)  
  Binary classification using a sigmoid + cross-entropy objective.  
  Optional L2 regularization** supported.

-Perceptron
  Classic online linear classifier. Converges when data is linearly separable.

-k-Nearest Neighbors (k-NN)
  “Lazy learning” classifier: no real training step, just distance + majority vote at inference time.

Unsupervised Learning
-k-Means Clustering  
  Iteratively assigns points to the closest centroid and updates centroids until convergence.

-PCA (Principal Component Analysis)
  Reduces dimensionality by projecting onto directions of maximum variance (eigenvectors of covariance matrix).


This project is a simple supervised learning experiment that demonstrates the full machine learning workflow in Python. It loads a dataset, trains a regression model, makes predictions, and evaluates performance.

The goal of this project is to understand how data is transformed into a predictive model using scikit-learn.

This project is meant for learning and demonstration purposes.

To run locally:

1. Install dependencies
2. Run: python main.py

