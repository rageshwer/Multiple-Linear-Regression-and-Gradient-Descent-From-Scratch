# Multiple Linear Regression from Scratch using Gradient Descent

## Project Overview

This project implements **Multiple Linear Regression (MLR)** from scratch using **Gradient Descent** and compares its performance with Scikit-learn’s built-in `LinearRegression`.

The objective of this project is to understand:

- How Gradient Descent works internally
- How weights and bias are updated
- How loss minimization occurs
- How Linear Regression is implemented mathematically

---

# Libraries Used

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
```

---

# Dataset Generation

Synthetic regression data is generated using:

```python
make_regression(
    n_samples=1000,
    n_features=5,
    random_state=42,
    noise=20
)
```

## Dataset Details

| Parameter | Value |
|---|---|
| Samples | 1000 |
| Features | 5 |
| Noise | 20 |
| Problem Type | Regression |

---

# Train Test Split

The dataset is divided into:

- 70% training data
- 30% testing data

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)
```

---

# Feature Scaling

Standardization is applied using `StandardScaler`.

```python
ss = StandardScaler()

X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)
```

Feature scaling helps Gradient Descent converge faster and prevents very large gradient updates.


# Implementing Multiple Linear Regression from Scratch
### Scikit-learn’s optimized implementation is used as the benchmark model.
A custom class named `MLR` is implemented using Gradient Descent.

---

# Mathematical Model

The regression equation is:

```math
y = m_1x_1 + m_2x_2 + ... + b
```

Vector form:

```math
y = m \cdot X + b
```

Where:

- \(m\) = weight vector
- \(X\) = feature vector
- \(b\) = intercept/bias

---

# Loss Function

The Mean Squared Error (MSE) loss function is used:

```math
L = \frac{1}{n}\sum (y_i - \hat{y_i})^2
```

The objective of Gradient Descent is to minimize this loss.

---

# Gradient Descent

Parameters are updated iteratively using:

## Weight Update Rule

```math
m = m - \alpha \frac{\partial L}{\partial m}
```

## Bias Update Rule

```math
b = b - \alpha \frac{\partial L}{\partial b}
```

Where:

- \( \alpha \) = learning rate

---

# Custom MLR Class

## Constructor

```python
def __init__(self, alpha=0.001, epochs=1000):
```

Initializes:

- learning rate
- number of epochs

---

# Gradient with Respect to Weights

```python
def dL_dm(self, X, y, m, b):
```

This method computes the gradient vector (Vectorized approach for faster and accurate computation):
```python
slope += 2 * (y[i] - np.matmul(m, X[i]) - b) * (-X[i])
```

Returns:

```python
gradient vector
```

---

# Gradient with Respect to Bias

```python
def dL_db(self, X, y, m, b):
```

This computes the derivative of loss with respect to bias.

```python
slope += 2 * (y[i] - np.dot(m, X[i]) - b) * (-1)
```

Returns:

```python
scalar gradient
```

---

# Training using Gradient Descent

The `fit()` method performs iterative optimization.

## Steps

### Step 1 — Initialize Parameters

```python
self.m_old = np.array([
    np.random.randint(100)
    for _ in range(X.shape[1])
])

self.b_old = np.random.randint(100)
```

---

### Step 2 — Compute Gradients
Computing gradient vector for each feature is implemented using vectorization (matix multiplication).
```python
self.dL_dm()
self.dL_db()
```

---

### Step 3 — Update Parameters

```python
self.m_new = self.m_old - (
    self.alpha * gradient
)
```

```python
self.b_new = self.b_old - (
    self.alpha * gradient
)
```

---

# Prediction

The `predict()` method computes predictions using:

```python
np.dot(self.m_old, X[i]) + self.b_old
```

for every datapoint.

---

# Model Evaluation

Root Mean Squared Error (RMSE) is used.

## RMSE Formula

```math
RMSE = \sqrt{\frac{1}{n}\sum (y_i - \hat{y_i})^2}
```

Lower RMSE indicates better predictions.

---

# Results
Measures performance of the manually implemented Gradient Descent model.
<img width="544" height="58" alt="Screenshot 2026-05-25 at 3 41 17 AM" src="https://github.com/user-attachments/assets/1e3d7a77-4852-4d7e-b369-9cddd4983715" />

---

# Comparison

The project compares:

- learned coefficients
- intercept
- RMSE

between:

1. Scikit-learn implementation
2. Custom implementation
<img width="1147" height="47" alt="Screenshot 2026-05-25 at 3 42 29 AM" src="https://github.com/user-attachments/assets/3c7e7c12-abc0-4c4e-8be3-5a353dc6c998" />

---

# Key Learnings

This project demonstrates:

- Working of Gradient Descent
- Multiple Linear Regression mathematics
- Vectorized operations using NumPy
- Loss minimization
- Parameter optimization
- Importance of:
  - feature scaling
  - learning rate
  - epochs

---
# Conclusion

This project successfully implements Multiple Linear Regression from scratch using Gradient Descent and validates the implementation by comparing it against Scikit-learn’s optimized `LinearRegression` model.

The project provides strong practical understanding of:

- regression mathematics
- optimization
- gradient computation
- machine learning fundamentals

---

# Author

Rageshwer Singh<br>
MSc (Data Science and AI)<br>
LinkedIn : https://www.linkedin.com/in/rageshwer-singh-06a178384/
