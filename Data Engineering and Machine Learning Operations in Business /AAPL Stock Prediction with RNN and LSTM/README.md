# Stock Development Analysis of Apple Inc.

Hello and welcome to an analysis of the stock development of Apple Inc. This repository aims to analyze different deep learning subjects through similar approaches in the hopes of discovering underlying and embedded patterns in the listed data.

## Notebook Structure  

The notebook is divided into 9 steps:

1. **Feature Selection**  
2. **Feature Engineering**  
3. **Data Preprocessing**  
4. **Train-Test (or Train-Validation-Test) Split**  
5. **Define Your Neural Network Architecture in PyTorch**  
6. **Training Loop**  
7. **Hyperparameter Experiments**  
8. **Model Evaluation**  
9. **Documentation & Discussion**  

Through these 9 steps, the process of Recurrent Neural Network (RNN) is described and executed.

---

## Conclusion  

Through our analysis based on the stochastic batch gradient approach, we found that our model tends to be overfitted. We experimented with time steps and feature values.  

- Working with time steps, we found that a lower number of steps resulted in a lower degree of overfitting with a validation loss of **0.0025** and trained loss **0.00078**.  
- The best-performing feature experiment resulted in a validation loss of **0.0022** and trained loss **0.00049**, producing a more precise model but still with overfitting issues.  
- Combining these findings, we achieved the lowest validation loss in **epoch 30 (0.0024)** and a trained loss of **0.000428**.  

The graph is not perfect since it does not follow the y-values exactly, but the trends from the actual values are clearly reflected in the predicted values.

---

## Code Implementation  

### Load Data and Libraries  

In this section, we import relevant libraries and load the dataset. The primary libraries used are `yfinance` and `torch`:  

- `torch` (a part of PyTorch) is a machine learning library used for deep learning operations and tensor computations.  
- `yfinance` is used to import the daily stock price of **AAPL** for the last **5 years (1248 days of data)**.  
- The **"Close"** column is our target value for prediction.  

Since the data is **sequential in nature**, it is appropriate to use a **Recurrent Neural Network (RNN)** or **Long Short-Term Memory (LSTM)** model. The problem is also framed as a **regression task**.

---

### Feature Engineering  

To improve prediction accuracy, we provide additional features:  

- **`lag_5`**: The closing price **5 days ago** (previous week's close).  
- **`rolling_mean_10`**: The **mean** closing price over the last 10 days.  
- **`rolling_sd_10`**: The **standard deviation** of closing prices over the last 10 days.  

Since **overfitting** is a concern, we use a **limited number of features**. While utilizing more features could improve predictions, excessive features can lead to **overfitting**, a common problem in supervised machine learning.

---

### Normalize the Data  

To ensure features are on the same scale, we apply **Min-Max Scaling** (normalization to a **[0,1]** range). This prevents features with large numerical values from dominating the model. However, **outliers may extend beyond the [0,1] range**.

---

### Split the Data  

To **prevent overfitting**, we split the dataset:  

- **80% Training Data**  
- **20% Test Data**  

The **test dataset remains untouched** and is only used for evaluating model performance. Additionally, we introduce **validation data** to fine-tune hyperparameters and improve generalization.

---

### Prepare the Data for RNN Input  

- **Timesteps (Hyperparameter)**: Determines how much historical data is used in predictions.  
- If too large, it risks **vanishing gradients**, leading to unnecessary computations.  
- Since our focus is **short-term predictions**, we stick with RNNs. For long-term dependencies, **LSTM** would be preferable.  

#### Data Transformation  

To optimize performance and enable GPU usage, we transform the data:  

1. **Pandas DataFrame → NumPy Array → PyTorch Tensor → PyTorch Dataset → PyTorch DataLoader**  
2. **Sequential sampling** (no shuffling) ensures that surrounding data maintains its contextual integrity.  

---

## Build the Recurrent Neural Network (RNN) Model  

A **Recurrent Neural Network (RNN)** processes sequential data by maintaining a **hidden state** that captures past information.  

### How RNN Works  

At each step, the RNN receives:  
1. **Current Input** (e.g., stock price on a given day)  
2. **Hidden State** (memory from previous steps)  

The hidden state updates using:  

\[
h_t = f(W_h h_{t-1} + W_x x_t + b)
\]

where:  

- \( W_h \) and \( W_x \) are weight matrices  
- \( b \) is the bias term  
- \( f \) is an activation function (e.g., **tanh** or **ReLU**)  

The output is calculated as:  

\[
y_t = W_y h_t + b_y
\]

This allows the RNN to **retain information from past inputs** and use it for future predictions.

---

## Train the Model  

The goal is to adjust model weights through an **iterative learning process** to improve prediction accuracy.  

### Training Steps  

1. **Forward Pass**  
   - The model processes input data and generates predictions using current parameters.  

2. **Loss Calculation**  
   - We use **Mean Squared Error (MSE)** as our loss function to measure prediction errors.  

3. **Backward Pass**  
   - Computes gradients to determine the necessary weight adjustments.  

4. **Weight Updates**  
   - Optimizer (e.g., **SGD** or **Adam**) updates the model parameters to minimize loss.  

This process is repeated over **multiple epochs** to achieve better generalization.

---

## Summary  

This project analyzed Apple Inc.'s stock prices using **Recurrent Neural Networks (RNNs)**. We:  

- **Engineered features** to improve predictions while minimizing overfitting.  
- **Normalized and split data** to enhance model performance.  
- **Implemented and trained an RNN**, focusing on **short-term trends** in stock prices.  
- **Identified overfitting issues**, experimenting with timesteps and feature selection.  

Through hyperparameter tuning, we achieved our **best validation loss at epoch 30 (0.0024) with a trained loss of 0.000428**. Despite the remaining overfitting challenges, our model successfully identified **trend patterns** in stock price movements.

