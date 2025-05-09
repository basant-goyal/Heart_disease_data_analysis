import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

#Read the CSV file
data = pd.read_csv("heart.csv") 
x = data['age'].values
y = data['chol'].values

#Visualize the data points through a 2D graph
plt.scatter(x, y, color='blue', label='Data points') 
plt.xlabel('age')
plt.ylabel('cholestrol')
plt.title('Data Points') 
plt.legend()
plt.show()

# Define the loss function (MSE)
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

#gradient descent
def gradient_descent(x, y, m, c, learning_rate): 
    n = len(y)
    y_pred = m * x + c
    dm = -(2/n) * np.sum(x * (y - y_pred)) 
    dc = -(2/n) * np.sum(y - y_pred)
    m -= learning_rate * dm 
    c -= learning_rate * dc 
    return m, c
#	Training the model 
m, c = 0, 0
learning_rate = 0.001
iterations = 1000

# Perform training
for i in range(iterations):
    m, c = gradient_descent(x, y, m, c, learning_rate) 
    if i % 100 == 0:
        y_pred = m * x + c
        loss = mean_squared_error(y, y_pred)
        print(f"Iteration {i}, Loss: {loss:.4f}, m: {m:.4f}, c: {c:.4f}")

# Final line plot 
y_pred = m * x + c
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, y_pred, color='red', label=f'Fitted line (m={m:.2f}, c={c:.2f})')
plt.xlabel('age')
plt.ylabel('cholestrol')
plt.title('Linear Regression Result') 
plt.legend()
plt.show()


