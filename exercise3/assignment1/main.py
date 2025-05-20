import numpy as np
import matplotlib.pyplot as plt

# for reproducibility
np.random.seed(40)

# dataset generation
m, a, b = 100, 2.5, 0.1

x = np.random.uniform(-10,10, m)
noise = np.random.normal(0, 5, m)

y = a * x + b + noise

# shuffling and splitting the dataset
indices = np.arange(m)
np.random.shuffle(indices)

x = x[indices]
y = y[indices]

split_index = int(0.8 * m)


# %80 as training percent 
x_train = x[:split_index]
y_train = y[:split_index]

# %20 as testing percent
x_test = x[split_index:]
y_test = y[split_index:]

#  hyphothesis function
def predict(x, alpha1, alpha2):
    return alpha1 + alpha2 * x

# mean squared erreor function
def msqe(y, y_predict):
    return np.mean((y - y_predict) ** 2)

# gradient descent function
def training(x, y, learning_rate=0.01, max_iter=1000):
    theta1, theta2 = 0, 0
    
    m = len(x)
    
    loss_array = []

    for i in range(max_iter):
        y_predict = predict(x, theta1, theta2)
    
        # gradient calculation
        gr_theta1 = (-2/m) * np.sum(y - y_predict)
        gr_theta2 = (-2/m) * np.sum((y - y_predict) * x)

        # update
        theta1 -= learning_rate * gr_theta1
        theta2 -= learning_rate * gr_theta2

        #save losses
        loss_array.append(msqe(y, y_predict))

    return theta1, theta2, loss_array

# Train with alpha=0.01
theta1_01, theta2_01, losses_01 = training(x_train, y_train, learning_rate=0.01)

# Train with alpha=0.05
theta1_05, theta2_05, losses_05 = training(x_train, y_train, learning_rate=0.05)


# Final prediction and loss on test set
y_test_pred_01 = predict(x_test, theta1_01, theta2_01)
y_test_pred_05 = predict(x_test, theta1_05, theta2_05)
test_loss_01 = msqe(y_test, y_test_pred_01)
test_loss_05 = msqe(y_test, y_test_pred_05)

#plotting block

plt.figure(figsize=(8, 5))
plt.scatter(x_train, y_train, label="Training Data", color='blue', alpha=0.6)
x_line = np.linspace(-10, 10, 100)
y_line = predict(x_line, theta1_01, theta2_01)
plt.plot(x_line, y_line, color='red', label=f"Regression Line (α=0.01)")
plt.title("Training Data and Regression Line")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

# ===== Plot 2: Loss vs Iteration =====
plt.figure(figsize=(8, 5))
plt.plot(losses_01, label="α = 0.01", color='green')
plt.plot(losses_05, label="α = 0.05", color='orange')
plt.title("Loss vs Iteration")
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.show()

# ===== Evaluation Report =====
print(f"Test Loss with α=0.01: {test_loss_01:.4f}")
print(f"Test Loss with α=0.05: {test_loss_05:.4f}")