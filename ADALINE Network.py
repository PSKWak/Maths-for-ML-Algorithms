import matplotlib.pyplot as plt
import numpy as np

p = np.array([[1, 1, 2, 2, -1, -2, -1, -2],
              [1, 2, -1, 0, 2, 1, -1, -2]])

t = np.array([[-1, -1, -1, -1, 1, 1, 1, 1],
              [-1, -1, 1, 1, -1, -1, 1, 1]])

#1.Plot the patterns and targets
plt.figure(figsize=(5, 5))
colors = {(-1, -1): 'red', (-1, 1): 'blue', (1, -1): 'green', (1, 1): 'purple'}
markers = {(-1, -1): 'o', (-1, 1): 's', (1, -1): '^', (1, 1): 'D'}
unique_labels = set(zip(t[0], t[1]))

for label in unique_labels:
    indices = np.where((t[0] == label[0]) & (t[1] == label[1]))[0]
    plt.scatter(p[0, indices], p[1, indices], color=colors[label], marker=markers[label], s=150, label=f'Class {label}')

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='blue', linewidth=0.5)
plt.title("Pattern and Targets")
plt.grid(True)
plt.show()


#2.Use an ADALINE network and the LMS learning rule to classify the patterns.
class ADALINE_NET():
    def __init__(self, n, alpha):
        self.w = -1 + 2 * np.random.rand(n)
        self.b = -1 + 2 * np.random.rand()
        self.eta = alpha
        self.converged = False
        self.epochs = 0

    def net_input(self, X):
        return np.dot(self.w, X) + self.b

    def train(self, X, T, max_epochs=1000,error_threshold=0.001):
        self.sse_history = []
        samples = X.shape[0]

        for epoch in range(max_epochs):
            total_error = 0
            self.epochs += 1

            for i in range(samples):
                output = self.net_input(X[i])
                error = T[i] - output
                self.w += self.eta * error * X[i]
                self.b += self.eta * error
                total_error += error ** 2

            self.sse_history.append(total_error)
            if total_error < error_threshold:
                self.converged = True
                break

    def predict(self, X):
        outputs = np.array([self.net_input(x) for x in X])
        return np.where(outputs >= 0, 1, -1)


X = p.T
T1 = t[0]
T2 = t[1]


adaline1 = ADALINE_NET(n=2, alpha=0.001)
adaline2 = ADALINE_NET(n=2, alpha=0.001)

adaline1.train(X, T1)
adaline2.train(X, T2)

# Previous SSE plotting
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.semilogy(adaline1.sse_history)
plt.title('ADALINE 1 SSE (Log Scale)')
plt.xlabel('Epoch')
plt.ylabel('Sum Squared Error')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.semilogy(adaline2.sse_history)
plt.title('ADALINE 2 SSE (Log Scale)')
plt.xlabel('Epoch')
plt.ylabel('Sum Squared Error')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 8))

x_min, x_max = p[0].min() - 1, p[0].max() + 1
y_min, y_max = p[1].min() - 1, p[1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Calculate decision boundaries
Z1 = adaline1.predict(grid_points)
Z1 = Z1.reshape(xx.shape)
Z2 = adaline2.predict(grid_points)
Z2 = Z2.reshape(xx.shape)

plt.contour(xx, yy, Z1, colors='blue', linewidths=1.5, levels=[0])
plt.contour(xx, yy, Z2, colors='red', linewidths=1.5, levels=[0])


for label in unique_labels:
    indices = np.where((t[0] == label[0]) & (t[1] == label[1]))[0]
    plt.scatter(p[0, indices], p[1, indices], color=colors[label],
                marker=markers[label],s=150, edgecolors='k')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Patterns with Both Decision Boundaries')
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.show()

# Results
print(f"ADALINE 1 - Converged: {adaline1.converged}, Epochs: {adaline1.epochs}")
print(f"Final SSE: {adaline1.sse_history[-1]:.4f}")
print(f"ADALINE 2 - Converged: {adaline2.converged}, Epochs: {adaline2.epochs}")
print(f"Final SSE: {adaline2.sse_history[-1]:.4f}")
predictions1 = adaline1.predict(X)
predictions2 = adaline2.predict(X)
print("\nTarget 1:", T1)
print("Predictions 1:", predictions1)
print("Target 2:", T2)
print("Predictions 2:", predictions2)






