import matplotlib.pyplot as plt
def func(x):
    return (x + 3) ** 2

def grad(x):
    return 2 * (x + 3)
# Parameters for Gradient Descent
x = 2                # Initial guess
learning_rate = 0.1  # Step size
iterations = 50          # Number of iterations

# For plotting
x_vals = [x]
y_vals = [func(x)]
# Gradient Descent Loop
for i in range(iterations):
    dx = grad(x)
    x = x - learning_rate * dx

    x_vals.append(x)
    y_vals.append(func(x))

    print(f"Iteration {i+1}: x = {x:.5f}, f(x) = {func(x):.5f}")
print(f"\nLocal minimum at x = {x:.5f}, f(x) = {func(x):.5f}")
x_plot = [i for i in range(-10, 5)]
y_plot = [func(i) for i in x_plot]

plt.plot(x_plot, y_plot, label='y = (x+3)^2')
plt.scatter(x_vals, y_vals, color='red', label="Gradient Descent Steps")
plt.title("Gradient Descent to find Local Minima")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()