import tensorflow as tf
import numpy as np

# Define LogicGate Model
class LogicGate(tf.Module):
    def __init__(self):
        super().__init__()
        self.built = False  # Track if model is initialized

    def __call__(self, x, train=True):
        # Initialize weights and bias on first call
        if not self.built:
            input_dim = x.shape[-1]  # Number of input features
            self.w = tf.Variable(tf.random.normal([input_dim, 2]), name="weights")  # 2 neurons
            self.b = tf.Variable(tf.zeros([2]), name="bias")
            self.w_out = tf.Variable(tf.random.normal([2, 1]), name="weights_out")  # Output layer
            self.b_out = tf.Variable(tf.zeros([1]), name="bias_out")
            self.built = True

        # Hidden layer
        hidden = tf.sigmoid(tf.add(tf.matmul(x, self.w), self.b))
        # Output layer
        z = tf.sigmoid(tf.add(tf.matmul(hidden, self.w_out), self.b_out))
        return z

# Loss function (Mean Squared Error)
def compute_loss(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))

# Training function
def train_model(model, x_train, y_train, learning_rate=0.5, epochs=6000):
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            y_pred = model(x_train)  # Forward pass
            loss = compute_loss(y_pred, y_train)

        # Update the parameters with respect to the gradient calculations
        grads = tape.gradient(loss, model.variables)
        for g, v in zip(grads, model.variables):
            v.assign_sub(learning_rate * g)

        # Print progress every 1000 epochs
        if epoch % 1000 == 0:
            acc = compute_accuracy(model, x_train, y_train)
            print(f"Epoch {epoch}, Loss: {loss.numpy():.4f}, Accuracy: {acc:.4f}")

# Accuracy function
def compute_accuracy(model, x, y_true):
    y_pred = model(x, train=False)
    y_pred_rounded = tf.round(y_pred)
    correct = tf.equal(y_pred_rounded, y_true)
    return tf.reduce_mean(tf.cast(correct, tf.float32)).numpy()

# Prepare XOR gate dataset
xor_table = np.array([[0, 0, 0],
                      [0, 1, 1],
                      [1, 0, 1],
                      [1, 1, 0]], dtype=np.float32)

x_train = xor_table[:, :2]  # Inputs: x1, x2
y_train = xor_table[:, 2:]  # Labels: y

# Initialize and train model
model = LogicGate()
train_model(model, x_train, y_train)

# Evaluate and print results
print("\nLearned Parameters:")
print("Weights (input to hidden):", model.w.numpy())
print("Biases (hidden):", model.b.numpy())
print("Weights (hidden to output):", model.w_out.numpy())
print("Bias (output):", model.b_out.numpy())

# Test model predictions
y_pred = model(x_train, train=False).numpy().round().astype(np.uint8)
print("\nPredicted XOR Truth Table:")
print(np.column_stack((xor_table[:, :2], y_pred)))

