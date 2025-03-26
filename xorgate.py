import tensorflow as tf
import numpy as np

# Define XOR Neural Network Model
class XORGate(tf.Module):
    def __init__(self, hidden_units=2):
        super().__init__()
        self.built = False
        self.hidden_units = hidden_units

    def __call__(self, x):
        if not self.built:
            input_dim = x.shape[-1]

            # First layer (Input to Hidden)
            self.w1 = tf.Variable(tf.random.normal([input_dim, self.hidden_units]), name="w1")
            self.b1 = tf.Variable(tf.zeros([self.hidden_units]), name="b1")

            # Second layer (Hidden to Output)
            self.w2 = tf.Variable(tf.random.normal([self.hidden_units, 1]), name="w2")
            self.b2 = tf.Variable(tf.zeros([1]), name="b2")

            self.built = True

        # Forward pass
        hidden = tf.nn.sigmoid(tf.add(tf.matmul(x, self.w1), self.b1))  # Hidden layer activation
        output = tf.nn.sigmoid(tf.add(tf.matmul(hidden, self.w2), self.b2))  # Output layer activation
        return output

# Loss function (Binary Cross-Entropy for better performance)
def compute_loss(y_pred, y_true):
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))

# Training function
def train_model(model, x_train, y_train, learning_rate=0.5, epochs=10000):
    optimizer = tf.optimizers.SGD(learning_rate)

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            y_pred = model(x_train)
            loss = compute_loss(y_pred, y_train)

        grads = tape.gradient(loss, [model.w1, model.b1, model.w2, model.b2])
        optimizer.apply_gradients(zip(grads, [model.w1, model.b1, model.w2, model.b2]))

        if epoch % 1000 == 0:
            acc = compute_accuracy(model, x_train, y_train)
            tf.print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")

# Accuracy function
def compute_accuracy(model, x, y_true):
    y_pred = model(x)
    y_pred_rounded = tf.round(y_pred)
    correct = tf.equal(y_pred_rounded, y_true)
    return tf.reduce_mean(tf.cast(correct, tf.float32)).numpy()

# XOR dataset
xor_table = np.array([[0, 0, 0],
                      [1, 0, 1],
                      [0, 1, 1],
                      [1, 1, 0]], dtype=np.float32)

x_train = xor_table[:, :2]  # Inputs: x1, x2
y_train = xor_table[:, 2:]  # Labels: y

# Train XOR model
model = XORGate()
train_model(model, x_train, y_train)

# Print learned parameters
print("\nLearned Parameters:")
print(f"Weights (input to hidden):\n{model.w1.numpy()}")
print(f"Biases (hidden):\n{model.b1.numpy()}")
print(f"Weights (hidden to output):\n{model.w2.numpy()}")
print(f"Bias (output):\n{model.b2.numpy()}")

# Test model predictions
y_pred = model(x_train).numpy().round().astype(np.uint8)
print("\nPredicted XOR Truth Table:")
print(np.column_stack((xor_table[:, :2], y_pred)))
