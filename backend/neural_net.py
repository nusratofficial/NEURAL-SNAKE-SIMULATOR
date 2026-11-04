"""
Deep Neural Network implementation with Adam Optimizer in pure NumPy.
Tracks layer-by-layer activations, weights, biases, and gradient deltas for real-time visualization.
"""
import numpy as np
from typing import List, Dict, Any, Tuple

class DenseLayer:
    def __init__(self, input_size: int, output_size: int, activation: str = "leaky_relu"):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        # He initialization
        std = np.sqrt(2.0 / input_size)
        self.weights = np.random.randn(output_size, input_size).astype(np.float32) * std
        self.biases = np.zeros(output_size, dtype=np.float32) + 0.01

        # Delta tracking for visualization
        self.weight_deltas = np.zeros_like(self.weights)
        self.bias_deltas = np.zeros_like(self.biases)

        # Adam moments
        self.m_w = np.zeros_like(self.weights)
        self.v_w = np.zeros_like(self.weights)
        self.m_b = np.zeros_like(self.biases)
        self.v_b = np.zeros_like(self.biases)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # x shape: (batch_size, input_size) or (input_size,)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        z = np.dot(x, self.weights.T) + self.biases
        if self.activation == "leaky_relu":
            a = np.where(z > 0, z, z * 0.01)
        elif self.activation == "relu":
            a = np.maximum(0, z)
        else: # linear
            a = z
        return z, a

    def activation_derivative(self, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        if self.activation == "leaky_relu":
            return np.where(z > 0, 1.0, 0.01).astype(np.float32)
        elif self.activation == "relu":
            return np.where(z > 0, 1.0, 0.0).astype(np.float32)
        else:
            return np.ones_like(z, dtype=np.float32)


class NeuralNetwork:
    def __init__(self, layer_sizes: List[int] = None):
        if layer_sizes is None:
            layer_sizes = [12, 18, 12, 3]
        self.layer_sizes = layer_sizes
        self.layers: List[DenseLayer] = []
        self.t = 0 # Adam step counter

        for i in range(len(layer_sizes) - 1):
            is_last = (i == len(layer_sizes) - 2)
            act = "linear" if is_last else "leaky_relu"
            self.layers.append(DenseLayer(layer_sizes[i], layer_sizes[i + 1], act))

        self.last_activations: List[List[float]] = []

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, np.ndarray]]]:
        if isinstance(x, list):
            x = np.array(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        current = x
        cache = []
        activations_list = [current[0].tolist()]

        for layer in self.layers:
            z, a = layer.forward(current)
            cache.append({"x": current, "z": z, "a": a})
            activations_list.append(a[0].tolist())
            current = a

        self.last_activations = activations_list
        return current, cache

    def predict(self, x: np.ndarray) -> np.ndarray:
        output, _ = self.forward(x)
        return output[0]

    def train_batch(self, samples: List[Dict[str, Any]], learning_rate: float = 0.001,
                    beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> float:
        if not samples:
            return 0.0

        self.t += 1
        batch_size = len(samples)

        states = np.array([s["state"] for s in samples], dtype=np.float32) # (B, 12)
        actions = np.array([s["action"] for s in samples], dtype=np.int32)
        target_qs = np.array([s["target_q"] for s in samples], dtype=np.float32)

        # Forward pass on whole batch
        current = states
        cache = []
        for layer in self.layers:
            z, a = layer.forward(current)
            cache.append({"x": current, "z": z, "a": a})
            current = a

        output = current # (B, 3)

        # Compute output gradient (Huber loss)
        dZ_out = np.zeros_like(output)
        total_loss = 0.0

        for k in range(batch_size):
            a = actions[k]
            pred_q = output[k, a]
            err = pred_q - target_qs[k]

            if abs(err) > 1.0:
                clipped = np.sign(err)
                total_loss += abs(err) - 0.5
            else:
                clipped = err
                total_loss += 0.5 * (err ** 2)

            dZ_out[k, a] = clipped

        # Backpropagation through layers
        current_dZ = dZ_out
        for l in reversed(range(len(self.layers))):
            layer = self.layers[l]
            layer_cache = cache[l]
            layer_in = layer_cache["x"] # (B, input_size)

            # Gradients
            # dW: (output_size, input_size) = (current_dZ.T @ layer_in) / B
            grad_w = np.dot(current_dZ.T, layer_in) / batch_size
            grad_b = np.sum(current_dZ, axis=0) / batch_size

            # Next layer dZ
            if l > 0:
                prev_layer = self.layers[l - 1]
                prev_cache = cache[l - 1]
                prev_deriv = prev_layer.activation_derivative(prev_cache["z"], prev_cache["a"])
                # (B, input_size)
                next_dZ = np.dot(current_dZ, layer.weights) * prev_deriv
                current_dZ = next_dZ

            # Adam updates
            beta1_t = beta1 ** self.t
            beta2_t = beta2 ** self.t

            # Update weights
            layer.m_w = beta1 * layer.m_w + (1.0 - beta1) * grad_w
            layer.v_w = beta2 * layer.v_w + (1.0 - beta2) * (grad_w ** 2)
            m_w_hat = layer.m_w / (1.0 - beta1_t)
            v_w_hat = layer.v_w / (1.0 - beta2_t)
            delta_w = -learning_rate * m_w_hat / (np.sqrt(v_w_hat) + eps)
            layer.weights += delta_w
            layer.weight_deltas = delta_w

            # Update biases
            layer.m_b = beta1 * layer.m_b + (1.0 - beta1) * grad_b
            layer.v_b = beta2 * layer.v_b + (1.0 - beta2) * (grad_b ** 2)
            m_b_hat = layer.m_b / (1.0 - beta1_t)
            v_b_hat = layer.v_b / (1.0 - beta2_t)
            delta_b = -learning_rate * m_b_hat / (np.sqrt(v_b_hat) + eps)
            layer.biases += delta_b
            layer.bias_deltas = delta_b

        return float(total_loss / batch_size)

    def clone(self) -> "NeuralNetwork":
        copy_net = NeuralNetwork(self.layer_sizes)
        copy_net.copy_from(self, tau=1.0)
        return copy_net

    def copy_from(self, source: "NeuralNetwork", tau: float = 1.0):
        for l in range(len(self.layers)):
            target_l = self.layers[l]
            source_l = source.layers[l]
            if tau == 1.0:
                target_l.weights[:] = source_l.weights[:]
                target_l.biases[:] = source_l.biases[:]
            else:
                target_l.weights[:] = tau * source_l.weights + (1.0 - tau) * target_l.weights
                target_l.biases[:] = tau * source_l.biases + (1.0 - tau) * target_l.biases

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layerSizes": self.layer_sizes,
            "t": self.t,
            "layers": [
                {
                    "inputSize": l.input_size,
                    "outputSize": l.output_size,
                    "activation": l.activation,
                    "weights": l.weights.tolist(),
                    "biases": l.biases.tolist(),
                    "weightDeltas": l.weight_deltas.tolist()
                }
                for l in self.layers
            ]
        }

    def from_dict(self, data: Dict[str, Any]):
        self.layer_sizes = data["layerSizes"]
        self.t = data.get("t", 100)
        self.layers = []
        for l_data in data["layers"]:
            layer = DenseLayer(l_data["inputSize"], l_data["outputSize"], l_data["activation"])
            layer.weights = np.array(l_data["weights"], dtype=np.float32)
            layer.biases = np.array(l_data["biases"], dtype=np.float32)
            self.layers.append(layer)

