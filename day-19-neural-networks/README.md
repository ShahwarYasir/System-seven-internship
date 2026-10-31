# Day 19 - Neural Networks

**System 7 | Week 3 | Module C (Day 5)**
Topics: perceptron, Keras workflow, SGD vs Adam, activation functions, learning rate / batch size / epochs, student results task.

## Files

| File | Question | Level | What it does |
|---|---|---|---|
| `q14_perceptron_and_gate.py` | Q14 | Easy | Single perceptron in NumPy that learns the AND gate |
| `q15_keras_sgd_vs_adam.py` | Q15 | Medium | Keras Sequential model on tabular data, loss curves, SGD vs Adam |
| `student_results_nn.py` | Practical task | | Neural network that predicts pass/fail, compared with a logistic-regression baseline |
| `activation_functions_demo.py` | Learning topic | | Plots ReLU, sigmoid, tanh and a softmax example |
| `hyperparameter_demo.py` | Learning topic | | Trains the Q15 network varying learning rate, batch size and epochs |
| `outputs/` | | | Saved plots |

## How to run

```bash
pip install -r requirements.txt
mkdir -p outputs
python q14_perceptron_and_gate.py   # and so on
```

## Results

**Q14.** The perceptron uses a step activation and the update rule `w += lr * (target - prediction) * x`. It converges in 5 epochs to weights of about (0.13, 0.096) and bias -0.2, and reproduces the full AND truth table. Plot: `outputs/q14_perceptron_boundary.png`.

**Q15.** Same architecture (30 inputs, 16 ReLU, 8 ReLU, 1 sigmoid) and same starting weights, 40 epochs on the breast cancer data:

| Optimizer | Epochs until val loss < 0.15 | Final val loss | Test accuracy |
|---|---|---|---|
| SGD (lr=0.01) | 35 | 0.1393 | 0.9123 |
| Adam (lr=0.001) | 12 | 0.0481 | 0.9474 |

Adam is about three times faster to reach a good loss. SGD with this learning rate is still improving at epoch 40, so given more epochs it would catch up. The comparison shows speed, not that SGD cannot work. Plot: `outputs/q15_sgd_vs_adam.png`.

### Practical task: predict student results

1,500 synthetic students (study hours, attendance, previous GPA, assignment average, sleep, absences), predicting pass or fail (60% pass). Network: 32 ReLU, dropout 0.2, 16 ReLU, 1 sigmoid, trained with early stopping (stopped at epoch 51, best at epoch 41).

| Model | Test accuracy | Recall (fail) | F1 (pass) |
|---|---|---|---|
| Logistic Regression | 0.9167 | 0.8750 | 0.9315 |
| Neural Network | 0.9000 | 0.8500 | 0.9180 |

The neural network does **not** beat the simple baseline here. The gap is 5 students out of 300, which is within noise. On small, mostly linear tabular data a logistic regression is often just as good, simpler and easier to explain, and the NN would need more data or genuinely non-linear patterns to justify itself. Because the data is synthetic, treat these numbers as a demonstration of the workflow rather than a finding about real students. Plot: `outputs/student_results_nn.png`.

## Learning notes

### Activation functions

Without an activation function a stack of layers collapses into one linear function, no matter how deep. The activation adds the non-linearity that lets a network learn curved patterns. Plots: `outputs/activation_functions.png`.

| Function | Output range | What it does | Where to use it |
|---|---|---|---|
| **ReLU** `max(0, z)` | 0 to infinity | Passes positive values, zeroes the negatives. Cheap and avoids vanishing gradients. | Default for hidden layers |
| **Sigmoid** `1 / (1 + e^-z)` | 0 to 1 | Squashes any value into a probability. Saturates for large values, so gradients get tiny. | Output layer for binary classification |
| **Softmax** | 0 to 1, sums to 1 | Turns a vector of scores into probabilities across classes (scores 2.0, 1.0, 0.1, -1.0 become 0.64, 0.23, 0.10, 0.03). | Output layer for multiclass classification |
| **Tanh** | -1 to 1 | A zero-centred sigmoid. | Older networks and some recurrent models |

Analogy: ReLU is a one-way valve (flow passes only in one direction), sigmoid is a dimmer switch between off and on, and softmax is dividing 100% of your attention among several options.

### Learning rate, batch size and epochs

Measured with `hyperparameter_demo.py` (Adam, breast cancer data). Plot: `outputs/hyperparameter_demo.png`.

**Learning rate (step size of each update):**

| lr | Final val loss | Test accuracy |
|---|---|---|
| 0.0001 | 0.3508 | 0.9123 |
| 0.001 | 0.0481 | 0.9474 |
| 0.01 | 0.0313 | 0.9649 |
| 0.1 | 0.0598 | 0.9561 |

Too small and training crawls (0.0001 had not converged after 40 epochs). Too large and it overshoots and gets noisy (0.1 was worse than 0.01). The sweet spot sits in the middle. Imagine walking downhill in fog: tiny steps take forever, huge steps make you leap past the valley.

**Batch size (samples per weight update):**

| Batch size | Updates per epoch | Final val loss | Test accuracy |
|---|---|---|---|
| 8 | about 46 | 0.0451 | 0.9737 |
| 32 | about 12 | 0.0481 | 0.9474 |
| 128 | about 3 | 0.1481 | 0.9035 |
| 364 (full batch) | 1 | 0.3591 | 0.8947 |

With a fixed number of epochs, small batches make many more updates, so they learn faster and add helpful noise, but each epoch takes longer on big data. Large batches are smoother and use hardware efficiently, but they need more epochs to reach the same loss.

**Epochs (passes over the data):** over 150 epochs the training loss kept falling (down to 0.0037) while validation loss bottomed out at epoch 67 and then stopped improving. That gap is the start of overfitting, which is why the student task uses `EarlyStopping` with `restore_best_weights=True`.
