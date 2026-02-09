# Train-a-LSTM-MODEL-for-sequence-prediction-in-Node.js-
Train a LSTM MODEL for sequence prediction in Node.js 

# LSTM Sequence Prediction - Traditional JavaScript

This is a Node.js implementation of an LSTM (Long Short-Term Memory) neural network for sequence prediction, written in traditional JavaScript syntax.

## Features

- Pure traditional JavaScript (no ES6+ features)
- Uses only `var` declarations (no `const` or `let`)
- No arrow functions
- Basic Node.js `http` module (no Express)
- Runs on localhost
- Trains on sine wave sequence data
- Makes predictions on time series data

## Requirements

- Node.js (any version that supports basic modules)

## How to Run

1. Save the code to a file (e.g., `lstm_sequence_prediction.js`)

2. Run the server:
   ```bash
   node lstm_sequence_prediction.js
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:3000
   ```

## API Endpoints

- **/** - Home page with sample predictions
- **/predict** - Get a JSON prediction from the model
- **/info** - Get model information in JSON format

## How It Works

### LSTM Architecture

The implementation includes:
- **Forget Gate**: Decides what information to discard from cell state
- **Input Gate**: Decides what new information to store
- **Cell State**: Maintains long-term memory
- **Output Gate**: Decides what to output based on cell state

### Training Data

The model is trained on a sine wave sequence:
- 100 epochs of training
- Sequence length of 10 timesteps
- Learning rate of 0.01

### Prediction

After training, the model can predict the next value in a sequence based on the previous 10 values.

## Code Structure

- `sigmoid()` / `tanh()` - Activation functions
- `initializeWeights()` - Random weight initialization
- `matrixMultiply()` - Matrix operations
- `LSTMCell()` - LSTM cell constructor
- `LSTMCell.prototype.forward()` - Forward propagation
- `trainLSTM()` - Training function
- `makePrediction()` - Prediction function
- HTTP server for web interface

## Limitations

This is a simplified educational implementation:
- Simplified backpropagation (not full BPTT)
- Basic gradient updates
- Small model size
- Limited training data

For production use, consider using libraries like TensorFlow.js or Brain.js.

## Example Output

```
Starting LSTM training...
Epoch 0, Loss: 0.234567
Epoch 10, Loss: 0.123456
Epoch 20, Loss: 0.098765
...
Training complete!
Server running on http://localhost:3000
```

## License

Free to use for educational purposes.
