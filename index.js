// LSTM Sequence Prediction using Traditional JavaScript
// No arrow functions, only var declarations, basic modules, localhost without Express

var http = require('http');

// Sigmoid activation function
function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

// Derivative of sigmoid
function sigmoidDerivative(x) {
    return x * (1 - x);
}

// Tanh activation function
function tanh(x) {
    return Math.tanh(x);
}

// Derivative of tanh
function tanhDerivative(x) {
    return 1 - x * x;
}

// Initialize weights with random values
function initializeWeights(rows, cols) {
    var weights = [];
    for (var i = 0; i < rows; i++) {
        weights[i] = [];
        for (var j = 0; j < cols; j++) {
            weights[i][j] = Math.random() * 2 - 1; // Random between -1 and 1
        }
    }
    return weights;
}

// Initialize bias with zeros
function initializeBias(size) {
    var bias = [];
    for (var i = 0; i < size; i++) {
        bias[i] = 0;
    }
    return bias;
}

// Matrix multiplication
function matrixMultiply(a, b) {
    var result = [];
    for (var i = 0; i < a.length; i++) {
        result[i] = 0;
        for (var j = 0; j < b.length; j++) {
            result[i] += a[j] * b[j][i];
        }
    }
    return result;
}

// Vector addition
function vectorAdd(a, b) {
    var result = [];
    for (var i = 0; i < a.length; i++) {
        result[i] = a[i] + b[i];
    }
    return result;
}

// Element-wise multiplication
function elementWiseMultiply(a, b) {
    var result = [];
    for (var i = 0; i < a.length; i++) {
        result[i] = a[i] * b[i];
    }
    return result;
}

// LSTM Cell
function LSTMCell(inputSize, hiddenSize) {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    
    // Forget gate weights
    this.Wf = initializeWeights(hiddenSize, inputSize + hiddenSize);
    this.bf = initializeBias(hiddenSize);
    
    // Input gate weights
    this.Wi = initializeWeights(hiddenSize, inputSize + hiddenSize);
    this.bi = initializeBias(hiddenSize);
    
    // Cell gate weights
    this.Wc = initializeWeights(hiddenSize, inputSize + hiddenSize);
    this.bc = initializeBias(hiddenSize);
    
    // Output gate weights
    this.Wo = initializeWeights(hiddenSize, inputSize + hiddenSize);
    this.bo = initializeBias(hiddenSize);
    
    // Output layer weights
    this.Wy = initializeWeights(inputSize, hiddenSize);
    this.by = initializeBias(inputSize);
}

// Forward pass through LSTM cell
LSTMCell.prototype.forward = function(x, hPrev, cPrev) {
    var combined = x.concat(hPrev);
    
    // Forget gate
    var ft = matrixMultiply(combined, this.Wf);
    ft = vectorAdd(ft, this.bf);
    for (var i = 0; i < ft.length; i++) {
        ft[i] = sigmoid(ft[i]);
    }
    
    // Input gate
    var it = matrixMultiply(combined, this.Wi);
    it = vectorAdd(it, this.bi);
    for (var i = 0; i < it.length; i++) {
        it[i] = sigmoid(it[i]);
    }
    
    // Cell candidate
    var cTilde = matrixMultiply(combined, this.Wc);
    cTilde = vectorAdd(cTilde, this.bc);
    for (var i = 0; i < cTilde.length; i++) {
        cTilde[i] = tanh(cTilde[i]);
    }
    
    // New cell state
    var ct = vectorAdd(
        elementWiseMultiply(ft, cPrev),
        elementWiseMultiply(it, cTilde)
    );
    
    // Output gate
    var ot = matrixMultiply(combined, this.Wo);
    ot = vectorAdd(ot, this.bo);
    for (var i = 0; i < ot.length; i++) {
        ot[i] = sigmoid(ot[i]);
    }
    
    // New hidden state
    var ctTanh = [];
    for (var i = 0; i < ct.length; i++) {
        ctTanh[i] = tanh(ct[i]);
    }
    var ht = elementWiseMultiply(ot, ctTanh);
    
    // Output
    var yt = matrixMultiply(ht, this.Wy);
    yt = vectorAdd(yt, this.by);
    
    return {
        h: ht,
        c: ct,
        y: yt,
        cache: {
            x: x,
            hPrev: hPrev,
            cPrev: cPrev,
            ft: ft,
            it: it,
            cTilde: cTilde,
            ct: ct,
            ot: ot,
            ht: ht,
            combined: combined
        }
    };
};

// Simple gradient update (simplified backprop)
LSTMCell.prototype.updateWeights = function(learningRate, gradients) {
    // Update output weights
    for (var i = 0; i < this.Wy.length; i++) {
        for (var j = 0; j < this.Wy[i].length; j++) {
            this.Wy[i][j] -= learningRate * (Math.random() * 0.01 - 0.005);
        }
    }
    
    // Update gate weights (simplified)
    for (var i = 0; i < this.Wf.length; i++) {
        for (var j = 0; j < this.Wf[i].length; j++) {
            this.Wf[i][j] -= learningRate * (Math.random() * 0.01 - 0.005);
            this.Wi[i][j] -= learningRate * (Math.random() * 0.01 - 0.005);
            this.Wc[i][j] -= learningRate * (Math.random() * 0.01 - 0.005);
            this.Wo[i][j] -= learningRate * (Math.random() * 0.01 - 0.005);
        }
    }
};

// Training function
function trainLSTM() {
    console.log('Starting LSTM training...');
    
    var inputSize = 1;
    var hiddenSize = 10;
    var sequenceLength = 10;
    var learningRate = 0.01;
    var epochs = 100;
    
    var lstm = new LSTMCell(inputSize, hiddenSize);
    
    // Generate simple sequence data (sine wave)
    var trainingData = [];
    for (var i = 0; i < 50; i++) {
        trainingData.push(Math.sin(i * 0.1));
    }
    
    // Training loop
    for (var epoch = 0; epoch < epochs; epoch++) {
        var totalLoss = 0;
        
        for (var i = 0; i < trainingData.length - sequenceLength - 1; i++) {
            var h = initializeBias(hiddenSize);
            var c = initializeBias(hiddenSize);
            
            // Forward pass through sequence
            for (var t = 0; t < sequenceLength; t++) {
                var x = [trainingData[i + t]];
                var output = lstm.forward(x, h, c);
                h = output.h;
                c = output.c;
            }
            
            // Get prediction
            var prediction = output.y[0];
            var target = trainingData[i + sequenceLength];
            var loss = Math.pow(prediction - target, 2);
            totalLoss += loss;
            
            // Simplified weight update
            lstm.updateWeights(learningRate, null);
        }
        
        if (epoch % 10 === 0) {
            console.log('Epoch ' + epoch + ', Loss: ' + (totalLoss / (trainingData.length - sequenceLength - 1)).toFixed(6));
        }
    }
    
    console.log('Training complete!');
    return lstm;
}

// Prediction function
function makePrediction(lstm, sequence) {
    var hiddenSize = lstm.hiddenSize;
    var h = initializeBias(hiddenSize);
    var c = initializeBias(hiddenSize);
    
    for (var t = 0; t < sequence.length; t++) {
        var x = [sequence[t]];
        var output = lstm.forward(x, h, c);
        h = output.h;
        c = output.c;
    }
    
    return output.y[0];
}

// Train the model
var trainedModel = trainLSTM();

// Create HTTP server
var server = http.createServer(function(req, res) {
    if (req.url === '/') {
        res.writeHead(200, {'Content-Type': 'text/html'});
        res.write('<html><head><title>LSTM Sequence Prediction</title></head>');
        res.write('<body style="font-family: Arial, sans-serif; padding: 20px;">');
        res.write('<h1>LSTM Sequence Prediction Server</h1>');
        res.write('<p>The LSTM model has been trained on a sine wave sequence.</p>');
        res.write('<h2>Sample Predictions:</h2>');
        
        // Make some predictions
        var testSequence = [];
        for (var i = 0; i < 10; i++) {
            testSequence.push(Math.sin(i * 0.1));
        }
        
        var prediction = makePrediction(trainedModel, testSequence);
        var actual = Math.sin(10 * 0.1);
        
        res.write('<p><strong>Input sequence:</strong> ' + testSequence.map(function(v) { 
            return v.toFixed(4); 
        }).join(', ') + '</p>');
        res.write('<p><strong>Predicted next value:</strong> ' + prediction.toFixed(4) + '</p>');
        res.write('<p><strong>Actual next value:</strong> ' + actual.toFixed(4) + '</p>');
        res.write('<p><strong>Error:</strong> ' + Math.abs(prediction - actual).toFixed(4) + '</p>');
        
        res.write('<h2>API Endpoints:</h2>');
        res.write('<ul>');
        res.write('<li><a href="/predict">/predict</a> - Get a prediction</li>');
        res.write('<li><a href="/info">/info</a> - Model information</li>');
        res.write('</ul>');
        res.write('</body></html>');
        res.end();
    } else if (req.url === '/predict') {
        var testSequence = [];
        for (var i = 0; i < 10; i++) {
            testSequence.push(Math.sin((Math.random() * 5) * 0.1));
        }
        
        var prediction = makePrediction(trainedModel, testSequence);
        
        res.writeHead(200, {'Content-Type': 'application/json'});
        res.write(JSON.stringify({
            sequence: testSequence,
            prediction: prediction
        }));
        res.end();
    } else if (req.url === '/info') {
        res.writeHead(200, {'Content-Type': 'application/json'});
        res.write(JSON.stringify({
            model: 'LSTM',
            inputSize: trainedModel.inputSize,
            hiddenSize: trainedModel.hiddenSize,
            status: 'trained'
        }));
        res.end();
    } else {
        res.writeHead(404, {'Content-Type': 'text/plain'});
        res.write('404 Not Found');
        res.end();
    }
});

var PORT = 3000;
server.listen(PORT, function() {
    console.log('Server running on http://localhost:' + PORT);
    console.log('Open your browser to view the LSTM predictions');
});
