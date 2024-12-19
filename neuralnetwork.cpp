#include "matrixStructure.cpp"
#include <random>
#include <iostream>

double relu(double x) { return (x > 0) ? x : 0.01 * x; }
double relu_derivative(double x) { return (x > 0) ? 1 : 0.01; }

matrix softmax(const matrix& layerOutput) {
    int rows = layerOutput.rows;
    int columns = layerOutput.columns;

    matrix softmaxOutput(rows, columns);

    for (int j = 0; j < columns; ++j) {
        double maxVal = layerOutput[0][j];
        for (int i = 1; i < rows; i++) {
            if (layerOutput[i][j] > maxVal) {
                maxVal = layerOutput[i][j];
            }
        }

        double sumExp = 0.0;
        for (int i = 0; i < rows; i++) {
            softmaxOutput[i][j] = std::exp(layerOutput[i][j] - maxVal);
            sumExp += softmaxOutput[i][j];
        }

        for (int i = 0; i < rows; i++) {
            softmaxOutput[i][j] /= sumExp;
        }
    }

    return softmaxOutput;
}

class layer {
public:
    int inputSize;
    int outputSize;
    matrix W; // weights
    matrix b; // biases
    matrix a; // activations
    matrix z; // pre activations
    matrix error; // error matrix
    matrix gradientW; // weight gradients
    matrix gradientb; // bias gradients

    // constructor
    layer(int input_size, int output_size)
        : inputSize(input_size),
          outputSize(output_size),
          W(outputSize, inputSize),
          b(outputSize, 1),
          a(outputSize, 1),
          z(outputSize, 1),
          error(outputSize, 1),
          gradientW(outputSize, inputSize),
          gradientb(outputSize, 1) {
        initialize_weights();
        initialize_bias();
    }

    // initialize weights with he initialization
    void initialize_weights() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dist(0.0, std::sqrt(2.0 / inputSize));

        for (int i = 0; i < W.rows; i++) {
            for (int j = 0; j < W.columns; j++) {
                W[i][j] = dist(gen);
            }
        }
    }

    // initialize biases to small values
    void initialize_bias() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> bias_dist(-0.1, 0.1);

        for (int i = 0; i < b.rows; i++) {
            b[i][0] = bias_dist(gen);
        }
    }

    // activation function
    virtual matrix activation(const matrix& z, bool isOutputLayer = false) {
        if (isOutputLayer) {
            return softmax(z);
        } else {
            matrix activated(z.rows, z.columns);
            for (int i = 0; i < z.rows; i++) {
                for (int j = 0; j < z.columns; j++) {
                    activated[i][j] = relu(z[i][j]);
                }
            }
            return activated;
        }
    }

    // forward pass with batch processing
    matrix forward(const matrix& x, bool isOutputLayer = false) {
        // z = W * x + b
        matrix zBatch = add(multiply(W, x), broadcast(b, x.columns));
        z = zBatch;
        a = activation(zBatch, isOutputLayer);
        return a;
    }

    // compute error for output and/or hidden layers
    void computeError(const matrix& nextLayerW, const matrix& nextError, bool isOutputLayer, const matrix& labels) {
        if (isOutputLayer) {
            // output layer error = (softmaxOutput - labels)
            matrix softmaxOutput = softmax(z); 
            error = add(softmaxOutput, negate(labels)); // = softmax - labels
        } else {
            // hidden layers error = (W_next^T * delta_next) * f'(z)
            matrix temp = multiply(transpose(nextLayerW), nextError);
            error = matrix(temp.rows, temp.columns);
            for (int i = 0; i < temp.rows; ++i) {
                for (int j = 0; j < temp.columns; ++j) {
                    error[i][j] = temp[i][j] * relu_derivative(z[i][j]);
                }
            }
        }
    }

    // compute gradients
    void computeGradients(const matrix& previousActivation) {
        // ∂J/∂W = δ * A[l-1]^T
        gradientW = multiply(error, transpose(previousActivation));
        // ∂J/∂b = sum of δ across batch (here batch size is 1)
        gradientb = sumColumns(error);
    }

    // update weights and biases
    void updateParameters(double learningRate) {
        W = add(W, multiplyScalar(gradientW, -learningRate));
        b = add(b, multiplyScalar(gradientb, -learningRate));
    }

private:
    // broadcasting bias across columns
    matrix broadcast(const matrix& b, int batchSize) {
        matrix result(b.rows, batchSize);
        for (int i = 0; i < b.rows; ++i) {
            for (int j = 0; j < batchSize; ++j) {
                result[i][j] = b[i][0];
            }
        }
        return result;
    }

    // summing columns
    matrix sumColumns(const matrix& m) {
        matrix result(m.rows, 1);
        for (int i = 0; i < m.rows; ++i) {
            for (int j = 0; j < m.columns; ++j) {
                result[i][0] += m[i][j];
            }
        }
        return result;
    }

    matrix negate(const matrix& m) {
        matrix result(m.rows, m.columns);
        for (int i = 0; i < m.rows; ++i) {
            for (int j = 0; j < m.columns; ++j) {
                result[i][j] = -m[i][j];
            }
        }
        return result;
    }

};



class neuralNetwork {
public:
    int numLayer;
    int currLayerCount;
    layer** layerArr;

    // constructor
    neuralNetwork(int numLayer) : numLayer(numLayer), currLayerCount(0) {
        layerArr = new layer*[numLayer];
        for (int i = 0; i < numLayer; i++){
            layerArr[i] = nullptr;
        }
    }

    // destructor
    ~neuralNetwork(){
        for (int i = 0; i < numLayer; i++) {
            if (layerArr[i] != nullptr){
                delete layerArr[i];
                layerArr[i] = nullptr;
            }
        }
        delete[] layerArr;
    }

    // append a layer to the network
    bool append(int inputSize, int outputSize){
        if (currLayerCount < numLayer) {
            layerArr[currLayerCount] = new layer(inputSize, outputSize);
            currLayerCount++;
            return true;
        } else {
            std::cout << "\nUnable to append to network, max number of layers reached.\n" << std::endl;
            return false;
        }
    }

    // compute the average cost over a batch
    // labels: (outputSize, batchSize) one-hot encoded
    // layerOutput: (outputSize, batchSize)
    double cost(const matrix& layerOutput, const matrix& labels) {
        double epsilon = 1e-9; 
        double totalCost = 0.0;
        int batchSize = layerOutput.columns;
        
        // using cross entropy: -1/batchSize * sum_over_samples(sum_over_output(labels * log(prob)))
        for (int sample = 0; sample < batchSize; sample++) {
            // in one-hot label only one position in each column is 1
            // cost_sample = - log(prob_of_correct_class)
            for (int i = 0; i < layerOutput.rows; i++) {
                if (labels[i][sample] == 1.0) {
                    totalCost += -std::log(layerOutput[i][sample] + epsilon);
                    break;
                }
            }
        }

        return totalCost / batchSize;
    }

    // forward pass can handle a batch of samples
    // x: (inputSize, batchSize)
    matrix forward(const matrix& x){
        matrix activation = x;
        if (currLayerCount == 0){
            std::cerr << "\nError: No layers in the network.\n";
            return matrix(1,1);
        }

        for (int i = 0; i < currLayerCount; i++){
            bool isOutputLayer = (i == currLayerCount - 1);
            activation = layerArr[i]->forward(activation, isOutputLayer);
        }
        return activation;
    }

    // backward pass for a batch of samples
    // x: (inputSize, batchSize)
    // labels: (outputSize, batchSize) one-hot encoded
    // this computes gradients for the entire batch and updates the parameters.
    void backward(const matrix& x, const matrix& labels, double learningRate){
        if (currLayerCount == 0){
            std::cerr << "\nError: No layers in the network.\n";
            return;
        }

        int batchSize = x.columns;

        // compute error for output layer
        // error_out = a_out - labels  (where a_out is last layers activation)
        {
            layer* outputLayer = layerArr[currLayerCount - 1];
            matrix softmaxOutput = softmax(outputLayer->z); 
            matrix outputError = add(softmaxOutput, negate(labels)); // (outputSize, batchSize)
            outputLayer->error = outputError; 
        }

        // compute error for hidden layers
        for (int i = currLayerCount - 2; i >= 0; i--) {
            layer* currentLayer = layerArr[i];
            layer* nextLayer = layerArr[i+1];
            // error_hiden = (W_next^T * error_next) * relu'(z)
            matrix temp = multiply(transpose(nextLayer->W), nextLayer->error); // (currentLayer.outputSize, batchSize)
            matrix hiddenError(temp.rows, temp.columns);
            for (int r = 0; r < temp.rows; r++) {
                for (int c = 0; c < temp.columns; c++) {
                    hiddenError[r][c] = temp[r][c] * relu_derivative(currentLayer->z[r][c]);
                }
            }
            currentLayer->error = hiddenError;
        }

        // update weights and biases for all layers
        // for first layer, previous activation = x
        // for all other layers, previous activation = layerArr[i-1]->a
        for (int i = 0; i < currLayerCount; i++){
            matrix prevActivation = (i == 0) ? x : layerArr[i - 1]->a;
            
            // compute gradients
            matrix gradW = multiply(layerArr[i]->error, transpose(prevActivation));
            matrix gradb = sumColumns(layerArr[i]->error);

            // average gradients over batch
            double invBatch = 1.0 / batchSize;
            gradW = multiplyScalar(gradW, invBatch);
            gradb = multiplyScalar(gradb, invBatch);

            layerArr[i]->gradientW = gradW;
            layerArr[i]->gradientb = gradb;

            // update parameters
            layerArr[i]->updateParameters(learningRate);
        }
    }

private:
    matrix negate(const matrix& m) {
        matrix result(m.rows, m.columns);
        for (int i = 0; i < m.rows; ++i) {
            for (int j = 0; j < m.columns; ++j) {
                result[i][j] = -m[i][j];
            }
        }
        return result;
    }

    matrix sumColumns(const matrix& m) {
        matrix result(m.rows, 1);
        for (int i = 0; i < m.rows; ++i) {
            double sumVal = 0.0;
            for (int j = 0; j < m.columns; ++j) {
                sumVal += m[i][j];
            }
            result[i][0] = sumVal;
        }
        return result;
    }

};
