#include "matrixStructure.cpp"
#include <random>
#include <cmath>

double relu(double x) { return (x > 0) ? x : 0; }
double relu_derivative(double x) { return (x > 0) ? 1 : 0; }
double sigmoid(double x) { return 1 / (1 + exp(-x)); }
double sigmoid_derivative(double x) { double a = sigmoid(x); return a * (1 - a); }

class layer {
public:
    int inputSize;
    int outputSize;
    matrix W; // weight matrix
    matrix b; // bias matrix

    // constructor
    layer(int input_size, int output_size)
        : inputSize(input_size), outputSize(output_size), W(outputSize, inputSize), b(outputSize, 1) {
        initialize_weights();
        initialize_bias();
    }

    // he initialization for weights
    void initialize_weights(){
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dist(0.0, std::sqrt(2.0 / inputSize));
        for (int i = 0; i < outputSize; i++){
            for (int j = 0; j < inputSize; j++){
                W[i][j] = dist(gen);
            }
        }
    }

    // initialize biases close to zero
    void initialize_bias(){
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> bias_dist(-0.01, 0.01);
        for (int i = 0; i < outputSize; i++) {
            b[i][0] = bias_dist(gen);
        }
    }

    // forward pass with ReLU activation
    matrix forward(const matrix& x) {
        matrix z = add(multiply(W, x), b);
        matrix output(W.rows, 1);
        for (int i = 0; i < W.rows; i++) {
            output[i][0] = relu(z[i][0]);
        }
        return output;
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
            delete layerArr[i];
            layerArr[i] = nullptr;
        }
        delete[] layerArr;
    }

    // append a layer to the network
    bool append(const layer& L){
        if (currLayerCount < numLayer) {
            layerArr[currLayerCount] = new layer(L);
            currLayerCount++;
            return true;
        } else {
            std::cout << "\n Unable to append to network, max number of layers reached.\n" << std::endl;
            return false;
        }
    }    

    // where x is the input matrix
    matrix forward(const matrix& x){
        matrix activation = x;
        for (int i = 0; i < numLayer; i++){
            activation = layerArr[i]->forward(activation);
        }
        return activation;
    }
};