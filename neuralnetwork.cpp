#include <iostream>

using namespace std;


class Neuron {
    public:
        int numberInputs;
        float* inputArray;
        float* weightArray;
        float bias;

        // constructor to initialize neuron class
        Neuron(int size) : numberInputs(size), bias(0) {
            inputArray = new float[numberInputs];
            weightArray = new float[numberInputs];

            // intialize input and weights
            for (int i = 0; i < numberInputs; i ++){
                inputArray[i] = 0;
                weightArray[i] = 1;
            }
        }

        // destructor
        ~Neuron(){
            delete[] inputArray;
            delete[] weightArray;
        } 

        // methods
        float forward();
    protected:
        float weightedSum();
        float activation(float sum);
};

float Neuron::weightedSum() {
    float sum = bias;
    for (int i = 0; i < numberInputs; i++) {
        sum += inputArray[i] * weightArray[i];
    }
    return sum;
}
float Neuron::activation(float sum) {
    // using parametric reLU where alpha = 0.1 | leaky reLU
    float alpha = 0.1;
    if (sum < 0){
        return alpha * sum;
    }
    else {
        return sum;
    }

}
float Neuron::forward() {
    float sum = weightedSum();
    return activation(sum);
}



// testing neuron class
int main() {
    int size = 2;
    Neuron inputNeuron(size);   // create neuron with 2 inputs

    // set inputs manually
    inputNeuron.inputArray[0] = 5.0f;
    inputNeuron.inputArray[1] = 2.0f;

    // set weights manually
    inputNeuron.weightArray[0] = 2.0f;
    inputNeuron.weightArray[1] = 5.0f;

    // set bias
    inputNeuron.bias = 0.5;

    // get result
    float result = inputNeuron.forward();

    // display result
    cout << "Neuron output: " << result << endl;

    return 0;
}