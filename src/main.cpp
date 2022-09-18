#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <random>
#include <chrono>
#include <stdio.h>
#include <omp.h>
#include "main.h"
#include "matrix/MatrixOperations.h"
#include <cassert>

//TODO: POPIS BATCH_ALGORITMU
/*
 *  V kazde vrstve pribyla nova matice - weight_deltas. Zde se budou shromazdovat vsechny zmeny na vahach, ktere jedna iterace sgd vypocita
 *  K update dojde az z prumeru souctu zmen po dokonceni cele davky
 *
 *  1. inicializace weight_deltas ve stejne velikosti jako weights
 *  2. forward propagace
 *  3. vypocet chyby v output neuronech jako (vraceny_vysledek - ocekavana_hodnota)
 *  4. IHNED POTE JE MOZNE VYPOCITAT ZMENU VAHY PRO OUTPUT VEKTORY
 *      . vypocitana chyba * input (input z hidden vrstvy kam smeruje spojeni s vahou, kterou prave aktualizujeme)
 *      . zmena se pricte do matice weight_deltas ulozene v output vrstve
 *  5. propagace chyby do nizsich vrstev
 *      . propaguje se standardnim zpusobem jako chyba * vaha
 *  6. IHNED PO PROPAGACI VYPOCET ZMENY VAH
 *      . vypocitana jako current_error * activation_derivace(raw_input z aktualni vrstvy) * input (z vrstvy o jedno nizsi)
 *      . opet pricist do matice zmen vah
 *  7. po dokonceni jedne davky dojde k update vah
 *      . projdou se vsechny vrstvy, weights -= learning_rate * weight_deltas / batch_size
 *      . JE NUTNE VYNULOVAT weight_deltas po aplikaci
 *
 *
 *  BACKTRACKING PROCHAZI SIT POUZE OD OUTPUTU K INPUTU, NIKOLIV ZPATKY. Chyby se pocitaji uz pri prvnim pruchodu.
 */

using namespace std;

chrono::high_resolution_clock::time_point t;

int main() {
    t = chrono::high_resolution_clock::now();
    std::mt19937 gen;

//    solve_mnist(gen);
    solve_xor(gen);
//    solve_xor_5(gen);
//    test_matrix_library();

    cout << "Runtime: " << chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - t).count() << " microsec" << endl;
    cout << "Runtime: " << chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - t).count() << " sec" << endl;
    return 0;
}

void test_matrix_library() {
//    vector<float> vector1{10, 20, 30, 10};
//    vector<float> vector2{1, 2, 3, 4};
//    vector<float> result_vector1(vector1.size());
//    MatrixOperations::add_vectors(vector1, vector2, result_vector1);
//    MatrixOperations::subtract_vector_and_float_product_to_vector(vector1, vector2, 2, 2, 2);
}

void solve_xor(std::mt19937 gen) {
    vector<Layer> network = create_network(vector<int>{2, 4, 4, 2}, gen);
    Dataset dataset = load_dataset("xor", 1);
    train_sgd(network, dataset);
//    print_network(network);

    int category = 0;
    for (int i = 0; i < dataset.test_vectors.size(); i++) {
        const vector<float> input = dataset.test_vectors[i];
        int label = dataset.test_labels[i];
        category = predict_category(network, input);

        cout << "Input [" << input[0] << ", " << input[1] << "]. Output 0: " << network.back().states[0] << " Output 1: " << network.back().states[1] << ", so result is "<< category << " when expected " << label << endl;
    }
}

void solve_xor_5(std::mt19937 gen) {
    vector<Layer> network = create_network(vector<int>{5, 4, 4, 2}, gen);

    Dataset dataset = load_dataset("xor_5", 1);
    train_sgd(network, dataset);
//    print_network(network);

    int category = 0;
    for (int i = 0; i < dataset.test_vectors.size(); i++) {
        const vector<float> input = dataset.test_vectors[i];
        int label = dataset.test_labels[i];
        category = predict_category(network, input);

        if (to_string(network.back().states[0]) == "-nan(ind)") {
            cout << "-nan(ind) detected" << endl;
        }
        cout << "Output 0: " << network.back().states[0] << " Output 1: " << network.back().states[1] << ", so result is "<< category << " when expected " << label << endl;
    }
}

void solve_mnist(std::mt19937 generator)  {
    Dataset dataset = load_dataset("fashion_mnist", 255);
    cout << "Dataset loaded after " << chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - t).count() << " sec" << endl;
    vector<Layer> network = create_network(vector<int>{784, 128, 10}, generator);
    train_sgd(network, dataset);

    cout << "Training finished after " << chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - t).count() << " sec" << endl;


    int category = 0;
    ofstream train_predictions("../trainPredictions");
    ofstream actual_test_predictions("../actualTestPredictions");

    for (auto & input : dataset.test_vectors) {
        category = predict_category(network, input);
        actual_test_predictions << to_string(category) + "\n";
    }

    for (auto & input : dataset.train_vectors) {
        category = predict_category(network, input);
        train_predictions << to_string(category) + "\n";
    }

    train_predictions.close();
    actual_test_predictions.close();
}

void train_sgd(vector<Layer> &network, const Dataset &dataset) {
    vector<int> expected = vector<int>(network.back().size);
    for (int ep = 0; ep < EPOCHS; ep++){
        for (int iteration = 0; iteration < dataset.train_vectors.size(); iteration++) {
            for (int batch = 0; batch < BATCH_SIZE; batch++) {
                const vector<float> input = dataset.train_vectors[iteration];
                int label = dataset.train_labels[iteration];
                category_to_vector(label, expected);
                evaluate(network, input);
                backpropagate(network, expected);
            }

            update_weights(network);
            if (to_string(network[0].errors[0]) == "-nan(ind)") {
//                print_network(network);
                assert(to_string(network[0].errors[0]) != "-nan(ind)");
            }
            if (ep % 100 == 0 && iteration == 0) {
                cout << "Epoch " << ep << ", iteration " << iteration << ", time " << chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - t).count() << " sec" << endl;
            }
        }

//        cout << "Performing evaluation of epoch " << ep << "..." << endl;
//        cout << "Accuracy: " << evaluate_model(network, dataset.test_vectors, dataset.test_labels) << "%" << endl;
    }
}

void backpropagate(vector<Layer> &network, vector<int> &expected) {
    Layer *lower_layer = &network[network.size() - 2];
    Layer *current_layer = &network.back();

    // compute initial errors as (output - target) and store them
    MatrixOperations::get_output_errors(network.back().states, expected, network.back().errors);

    // update weights with softmax derivative - still doesn't work
//    vector<float> derivatives = vector<float>(current_layer->raw_states.size());
//    softmax_derivate(current_layer->raw_states, derivatives);
//    // add output layer deltas - in weight_deltas is the "guilt amount" that helped with the error
//    MatrixOperations::compute_and_add_weight_deltas(lower_layer->states, current_layer->errors, derivatives, current_layer->weight_deltas);

//    // add output layer deltas - in weight_deltas is the "guilt amount" that helped with the error
    MatrixOperations::compute_and_add_output_weight_deltas(lower_layer->states, current_layer->errors, current_layer->weight_deltas);

    // propagate errors from the output layer
    MatrixOperations::propagate_error(current_layer->weights, current_layer->errors, lower_layer->errors);

    for (int i = network.size() - 2; i > 0; --i) {
        lower_layer = &network[i - 1];
        current_layer = &network[i];

        // take errors from upper layer and multiply them with weights along the way
        MatrixOperations::propagate_error(current_layer->weights, current_layer->errors, lower_layer->errors);

        // compute and save deltas
        // TODO DON'T create new vector each time!
        vector<float> derivatives = vector<float>(current_layer->raw_states.size());
        MatrixOperations::apply_function(current_layer->raw_states, derivatives, current_layer->eval_function_derivative);
        MatrixOperations::compute_and_add_weight_deltas(lower_layer->states, current_layer->errors, derivatives, current_layer->weight_deltas);
    }
}


void update_weights(vector<Layer> &network) {
    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 1; i < network.size(); ++i) {
        MatrixOperations::subtract_matrices_with_learning_rate(network[i].weights, network[i].weight_deltas, network[i].learning_rate, BATCH_SIZE);
    }
}


int predict_category(vector<Layer> &network, const vector<float> &input) {
    evaluate(network, input);
    return vector_to_category(network.back().states);
}

void evaluate(vector<Layer> &network, const vector<float> &input) {
    // TODO eventually delete this check for better performance
    assert(input.size() == network[0].size);

    // set inputs to the first layer
    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < input.size(); ++i) {
        network.front().states[i] = input[i];
    }

    Layer *previous_layer;
    Layer *current_layer;
    // set new states for each layer based on biases, weigths and states from previous layer
    for (int i = 1; i < network.size(); i++) {
        previous_layer = &network[i - 1];
        current_layer = &network[i];

        MatrixOperations::multiply_matrix_vector_horizontal(current_layer->weights, previous_layer->states,
                                                            current_layer->raw_states);
        MatrixOperations::add_vectors(current_layer->raw_states, current_layer->biases);

        if (i == network.size() - 1) {
            softmax(current_layer->raw_states, current_layer->states);
        }
        else {
            MatrixOperations::apply_function(current_layer->raw_states, current_layer->states, current_layer->eval_function);
        }
    }

}

int vector_to_category(const vector<float> &values) {
    if (values.empty()) {
        return 0;
    }

    // find category (neuron with index corresponding to category) with maximal value
    float max_value = values[0];
    int category = 0;
    for (int i = 0; i < values.size(); ++i) {
        if (values[i] > max_value) {
            max_value = values[i];
            category = i;
        }
    }
    return category;
}

void category_to_vector(const int category, vector<int> &output) {
    for (int i = 0; i < output.size(); ++i) {
        if (category == i) {
            output[i] = 1;
        }
        else {
            output[i] = 0;
        }
    }
}


float leaky_relu(const float x) {
    return x >= 0 ? x : 0.05f * x;
}

float leaky_relu_derivative(const float x) {
    return x >= 0 ? 1 : 0.05f;
}

void softmax(const vector<float> &input, vector<float> &output) {
    float sum = 0;
    float max_elem = *std::max_element(input.begin(), input.end());

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < input.size(); ++i) {
        sum += exp(input[i] - max_elem);
    }

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < input.size(); i++)
        output[i] = exp(input[i] - max_elem) / sum;
}

void softmax_derivate(const vector<float> &raw_states, vector<float> &derivated_states) {
    vector<float> states(raw_states.size());
    softmax(raw_states, states);

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < derivated_states.size(); i++) {
        derivated_states[i] = states[i] * (1 - states[i]);
    }
}

float cross_entropy(vector<float> &output, vector<int> &expected) {
    float result = 0;
    for (int i = 0; i < output.size(); i++) {
        result += log(output[i] + 1e-15f) * expected[i];
    }

    return -result;
}


vector<Layer> create_network(const vector<int>& layers_sizes, std::mt19937 generator) {
    vector<Layer> network;

    uniform_real_distribution<float> distribution(-1, 1);
    float rand;
    for (int i = 0; i < layers_sizes.size(); ++i)
    {
        Layer new_layer;
        new_layer.size = layers_sizes[i];
        new_layer.level = i;
        new_layer.learning_rate = 0.1;
        new_layer.eval_function = &leaky_relu;
        new_layer.eval_function_derivative = &leaky_relu_derivative;
        new_layer.states.reserve(new_layer.size);
        new_layer.raw_states.reserve(new_layer.size);
        new_layer.biases.reserve(new_layer.size);
        new_layer.bias_deltas.reserve(new_layer.size);
        new_layer.errors.reserve(new_layer.size);
        for (int j = 0; j < new_layer.size; ++j) {
            new_layer.states.emplace_back(0);
            new_layer.raw_states.emplace_back(0);
            new_layer.errors.emplace_back(0);
            rand = distribution(generator);
            new_layer.biases.emplace_back(rand);
            new_layer.bias_deltas.emplace_back(0.0f);
        }

        if (i > 0) {
            int rows = layers_sizes[i];
            int cols = layers_sizes[i-1];
            new_layer.weights.reserve(rows);
            new_layer.weight_deltas.reserve(rows);
            for (int j = 0; j < rows; ++j) {
                vector<float> row;
                vector<float> zero_row;
                row.reserve(cols);
                zero_row.reserve(cols);
                for (int k = 0; k < cols; ++k) {
                    rand = distribution(generator);
                    row.emplace_back(rand * sqrt(2.0f / (cols + rows)));
                    zero_row.emplace_back(0.0f);
                }
                new_layer.weights.emplace_back(row);
                new_layer.weight_deltas.emplace_back(zero_row);
            }
        }

        network.push_back(new_layer);
    }

    return network;
}

Dataset load_dataset(const string &dataset_name, const float input_range) {
    cout << "Loading dataset " << dataset_name << "..." << endl;
    Dataset dataset;
    string path = "../data/" + dataset_name;
    string line, pixel;

    //===================== test vectors ========================
    ifstream test_vectors(path + "_test_vectors.csv");
    while(getline(test_vectors, line))
    {
        vector<float> row;
        stringstream str(line);

        while(getline(str, pixel, ','))
            row.emplace_back(stof(pixel) / input_range);

        dataset.test_vectors.emplace_back(row);
    }
    test_vectors.close();

    //========================= test labels ============================

    ifstream test_labelS(path + "_test_labels.csv");
    while(getline(test_labelS, line))
    {
        dataset.test_labels.emplace_back(stoi(line));
    }
    test_labelS.close();

    //============================== train vectors =================================

    ifstream train_vectors(path + "_train_vectors.csv");
    while(getline(train_vectors, line))
    {
        vector<float> row;
        stringstream str(line);

        while(getline(str, pixel, ','))
            row.emplace_back(stof(pixel) / input_range);

        dataset.train_vectors.emplace_back(row);
    }
    train_vectors.close();

    //===============================================================

    ifstream train_labels(path + "_train_labels.csv");
    while(getline(train_labels, line))
    {
        dataset.train_labels.emplace_back(stoi(line));
    }
    train_labels.close();

    cout << "Dataset loaded." << endl;
    return dataset;
}

void print_network(const vector<Layer> &network) {
    for (const auto & layer : network) {
        cout << "================ Layer: " << layer.level << " (size: " << layer.size << ") ====================" << endl;

        cout << endl << "Raw states:"<< endl;
        for (const float & state : layer.raw_states) {
            cout << state << " ";
        }
        cout << endl << "States:"<< endl;
        for (const float & state : layer.states) {
            cout << state << " ";
        }
        cout << endl << "Biases:"<< endl;
        for (const float & bias : layer.biases) {
            cout << bias << " ";
        }
        cout << endl << "Biase deltas:"<< endl;
        for (const float & bias_delta : layer.bias_deltas) {
            cout << bias_delta << " ";
        }
        cout << endl << "Errors:"<< endl;
        for (const float & error : layer.errors) {
            cout << error << " ";
        }
        cout << endl << "Weights:" << endl;
        for (const auto & weight : layer.weights) {
            for (const float & j : weight) {
                cout << j << " ";
            }
            cout << endl;
        }
        cout << endl << "Weight deltas:" << endl;
        for (const auto & weight_delta : layer.weight_deltas) {
            for (const float & j : weight_delta) {
                cout << j << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    cout << "============================================================================================================" << endl << endl << endl;
}


