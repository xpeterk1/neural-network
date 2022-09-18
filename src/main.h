#define BATCH_SIZE 2
#define EPOCHS 5000
// TODO change to 16 on aisa
#define NUM_THREADS 6

using namespace std;

struct Layer {

    // States of all neurons from the layer
    vector<float> states;

    // States prior to eval function application
    vector<float> raw_states;

    // Biases of all neurons from the layer
    vector<float> biases;

    // Vector to accumulate the deltas
    vector<float> bias_deltas;

    // Errors of all neurons from the layer
    vector<float> errors;

    // Weights of all neurons (weights are coming to the neuron to the lower layer)
    vector<vector<float>> weights;

    // Matrix to accumulate all update deltas
    vector<vector<float>> weight_deltas;

    // Size of the layer
    int size;

    // Level of the layer
    int level;

    // Output evaluation function of the layer
    float (*eval_function)(float);

    // Derivative function of the evaluation function
    float (*eval_function_derivative)(float);

    // Learning rate for backpropagation
    float learning_rate;
};

struct Dataset {

    // Training values
    vector<vector<float>> train_vectors;
    vector<int> train_labels;

    // Testing values
    vector<vector<float>> test_vectors;
    vector<int> test_labels;
};

void solve_mnist(std::mt19937 generator);

void solve_xor(std::mt19937 generator);

void solve_xor_5(std::mt19937 generator);

/**
 * Create network from array of layer sizes
 * @param layers_sizes Sizes of individual layers, first value is the size of the input layer, last value is the number our output neurons
 * @return Created network
 */
vector<Layer> create_network(const vector<int> &layers_sizes, std::mt19937 generator);

/**
 * Evaluate the input values in network. Returns category.
 * @param network Trained network
 * @param inputs Vector of inputs values
 * @return category
 */
int predict_category(vector<Layer> &network, const vector<float> &input);

/**
 * Convert values/states of output neurons to category. The highest value is taken as the result category.
 * @param values vector of values/states of output neurons
 * @return category of network evaluation
 */
int vector_to_category(const vector<float> &values);

/**
 * Evaluate the input values in network and sets neurons states.
 * @param network Trained network
 * @param inputs Vector of inputs values
 */
void evaluate(vector<Layer> &network, const vector<float> &input);

void train_sgd(vector<Layer> &network, const Dataset &dataset);

void softmax(const vector<float> &input, vector<float> &output);

void softmax_derivate(const vector<float> &raw_states, vector<float> &derivated_states);

void test_matrix_library();

void print_network(const vector<Layer> &network);

/**
 * Pass the network backwards and store updates to the weights in the vectors. After BATCH_SIZE iterations, call update
 * weights to actually update the weight matrix
 * @param network Network
 */
void backpropagate(vector<Layer> &network, vector<int> &expected);

/**
 * Pass through the network and update the weights using precomputed sum of weight deltas
 * @param network Network
 */
void update_weights(vector<Layer> &network);

Dataset load_dataset(const string &dataset_name, float input_range);

float leaky_relu_derivative(float x);

float leaky_relu(float x);

float cross_entropy(vector<float> &output, vector<int> &expected);

void category_to_vector(int category, vector<int> &output);

float evaluate_model(vector<Layer> &network, vector<vector<float>> test_vec, vector<int> test_lab);
