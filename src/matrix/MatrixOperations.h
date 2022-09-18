#ifndef NEURAL_NETWORK_MATRIXOPERATIONS_H
#define NEURAL_NETWORK_MATRIXOPERATIONS_H
#define NUM_THREADS 6

#include <vector>

using namespace std;

class MatrixOperations {
public:
    static void multiply_matrix_vector_horizontal(const vector<vector<float>> &matrix, const vector<float> &input_vector, vector<float> &output_vector);
    static void multiply_matrix_vector_vertical(const vector<vector<float>> &matrix, const vector<float> &input_vector, vector<float> &output_vector);
    static void propagate_error(const vector<vector<float>> &transition_matrix, const vector<float> &source_error_vector, vector<float> &output_error_vector);
    static void apply_function(vector<float> &vector, std::vector<float> &output_vector, float (*eval_function)(float));
    static void add_vectors(const vector<float> &vector1, const vector<float> &vector2, vector<float> &output_vector);
    static void add_vectors(vector<float> &output_vector, const vector<float> &vector2);
    static float sum(const vector<float> &vector);
    static void compute_and_add_output_weight_deltas(vector<float> &states_below, vector<float> &errors_current, vector<vector<float>> &output_matrix);
    static void compute_and_add_weight_deltas(vector<float> &states_below, vector<float> &errors_current,
                                              const vector<float> &evaluated_activation_derivatives, vector<vector<float>> &output_matrix);
    static void set_value(vector<float> &vector, float x);
    static void get_output_errors(const vector<float> &states, const vector<int> &expected_values, vector<float> &output);
    static void subtract_matrices_with_learning_rate(vector<vector<float>> &output_matrix, vector<vector<float>> &deltas_matrix, float learning_rate, int batch_size);

private:
    static float get_vectors_dot_product(const vector<float> &vector1, const vector<float> &vector2);
};



#endif //NEURAL_NETWORK_MATRIXOPERATIONS_H
