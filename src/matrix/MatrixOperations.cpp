#include "MatrixOperations.h"
#include <iostream>
#include <vector>
#include <stdio.h>
#include <omp.h>
#include <cassert>

using namespace std;

void MatrixOperations::multiply_matrix_vector_horizontal(const vector<vector<float>> &matrix, const vector<float> &input_vector, vector<float> &output_vector) {
    // TODO eventulay delete this check for better performance
    assert(matrix[0].size() == input_vector.size() && matrix.size() == output_vector.size());
    assert(matrix[0].size() == matrix[matrix.size()-1].size());

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < matrix.size(); ++i) {
        output_vector[i] = get_vectors_dot_product(matrix[i], input_vector);
    }
}

void MatrixOperations::multiply_matrix_vector_vertical(const vector<vector<float>> &matrix, const vector<float> &input_vector, vector<float> &output_vector) {
    // TODO eventulay delete this check for better performance
    assert(matrix.size() == input_vector.size() && matrix[0].size() == output_vector.size());
    assert(matrix[0].size() == matrix[matrix.size()-1].size());

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < output_vector.size(); ++i) {
        output_vector[i] = 0;
        for (int j = 0; j < matrix.size(); ++j) {
            output_vector[i] += input_vector[j] * matrix[j][i];
        }
    }
}

float MatrixOperations::get_vectors_dot_product(const vector<float> &vector1, const vector<float> &vector2) {
    // TODO eventulay delete this check for better performance
    assert(vector1.size() == vector2.size() );

    float result = 0;
    #pragma omp parallel for reduction(+:result) num_threads(NUM_THREADS)
    for (int i = 0; i < vector1.size(); ++i) {
        result += vector1[i] * vector2[i];
    }
    return result;
}

void MatrixOperations::add_vectors(const vector<float> &vector1, const vector<float> &vector2, vector<float> &output_vector) {
    // TODO eventulay delete this check for better performance
    assert(vector1.size() == vector2.size() && vector1.size() == output_vector.size());

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < vector1.size(); ++i) {
        output_vector[i] = vector1[i] + vector2[i];
    }
}

void MatrixOperations::add_vectors(vector<float> &output_vector, const vector<float> &vector2) {
    // TODO eventulay delete this check for better performance
    assert(output_vector.size() == vector2.size());

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < output_vector.size(); ++i) {
        output_vector[i] += vector2[i];
    }
}

void MatrixOperations::apply_function(vector<float> &vector, std::vector<float> &output_vector, float (*eval_function)(float)) {
    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < output_vector.size(); i++) {
        output_vector[i] = eval_function(vector[i]);
    }
}

float MatrixOperations::sum(const vector<float> &vector) {
    float sum = 0;
    #pragma omp parallel for reduction(+:sum) num_threads(NUM_THREADS)
    for (int i = 0; i < vector.size(); ++i) {
        sum += vector[i];
    }
    return sum;
}

//void MatrixOperations::subtract_vector_and_float_product_to_vector(vector<float> &output_vector, const vector<float> &vector, float f) {
//    // TODO eventulay delete this check for better performance
//    assert(output_vector.size() == vector.size());
//
//    #pragma omp parallel for num_threads(NUM_THREADS)
//    for (int i = 0; i < output_vector.size(); ++i) {
//        output_vector[i] -= vector[i] * f;
//    }
//}

void MatrixOperations::set_value(vector<float> &vector, const float x) {
    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < vector.size(); ++i) {
        vector[i] = x;
    }
}


void MatrixOperations::get_output_errors(const vector<float> &states, const vector<int> &expected_values, vector<float> &output) {
    // TODO eventually delete this check for better performance
    assert(states.size() == expected_values.size());
    assert(output.size() == expected_values.size());

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < states.size(); ++i) {
        output[i] = states[i] - expected_values[i];
    }
}

void MatrixOperations::compute_and_add_output_weight_deltas(vector<float> &states_below, vector<float> &errors_current, vector<vector<float>> &output_matrix) {
    // TODO eventually delete this check for better performance
    assert(output_matrix.size() == errors_current.size());
    assert(output_matrix[0].size() == states_below.size());

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < states_below.size(); i++) {
        for (int j = 0; j < errors_current.size(); j++)
        {
            output_matrix[j][i] += states_below[i] * errors_current[j];
        }
    }
}

void MatrixOperations::compute_and_add_weight_deltas(vector<float> &states_below, vector<float> &errors_current,
                                                     const vector<float> &evaluated_activation_derivatives, vector<vector<float>> &output_matrix) {
    // TODO eventually delete this check for better performance
    assert(output_matrix.size() == errors_current.size());
    assert(output_matrix[0].size() == states_below.size());

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < states_below.size(); i++) {
        for (int j = 0; j < errors_current.size(); j++)
        {
            output_matrix[j][i] += states_below[i] * errors_current[j] * evaluated_activation_derivatives[j];
        }
    }
}

void MatrixOperations::propagate_error(const vector<vector<float>> &matrix, const vector<float> &input_vector,
                                       vector<float> &output_vector) {
    // TODO eventually delete this check for better performance
    assert(matrix.size() == input_vector.size());
    assert(matrix[0].size() == output_vector.size());

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < output_vector.size(); i++) {
        output_vector[i] = 0;
        for (int j = 0; j < input_vector.size(); j++){
            output_vector[i] += input_vector[j] * matrix[j][i];
        }
    }
}

void MatrixOperations::subtract_matrices_with_learning_rate(vector<vector<float>> &output_matrix, vector<vector<float>> &deltas_matrix, float learning_rate, int batch_size) {
    // TODO eventually delete this check for better performance
    assert(output_matrix.size() == deltas_matrix.size());
    assert(output_matrix[0].size() == deltas_matrix[0].size());

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < output_matrix.size(); i++) {
        for (int j = 0; j < output_matrix[0].size(); j++)
        {
            output_matrix[i][j] -= learning_rate * (deltas_matrix[i][j] / (float)batch_size);
            deltas_matrix[i][j] = 0;
        }
    }
}
