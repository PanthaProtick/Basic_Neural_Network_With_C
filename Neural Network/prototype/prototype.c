#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include<string.h>

typedef struct Layer Layer;

typedef struct{
    double value;
    double activated_value;
    double* weights;
    double* grad_weights;
    double bias;
    double grad_bias;
} Node;

struct Layer {
    Node* nodes;
    int num_nodes;
    Layer* next_layer;
    Layer* prev_layer;
};

// --- Dataset ---
typedef struct {
    double** X;
    double* y;
    int num_samples;
    int num_features;
} Dataset;

Dataset load_csv(const char* filename) {
    Dataset ds = {NULL, NULL, 0, 0};
    FILE* f = fopen(filename, "r");
    if(!f) {
        fprintf(stderr, "Error: could not open file '%s'\n", filename);
        return ds;
    }

    // First pass: count rows and columns
    char line[4096];
    int first = 1;
    while(fgets(line, sizeof(line), f)) {
        if(first) {
            // Count commas to determine number of columns
            int cols = 1;
            for(int i = 0; line[i]; i++) if(line[i] == ',') cols++;
            ds.num_features = cols - 1;  // last column is target
            first = 0;
        }
        ds.num_samples++;
    }

    // Allocate
    ds.X = (double**)malloc(ds.num_samples * sizeof(double*));
    ds.y = (double*)malloc(ds.num_samples * sizeof(double));
    for(int i = 0; i < ds.num_samples; i++) {
        ds.X[i] = (double*)malloc(ds.num_features * sizeof(double));
    }

    // Second pass: read values
    rewind(f);
    int row = 0;
    while(fgets(line, sizeof(line), f) && row < ds.num_samples) {
        char* token = strtok(line, ",");
        for(int col = 0; col < ds.num_features; col++) {
            ds.X[row][col] = atof(token);
            token = strtok(NULL, ",");
        }
        ds.y[row] = atof(token);  // last column is target
        row++;
    }

    fclose(f);
    printf("Loaded '%s': %d samples, %d features\n", filename, ds.num_samples, ds.num_features);
    return ds;
}

void free_dataset(Dataset* ds) {
    for(int i = 0; i < ds->num_samples; i++) {
        free(ds->X[i]);
    }
    free(ds->X);
    free(ds->y);
}

// --- Network ---
double relu(double x) {
    return x > 0 ? x : 0;
}

double relu_derivative(double x) {
    return x > 0 ? 1 : 0;
}

double random_weight_he(int fan_in) {
    double u1 = ((double)rand() + 1.0) / ((double)RAND_MAX + 1.0);
    double u2 = ((double)rand() + 1.0) / ((double)RAND_MAX + 1.0);
    double gaussian = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979 * u2);
    return gaussian * sqrt(2.0 / fan_in);
}

double forward_pass(Layer* current_layer) {
    if(current_layer->next_layer == NULL) {
        return current_layer->nodes[0].activated_value;
    }

    Layer* next = current_layer->next_layer;
    for(int i = 0; i < next->num_nodes; i++) {
        double sum = next->nodes[i].bias;
        for(int j = 0; j < current_layer->num_nodes; j++) {
            sum += current_layer->nodes[j].activated_value * next->nodes[i].weights[j];
        }
        next->nodes[i].value = sum;

        if(next->next_layer == NULL) {
            next->nodes[i].activated_value = sum;
        } else {
            next->nodes[i].activated_value = relu(sum);
        }
    }

    return forward_pass(next);
}

void backward_pass(Layer* current_layer, double* error) {
    if(current_layer->prev_layer == NULL) {
        free(error);
        return;
    }

    Layer* prev = current_layer->prev_layer;
    int is_output = (current_layer->next_layer == NULL);

    double* prev_error = (double*)malloc(prev->num_nodes * sizeof(double));
    for(int j = 0; j < prev->num_nodes; j++) prev_error[j] = 0;

    for(int i = 0; i < current_layer->num_nodes; i++) {
        double act_grad = is_output ? 1.0 : relu_derivative(current_layer->nodes[i].value);
        double grad = error[i] * act_grad;

        current_layer->nodes[i].grad_bias = grad;

        for(int j = 0; j < prev->num_nodes; j++) {
            current_layer->nodes[i].grad_weights[j] = grad * prev->nodes[j].activated_value;
            prev_error[j] += grad * current_layer->nodes[i].weights[j];
        }
    }

    free(error);
    backward_pass(prev, prev_error);
}

void update_weights(Layer* layer, double learning_rate) {
    if(layer == NULL) return;

    for(int i = 0; i < layer->num_nodes; i++) {
        if(layer->prev_layer != NULL) {
            layer->nodes[i].bias -= learning_rate * layer->nodes[i].grad_bias;
            for(int j = 0; j < layer->prev_layer->num_nodes; j++) {
                layer->nodes[i].weights[j] -= learning_rate * layer->nodes[i].grad_weights[j];
            }
        }
    }

    update_weights(layer->next_layer, learning_rate);
}

void init_layer(Layer* layer, int num_nodes, Layer* prev_layer) {
    layer->num_nodes = num_nodes;
    layer->nodes = (Node*)malloc(sizeof(Node) * num_nodes);
    layer->prev_layer = prev_layer;
    layer->next_layer = NULL;

    for(int i = 0; i < num_nodes; i++) {
        layer->nodes[i].value = 0;
        layer->nodes[i].activated_value = 0;
        layer->nodes[i].bias = 0;
        layer->nodes[i].grad_bias = 0;
        layer->nodes[i].weights = NULL;
        layer->nodes[i].grad_weights = NULL;

        if(prev_layer != NULL) {
            int prev_count = prev_layer->num_nodes;
            layer->nodes[i].weights = (double*)malloc(sizeof(double) * prev_count);
            layer->nodes[i].grad_weights = (double*)malloc(sizeof(double) * prev_count);
            layer->nodes[i].bias = random_weight_he(prev_count);

            for(int j = 0; j < prev_count; j++) {
                layer->nodes[i].weights[j] = random_weight_he(prev_count);
                layer->nodes[i].grad_weights[j] = 0;
            }
        }
    }
}

// Builds a fully connected network from an array of layer sizes
Layer* build_network(int* layer_sizes, int num_layers) {
    Layer* layers = (Layer*)malloc(num_layers * sizeof(Layer));

    init_layer(&layers[0], layer_sizes[0], NULL);
    for(int i = 1; i < num_layers; i++) {
        init_layer(&layers[i], layer_sizes[i], &layers[i-1]);
        layers[i-1].next_layer = &layers[i];
    }
    layers[num_layers - 1].next_layer = NULL;

    return layers;
}

void free_network(Layer* layers, int num_layers) {
    for(int l = 0; l < num_layers; l++) {
        for(int i = 0; i < layers[l].num_nodes; i++) {
            free(layers[l].nodes[i].weights);
            free(layers[l].nodes[i].grad_weights);
        }
        free(layers[l].nodes);
    }
    free(layers);
}

void set_input(Layer* input_layer, double* inputs, int count) {
    for(int i = 0; i < count; i++) {
        input_layer->nodes[i].value = inputs[i];
        input_layer->nodes[i].activated_value = inputs[i];
    }
}

void print_predictions(Layer* input_layer, Dataset* ds) {
    printf("\nPredictions:\n");
    for(int i = 0; i < ds->num_samples; i++) {
        set_input(input_layer, ds->X[i], ds->num_features);
        double pred = forward_pass(input_layer);
        printf("Sample %d => Pred: %.4f, Target: %.1f\n", i + 1, pred, ds->y[i]);
    }
}

int main(){
    srand((unsigned int)time(NULL));

    // --- Load data ---
    Dataset ds = load_csv("BostonHousing.csv");
    if(ds.num_samples == 0) return 1;

    // --- Define architecture ---
    int layer_sizes[] = {ds.num_features, 4, 1};
    int num_layers = sizeof(layer_sizes) / sizeof(layer_sizes[0]);

    // --- Build network ---
    Layer* layers = build_network(layer_sizes, num_layers);

    // --- Hyperparameters ---
    double learning_rate = 0.01;
    int epochs = 8000;

    printf("\nBefore training:\n");
    print_predictions(&layers[0], &ds);

    // --- Training loop ---
    for(int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0;

        for(int i = 0; i < ds.num_samples; i++) {
            set_input(&layers[0], ds.X[i], ds.num_features);

            double pred = forward_pass(&layers[0]);
            double err = pred - ds.y[i];
            total_loss += err * err;

            double* output_error = (double*)malloc(sizeof(double));
            output_error[0] = 2.0 * err;

            backward_pass(&layers[num_layers - 1], output_error);
            update_weights(&layers[1], learning_rate);
        }

        if((epoch + 1) % 1000 == 0) {
            printf("Epoch %d, MSE: %.6f\n", epoch + 1, total_loss / ds.num_samples);
        }
    }

    printf("\nAfter training:\n");
    print_predictions(&layers[0], &ds);

    free_network(layers, num_layers);
    free_dataset(&ds);

    return 0;
}