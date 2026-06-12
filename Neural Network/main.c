#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

// ─── STRUCTS ──────────────────────────────────────────────────────────────────

typedef struct Layer Layer;

typedef struct {
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

// ─── ACTIVATION ───────────────────────────────────────────────────────────────

double relu(double x)            { return x > 0.0 ? x : 0.0; }
double relu_derivative(double x) { return x > 0.0 ? 1.0 : 0.0; }

// ─── WEIGHT INIT ──────────────────────────────────────────────────────────────

// Box-Muller Gaussian
double random_weight_he(int fan_in) {
    double u1 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
    double u2 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
    double g  = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979 * u2);
    return g * sqrt(2.0 / fan_in);
}

// ─── LAYER / NETWORK ──────────────────────────────────────────────────────────

void init_layer(Layer* layer, int num_nodes, Layer* prev_layer) {
    layer->num_nodes  = num_nodes;
    layer->nodes      = (Node*)malloc(num_nodes * sizeof(Node));
    layer->prev_layer = prev_layer;
    layer->next_layer = NULL;

    for (int i = 0; i < num_nodes; i++) {
        layer->nodes[i].value           = 0.0;
        layer->nodes[i].activated_value = 0.0;
        layer->nodes[i].bias            = 0.0;
        layer->nodes[i].grad_bias       = 0.0;
        layer->nodes[i].weights         = NULL;
        layer->nodes[i].grad_weights    = NULL;

        if (prev_layer != NULL) {
            int n = prev_layer->num_nodes;
            layer->nodes[i].weights      = (double*)malloc(n * sizeof(double));
            layer->nodes[i].grad_weights = (double*)malloc(n * sizeof(double));
            layer->nodes[i].bias         = random_weight_he(n);
            for (int j = 0; j < n; j++) {
                layer->nodes[i].weights[j]      = random_weight_he(n);
                layer->nodes[i].grad_weights[j] = 0.0;
            }
        }
    }
}

// layer_sizes[0] = num inputs, layer_sizes[num_layers-1] = num outputs
Layer* build_network(int* layer_sizes, int num_layers) {
    Layer* layers = (Layer*)malloc(num_layers * sizeof(Layer));
    init_layer(&layers[0], layer_sizes[0], NULL);
    for (int i = 1; i < num_layers; i++) {
        init_layer(&layers[i], layer_sizes[i], &layers[i - 1]);
        layers[i - 1].next_layer = &layers[i];
    }
    layers[num_layers - 1].next_layer = NULL;
    return layers;
}

void free_network(Layer* layers, int num_layers) {
    for (int l = 0; l < num_layers; l++) {
        for (int i = 0; i < layers[l].num_nodes; i++) {
            free(layers[l].nodes[i].weights);
            free(layers[l].nodes[i].grad_weights);
        }
        free(layers[l].nodes);
    }
    free(layers);
}

void set_input(Layer* input_layer, double* inputs, int count) {
    for (int i = 0; i < count; i++) {
        input_layer->nodes[i].value = inputs[i];
        input_layer->nodes[i].activated_value = inputs[i];
    }
}

// ─── FORWARD PASS ─────────────────────────────────────────────────────────────

double forward_pass(Layer* layer) {
    if (layer->next_layer == NULL)
        return layer->nodes[0].activated_value;

    Layer* next = layer->next_layer;
    int is_output = (next->next_layer == NULL);

    for (int i = 0; i < next->num_nodes; i++) {
        double sum = next->nodes[i].bias;
        for (int j = 0; j < layer->num_nodes; j++)
            sum += layer->nodes[j].activated_value * next->nodes[i].weights[j];

        next->nodes[i].value           = sum;
        next->nodes[i].activated_value = is_output ? sum : relu(sum);
    }
    return forward_pass(next);
}

// ─── BACKWARD PASS ────────────────────────────────────────────────────────────

void backward_pass(Layer* current_layer, double* error) {
    if (current_layer->prev_layer == NULL) {
        free(error);
        return;
    }

    Layer* prev = current_layer->prev_layer;
    int is_output = (current_layer->next_layer == NULL);

    double* prev_error = (double*)calloc(prev->num_nodes, sizeof(double));

    for (int i = 0; i < current_layer->num_nodes; i++) {
        double act_grad = is_output ? 1.0 : relu_derivative(current_layer->nodes[i].value);
        double delta = error[i] * act_grad;

        current_layer->nodes[i].grad_bias = delta;

        for (int j = 0; j < prev->num_nodes; j++) {
            current_layer->nodes[i].grad_weights[j] = delta * prev->nodes[j].activated_value;
            prev_error[j] += delta * current_layer->nodes[i].weights[j];
        }
    }

    // Free the incoming error before recursing.
    free(error);
    backward_pass(prev, prev_error);
}

// ─── WEIGHT UPDATE ────────────────────────────────────────────────────────────

void update_weights(Layer* layer, double learning_rate) {
    if (layer == NULL) return;

    // Input layer has no weights to update.
    if (layer->prev_layer != NULL) {
        for (int i = 0; i < layer->num_nodes; i++) {
            layer->nodes[i].bias -= learning_rate * layer->nodes[i].grad_bias;
            for (int j = 0; j < layer->prev_layer->num_nodes; j++)
                layer->nodes[i].weights[j] -= learning_rate * layer->nodes[i].grad_weights[j];
        }
    }
    update_weights(layer->next_layer, learning_rate);
}

// ─── CSV LOADER ───────────────────────────────────────────────────────────────

typedef struct {
    double** features;
    double* labels;
    int entries;
    int num_features;
} Dataset;

int count_char(const char* str, char c) {
    int n = 0;
    while (*str) { if (*str++ == c) n++; }
    return n;
}

// Counts data rows (skips the header line already consumed by caller).
int count_rows(FILE* f) {
    int rows = 0;
    char line[4096];
    fgets(line, sizeof(line), f);          // skip header
    while (fgets(line, sizeof(line), f))
        if (strlen(line) > 1) rows++;
    rewind(f);
    return rows;
}

Dataset load_csv(const char* filename) {
    Dataset ds = {NULL, NULL, 0, 0};
    FILE* f = fopen(filename, "r");
    if (!f) { printf("Error: Could not open '%s'\n", filename); return ds; }

    char line[4096];
    if (!fgets(line, sizeof(line), f)) {
        printf("Error: Empty file\n"); fclose(f); return ds;
    }

    int cols        = count_char(line, ',') + 1;
    ds.num_features = cols - 1;
    ds.entries      = count_rows(f);
    fgets(line, sizeof(line), f);          // skip header again after rewind

    ds.features = (double**)malloc(ds.entries * sizeof(double*));
    for (int i = 0; i < ds.entries; i++)
        ds.features[i] = (double*)malloc(ds.num_features * sizeof(double));
    ds.labels = (double*)malloc(ds.entries * sizeof(double));

    int row = 0;
    while (fgets(line, sizeof(line), f) && row < ds.entries) {
        if (strlen(line) <= 1) continue;
        char* tok = strtok(line, ",");
        int   col = 0;
        while (tok) {
            double v = atof(tok);
            if (col < ds.num_features) ds.features[row][col] = v;
            else                       ds.labels[row]        = v;
            col++;
            tok = strtok(NULL, ",");
        }
        row++;
    }
    fclose(f);
    printf("Loaded '%s': %d entries, %d features\n", filename, ds.entries, ds.num_features);
    return ds;
}

void free_dataset(Dataset* ds) {
    for (int i = 0; i < ds->entries; i++) free(ds->features[i]);
    free(ds->features);
    free(ds->labels);
}

// ─── SHUFFLE ──────────────────────────────────────────────────────────────────

void shuffle_dataset(Dataset* ds) {
    for (int i = ds->entries - 1; i > 0; i--) {
        int j = rand() % (i + 1);

        double* tmp_row  = ds->features[i];
        ds->features[i]  = ds->features[j];
        ds->features[j]  = tmp_row;

        double tmp_lbl  = ds->labels[i];
        ds->labels[i]   = ds->labels[j];
        ds->labels[j]   = tmp_lbl;
    }
    printf("Dataset shuffled\n");
}

// ─── TRAIN / TEST SPLIT ───────────────────────────────────────────────────────

typedef struct { double** features; double* labels; int entries; } Split;
typedef struct { Split train; Split test; } TrainTestSplit;

TrainTestSplit train_test_split(Dataset* ds, double train_ratio) {
    TrainTestSplit tts;
    int train_size = (int)(ds->entries * train_ratio);

    tts.train.features = ds->features;
    tts.train.labels   = ds->labels;
    tts.train.entries  = train_size;

    tts.test.features  = ds->features + train_size;
    tts.test.labels    = ds->labels   + train_size;
    tts.test.entries   = ds->entries  - train_size;

    printf("Train size: %d | Test size: %d\n\n", tts.train.entries, tts.test.entries);
    return tts;
}

// ─── SCALING ──────────────────────────────────────────────────────────────────

typedef struct { double min; double max; } ScaleParams;

ScaleParams* fit_scale(double** arr, int entries, int num_features) {
    ScaleParams* p = (ScaleParams*)malloc(num_features * sizeof(ScaleParams));
    for (int j = 0; j < num_features; j++) {
        p[j].min = p[j].max = arr[0][j];
        for (int i = 1; i < entries; i++) {
            if (arr[i][j] < p[j].min) p[j].min = arr[i][j];
            if (arr[i][j] > p[j].max) p[j].max = arr[i][j];
        }
    }
    return p;
}

void apply_scale(double** arr, int entries, int num_features, ScaleParams* p) {
    for (int j = 0; j < num_features; j++) {
        double rng = p[j].max - p[j].min;
        if (rng == 0.0) continue;
        for (int i = 0; i < entries; i++)
            arr[i][j] = (arr[i][j] - p[j].min) / rng;
    }
}

ScaleParams fit_label_scale(double* labels, int entries) {
    ScaleParams p = { labels[0], labels[0] };
    for (int i = 1; i < entries; i++) {
        if (labels[i] < p.min) p.min = labels[i];
        if (labels[i] > p.max) p.max = labels[i];
    }
    printf("Label scale | Min: %.4f | Max: %.4f\n", p.min, p.max);
    return p;
}

void apply_label_scale(double* labels, int entries, ScaleParams* p) {
    double rng = p->max - p->min;
    if (rng == 0.0) return;
    for (int i = 0; i < entries; i++)
        labels[i] = (labels[i] - p->min) / rng;
}

double unscale(double val, ScaleParams* p) {
    return val * (p->max - p->min) + p->min;
}

// ─── EVALUATE ─────────────────────────────────────────────────────────────────

double evaluate(Layer* layers, int num_features,
                Split* split, ScaleParams* label_scale, int print_preds) {
    double loss = 0.0;
    for (int i = 0; i < split->entries; i++) {
        set_input(&layers[0], split->features[i], num_features);
        double pred = forward_pass(&layers[0]);
        double err  = pred - split->labels[i];
        loss += err * err;

        if (print_preds) {
            double pred_real   = unscale(pred, label_scale);
            double actual_real = unscale(split->labels[i], label_scale);
            printf("Predicted: %8.4f | Actual: %8.4f | Error: %8.4f\n",
                   pred_real, actual_real, pred_real - actual_real);
        }
    }
    return loss / split->entries;
}

// Compute coefficient of determination (R^2) on unscaled labels
double r2_score(Layer* layers, int num_features, Split* split, ScaleParams* label_scale) {
    if (split->entries == 0) return 0.0;
    double ss_res = 0.0;
    double ss_tot = 0.0;
    double mean = 0.0;
    for (int i = 0; i < split->entries; i++)
        mean += unscale(split->labels[i], label_scale);
    mean /= split->entries;

    for (int i = 0; i < split->entries; i++) {
        set_input(&layers[0], split->features[i], num_features);
        double pred = forward_pass(&layers[0]);
        double pred_real   = unscale(pred, label_scale);
        double actual_real = unscale(split->labels[i], label_scale);

        double diff = actual_real - pred_real;
        ss_res += diff * diff;

        double diff2 = actual_real - mean;
        ss_tot += diff2 * diff2;
    }
    if (ss_tot == 0.0) return 0.0;
    return 1.0 - (ss_res / ss_tot);
}

// ─── EARLY STOPPING ───────────────────────────────────────────────────────────
//
// Snapshots weights for ALL layers (hidden + output), not just one layer.

typedef struct {
    // Flat arrays: best_weights[l] holds the weight snapshot for layer l.
    // Only layers with prev_layer != NULL have weights; we allocate for all
    // but only use indices 1..num_layers-1.
    double** best_weights;       // [num_layers][weights per node * num_nodes]
    double*  best_biases;        // [num_layers * max_nodes]  (flattened per layer)
    int*     layer_num_nodes;    // needed to iterate during restore
    int*     layer_prev_nodes;   // number of weights per node in layer l

    double   best_loss;
    int      best_epoch;
    int      patience;
    int      patience_counter;
    int      num_layers;
} EarlyStopping;

EarlyStopping init_early_stopping(Layer* layers, int num_layers, int patience) {
    EarlyStopping es;
    es.num_layers        = num_layers;
    es.best_loss         = 1e18;
    es.best_epoch        = 0;
    es.patience          = patience;
    es.patience_counter  = 0;

    es.best_weights     = (double**)calloc(num_layers, sizeof(double*));
    es.best_biases      = NULL;   // allocated per-layer below
    es.layer_num_nodes  = (int*)malloc(num_layers * sizeof(int));
    es.layer_prev_nodes = (int*)malloc(num_layers * sizeof(int));

    // Count total bias slots needed
    int total_nodes = 0;
    for (int l = 0; l < num_layers; l++) {
        es.layer_num_nodes[l]  = layers[l].num_nodes;
        es.layer_prev_nodes[l] = (l == 0) ? 0 : layers[l - 1].num_nodes;
        total_nodes += layers[l].num_nodes;
    }
    es.best_biases = (double*)calloc(total_nodes, sizeof(double));

    // Allocate weight snapshot buffers for each non-input layer
    for (int l = 1; l < num_layers; l++) {
        int n_weights = layers[l].num_nodes * layers[l - 1].num_nodes;
        es.best_weights[l] = (double*)malloc(n_weights * sizeof(double));
    }
    return es;
}

void save_weights(EarlyStopping* es, Layer* layers) {
    int bias_offset = 0;
    for (int l = 0; l < es->num_layers; l++) {
        for (int i = 0; i < es->layer_num_nodes[l]; i++) {
            es->best_biases[bias_offset++] = layers[l].nodes[i].bias;
        }
        if (l == 0) continue;
        int prev_n = es->layer_prev_nodes[l];
        for (int i = 0; i < es->layer_num_nodes[l]; i++)
            for (int j = 0; j < prev_n; j++)
                es->best_weights[l][i * prev_n + j] = layers[l].nodes[i].weights[j];
    }
}

void restore_best_weights(EarlyStopping* es, Layer* layers) {
    int bias_offset = 0;
    for (int l = 0; l < es->num_layers; l++) {
        for (int i = 0; i < es->layer_num_nodes[l]; i++) {
            layers[l].nodes[i].bias = es->best_biases[bias_offset++];
        }
        if (l == 0) continue;
        int prev_n = es->layer_prev_nodes[l];
        for (int i = 0; i < es->layer_num_nodes[l]; i++)
            for (int j = 0; j < prev_n; j++)
                layers[l].nodes[i].weights[j] = es->best_weights[l][i * prev_n + j];
    }
}

// Returns 1 → stop, 0 → continue
int update_early_stopping(EarlyStopping* es, Layer* layers, double test_loss, int epoch) {
    if (test_loss < es->best_loss) {
        es->best_loss        = test_loss;
        es->best_epoch       = epoch;
        es->patience_counter = 0;
        save_weights(es, layers);
    } else {
        es->patience_counter++;
        if (es->patience_counter >= es->patience) return 1;
    }
    return 0;
}

void free_early_stopping(EarlyStopping* es) {
    for (int l = 1; l < es->num_layers; l++) free(es->best_weights[l]);
    free(es->best_weights);
    free(es->best_biases);
    free(es->layer_num_nodes);
    free(es->layer_prev_nodes);
}

// ─── MAIN ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {

    srand((unsigned int)time(NULL));

    // --- Filename ---
    char filename[256];
    if (argc >= 2) {
        strncpy(filename, argv[1], sizeof(filename) - 1);
        filename[sizeof(filename) - 1] = '\0';
    } else {
        printf("Enter CSV filename: ");
        scanf("%255s", filename);
    }

    // --- Load ---
    Dataset ds = load_csv(filename);
    if (ds.entries == 0 || ds.num_features == 0) {
        printf("Error: Failed to load dataset.\n");
        return 1;
    }

    // --- Shuffle ---
    shuffle_dataset(&ds);

    // --- 80/20 split ---
    TrainTestSplit tts = train_test_split(&ds, 0.80);

    // --- Scale features (fit on train only) ---
    ScaleParams* feat_scale = fit_scale(tts.train.features, tts.train.entries, ds.num_features);
    apply_scale(tts.train.features, tts.train.entries, ds.num_features, feat_scale);
    apply_scale(tts.test.features,  tts.test.entries,  ds.num_features, feat_scale);

    // --- Scale labels (fit on train only) ---
    ScaleParams label_scale = fit_label_scale(tts.train.labels, tts.train.entries);
    apply_label_scale(tts.train.labels, tts.train.entries, &label_scale);
    apply_label_scale(tts.test.labels,  tts.test.entries,  &label_scale);

    // --- Architecture ---
    // Adjust hidden layers / sizes here as needed.
    int layer_sizes[] = { ds.num_features, 16, 1 };
    int num_layers    = (int)(sizeof(layer_sizes) / sizeof(layer_sizes[0]));

    Layer* layers = build_network(layer_sizes, num_layers);

    // --- Hyperparameters ---
    double learning_rate = 0.01;
    int    epochs        = 10000;
    int    patience      = 1500;

    // --- Early stopping ---
    EarlyStopping es = init_early_stopping(layers, num_layers, patience);

    printf("\nTraining: %d layers, lr=%.4f, patience=%d\n\n", num_layers, learning_rate, patience);

    // --- Training loop (stochastic / online gradient descent) ---
    int stopped_epoch = epochs;

    for (int epoch = 0; epoch < epochs; epoch++) {

        for (int i = 0; i < tts.train.entries; i++) {
            set_input(&layers[0], tts.train.features[i], ds.num_features);

            double pred = forward_pass(&layers[0]);
            double err  = pred - tts.train.labels[i];

            double* output_error = (double*)malloc(sizeof(double));
            output_error[0] = 2.0 * err;

            backward_pass(&layers[num_layers - 1], output_error);

            update_weights(&layers[0], learning_rate);
        }

        double train_loss = evaluate(layers, ds.num_features,
                                     &tts.train, &label_scale, 0);
        double test_loss  = evaluate(layers, ds.num_features,
                                     &tts.test,  &label_scale, 0);

        if (epoch % 500 == 0) {
            printf("Epoch %5d | Train MSE: %.6f | Test MSE: %.6f | Patience: %d/%d\n",
                   epoch, train_loss, test_loss, es.patience_counter, patience);
        }

        if (update_early_stopping(&es, layers, test_loss, epoch)) {
            stopped_epoch = epoch;
            printf("\nEarly stopping at epoch %d\n", epoch);
            printf("Best test MSE: %.6f at epoch %d\n", es.best_loss, es.best_epoch);
            break;
        }
    }

    if (stopped_epoch == epochs)
        printf("\nTraining completed all %d epochs.\n", epochs);

    // --- Restore best weights ---
    restore_best_weights(&es, layers);
    printf("Weights restored to epoch %d\n", es.best_epoch);

    // --- Final evaluation ---
    printf("\n--- Train Set ---\n");
    double final_train = evaluate(layers, ds.num_features,
                                  &tts.train, &label_scale, 1);
    printf("Train MSE (scaled): %.6f\n", final_train);

    printf("\n--- Test Set ---\n");
    double final_test = evaluate(layers, ds.num_features,
                                 &tts.test, &label_scale, 1);
    printf("Test  MSE (scaled): %.6f\n", final_test);

    // --- Cleanup ---
    double re_train = r2_score(layers, ds.num_features, &tts.train, &label_scale);
    double re_test  = r2_score(layers, ds.num_features, &tts.test,  &label_scale);
    printf("\nTrain R^2: %.6f\n", re_train);
    printf("Test  R^2: %.6f\n", re_test);

    free_early_stopping(&es);
    free_network(layers, num_layers);
    free(feat_scale);
    free_dataset(&ds);

    return 0;
}