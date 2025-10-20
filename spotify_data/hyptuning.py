# Hyperparameter tuning script for autoencoder
# ChatGPT and Caleb Bessit
#  |→ Boilerplate and cleaning up done by ChatGPT, logic and decisions done by Caleb Bessit
# 20 October 2025

import os, itertools, csv, time
import numpy as np
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from utils import load_data, calculate_weights, classification_metrics
from setup import VERSION

# ================================================================
# 1. Load and scale data
# ================================================================
NUM_CLASSES = 3
if VERSION == "popularity":
    X_train, y_train = load_data("train", NUM_CLASSES)
    X_test, y_test   = load_data("test", NUM_CLASSES)
elif VERSION == "genre":
    X_train, y_train = load_data("train_genre", NUM_CLASSES)
    X_test, y_test   = load_data("test_genre", NUM_CLASSES)


print(f"Loaded data for {VERSION} classification task.")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
input_dim = X_train.shape[1]

# ================================================================
# 2. Define configuration grid
# ================================================================
layer_counts = [1, 2, 3]
hidden_widths = [32, 64, 128]
latent_dims = [4, 8, 16, 32]
activations = ['relu', 'tanh']

configs = []
for n_layers, width, latent, act in itertools.product(layer_counts, hidden_widths, latent_dims, activations):
    hidden_sizes = [width // (2 ** i) for i in range(n_layers)]  #Have hidden layer sizes that try every combination for layer sizes, in decreasing powers of two in each subsequent layer
    configs.append({
        "hidden_sizes": hidden_sizes,
        "latent_dim": latent,
        "activation": act
    })

print(f"Generated {len(configs)} configurations to evaluate.")

# ================================================================
# 3. Build autoencoder function
# ================================================================
def build_autoencoder(input_dim, hidden_sizes, latent_dim, activation):
    encoder = models.Sequential([layers.Input(shape=(input_dim,))])
    for h in hidden_sizes:
        encoder.add(layers.Dense(h, activation=activation))
    encoder.add(layers.Dense(latent_dim, activation=activation))

    decoder = models.Sequential([layers.Input(shape=(latent_dim,))])
    for h in reversed(hidden_sizes):
        decoder.add(layers.Dense(h, activation=activation))
    decoder.add(layers.Dense(input_dim))

    autoencoder = models.Sequential([encoder, decoder])
    autoencoder.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='mse'
    )
    return autoencoder, encoder

# ================================================================
# 4. Results logging setup
# ================================================================

# TODO: Please double-check path
results_path = "/scratch/dnxmat002/ai_results"
os.makedirs(results_path, exist_ok=True)
results_file = os.path.join(results_path, f"autoencoder_tuning_{VERSION}.csv")
if not os.path.exists(results_file):
    with open(results_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "hidden_sizes", "latent_dim", "activation",
            "val_loss", "test_accuracy", "test_f1", "test_bal_acc", "train_time_sec"
        ])

# ================================================================
# 5. Loop over configurations
# ================================================================
for i, cfg in enumerate(configs, start=1):
    print(f"\n[{i}/{len(configs)}] Training autoencoder: {cfg}")

    autoencoder, encoder = build_autoencoder(
        input_dim, cfg["hidden_sizes"], cfg["latent_dim"], cfg["activation"]
    )

    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    start = time.time()
    hist = autoencoder.fit(
        X_train_scaled, X_train_scaled,
        epochs=30, batch_size=64,
        validation_split=0.125, verbose=0,
        callbacks=[early_stop]
    )
    train_time = time.time() - start
    val_loss = min(hist.history["val_loss"])

    # Encode features
    X_train_encoded = encoder.predict(X_train_scaled, verbose=0)
    X_test_encoded = encoder.predict(X_test_scaled, verbose=0)

    # ============================================================
    # 6. Downstream MLP classifier
    # ============================================================
    class_head = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        random_state=0,
        max_iter=500
    )
    class_weights, sample_weights = calculate_weights(y_train)

    clf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", class_head)
    ])
    clf_pipeline.fit(X_train_encoded, y_train, mlp__sample_weight=sample_weights)

    y_pred = clf_pipeline.predict(X_test_encoded)
    class_results = classification_metrics(y_test, y_pred)

    acc, f1, bal_acc = class_results["accuracy"], class_results["f1"], class_results["balanced_accuracy"]

    # ============================================================
    # 7. Save results
    # ============================================================
    with open(results_file, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            cfg["hidden_sizes"], cfg["latent_dim"], cfg["activation"],
            round(val_loss, 6), round(acc, 4), round(f1, 4), round(bal_acc,4),round(train_time, 2)
        ])

    print(f" -> val_loss={val_loss:.4f}, acc={acc:.3f}, f1={f1:.3f}, bal_acc={bal_acc:.3f}, time={train_time/60:.1f} min")

print(f"\nAll configurations evaluated. Results saved to {results_file}.")
