# Hyperparameter tuning across HPC nodes
# ChatGPT + Caleb Bessit — 20 Oct 2025

import os, itertools, csv, time, argparse, fcntl
import numpy as np
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from utils_exp import load_data, calculate_weights, classification_metrics
from setup import VERSION


# ================================================================
# -1. Define feature subsets
# ================================================================


metadata   = ['year', 'duration_ms']
physical   = ['loudness','tempo','instrumentalness','speechiness','liveness']
perceptual = ['danceability','energy','valence','acousticness']
structural = ['key','mode','time_signature']

# if predicting genre, add popularity to metadata, and if predicting popularity, add genre to metadata

# ================================================================
# 0. Parse arguments
# ================================================================

parser = argparse.ArgumentParser(description="Distributed autoencoder tuning.")
parser.add_argument("--start", type=int, default=0, help="Start index of configs to process.")
parser.add_argument("--end", type=int, default=None, help="End index (exclusive) of configs to process.")

# TODO: CHANGE PATH IF NECESSARY!
parser.add_argument("--results_dir", type=str, default="/scratch/dnxmat002/ai_results", help="Path to results directory.")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=64)
args = parser.parse_args()

# ================================================================
# 1. Load and scale data
# ================================================================


configs = [
    {"target":"popularity",
     "non-target":"genre",
     "num_classes":3},
    {"target":"popularity",
     "non-target":"genre",
     "num_classes":5},
    {"target":"genre",
     "non-target":"popularity",
     "num_classes":5}
]

subsets = [ physical, perceptual, structural]
subset_names = ["physical","perceptual","structural","metadata"]

for idx in range(len(configs)):
    config = configs[idx]
    target, num_classes = config["target"], config["num_classes"]

    subsets.append(metadata+[config["non-target"]])
    for subset_idx in subsets:
        subset, subset_name = subsets[subset_idx], subset_name[subset_idx]
        

        X_train, y_train = load_data("train",num_classes,subset,target) if target=="popularity" else load_data("train_genre",num_classes,subset,target)
        X_test, y_test   = load_data("test",num_classes,subset,target) if target=="popularity" else load_data("test_genre",num_classes,subset,target)

        print(f"Loaded data for {num_classes}-class classification task for {target} with config {idx}.")
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
            hidden_sizes = [width // (2 ** i) for i in range(n_layers)] #Have hidden layer sizes that try every combination for layer sizes, in decreasing powers of two in each subsequent layer
            configs.append({
                "hidden_sizes": hidden_sizes,
                "latent_dim": latent,
                "activation": act
            })

        n_total = len(configs)
        start = args.start
        end = args.end if args.end is not None else n_total
        configs = configs[start:end]
        print(f"Running configs {start}:{end} of total {n_total}")

        # ================================================================
        # 3. Build autoencoder
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
        # 4. Results setup with file locking
        # ================================================================
        os.makedirs(args.results_dir, exist_ok=True)
        results_file = os.path.join(args.results_dir, f"autoencoder_tuning_{target}_{num_classes}_classes_using_{subset_name}.csv")
        if not os.path.exists(results_file):
            with open(results_file, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "hidden_sizes", "latent_dim", "activation",
                    "val_loss", "test_accuracy", "test_f1", "test_bal_acc", "train_time_sec"
                ])

        def safe_write_row(path, row):
            """Write a row to CSV with file lock to prevent collisions."""
            with open(path, "a", newline='\n') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                writer = csv.writer(f)
                writer.writerow(row)
                fcntl.flock(f, fcntl.LOCK_UN)

        # ================================================================
        # 5. Loop over assigned configurations
        # ================================================================
        for i, cfg in enumerate(configs, start=start+1):
            print(f"\n[{i}/{n_total}] Training autoencoder: {cfg}")

            autoencoder, encoder = build_autoencoder(
                input_dim, cfg["hidden_sizes"], cfg["latent_dim"], cfg["activation"]
            )

            early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            start_time = time.time()
            hist = autoencoder.fit(
                X_train_scaled, X_train_scaled,
                epochs=args.epochs, batch_size=args.batch_size,
                validation_split=0.125, verbose=0,
                callbacks=[early_stop]
            )
            train_time = time.time() - start_time
            val_loss = float(np.min(hist.history["val_loss"]))

            # Encode features
            X_train_encoded = encoder.predict(X_train_scaled, verbose=0)
            X_test_encoded = encoder.predict(X_test_scaled, verbose=0)

            # Classification head
            class_head = MLPClassifier(hidden_layer_sizes=(64, 32), random_state=0, max_iter=500)
            class_weights, sample_weights = calculate_weights(y_train)
            clf_pipeline = Pipeline([("scaler", StandardScaler()), ("mlp", class_head)])
            clf_pipeline.fit(X_train_encoded, y_train, mlp__sample_weight=sample_weights)

            y_pred = clf_pipeline.predict(X_test_encoded)
            class_results = classification_metrics(y_test, y_pred)
            acc = class_results["accuracy"]
            f1 = class_results["f1"]
            bal_acc = class_results["balanced_accuracy"]

            row = [
                cfg["hidden_sizes"], cfg["latent_dim"], cfg["activation"],
                round(val_loss, 6), round(acc, 4), round(f1, 4),
                round(bal_acc, 4), round(train_time, 2)
            ]
            safe_write_row(results_file, row)

            print(f" -> val_loss={val_loss:.4f}, acc={acc:.3f}, f1={f1:.3f}, bal_acc={bal_acc:.3f}, time={train_time/60:.1f} min")

        print(f"\nFinished configs {start}:{end}. Results appended to {results_file}")
