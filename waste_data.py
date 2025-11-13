import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers, mixed_precision # pyright: ignore[reportMissingImports]
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # pyright: ignore[reportMissingImports]

# ===============================================================
# ⚙️ INTEL + MIXED PRECISION OPTIMIZATIONS
# ===============================================================
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"

try:
    import intel_extension_for_tensorflow as itex  # type: ignore
    print("✅ Intel Extension for TensorFlow (ITEX) loaded successfully.")
except ImportError:
    print("⚠️ ITEX not installed. Using default TensorFlow optimizations.")

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)
mixed_precision.set_global_policy("mixed_float16")

print(f"TensorFlow version: {tf.__version__}")
print("Devices available:", tf.config.list_physical_devices())

# ===============================================================
# 🧩 DATA LOADING
# ===============================================================
def load_data(data_dir, img_size=(224, 224), batch_size=32, val_split=0.3, seed=123):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        validation_split=val_split,
        subset="training",
    )
    val_test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        validation_split=val_split,
        subset="validation",
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"\nDetected Classes: {class_names} ({num_classes} total)")

    val_test_card = tf.data.experimental.cardinality(val_test_ds).numpy()
    val_batches = val_test_card // 2
    val_ds = val_test_ds.take(val_batches)
    test_ds = val_test_ds.skip(val_batches)

    def prep(ds):
        ds = ds.map(
            lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), tf.one_hot(y, num_classes)),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        return ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return prep(train_ds), prep(val_ds), prep(test_ds), class_names, num_classes, train_ds

# ===============================================================
# 💡 CLASS WEIGHTS
# ===============================================================
def compute_weights(train_ds, num_classes):
    y_labels = np.array([labels.numpy() for _, labels in train_ds.unbatch()])
    class_weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=y_labels)
    weights = dict(enumerate(class_weights))
    print("\n⚖️ Computed Class Weights:")
    for i, w in weights.items():
        print(f" Class {i}: {w:.3f}")
    return weights

# ===============================================================
# 🔥 FOCAL LOSS
# ===============================================================
@tf.keras.utils.register_keras_serializable(package="custom_losses")
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, from_logits=False, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        fl = self.alpha * tf.pow(1 - y_pred, self.gamma) * ce
        return tf.reduce_sum(fl, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha, "from_logits": self.from_logits})
        return config

# ===============================================================
# 🧱 ADAPTIVE AUGMENT
# ===============================================================
@tf.keras.utils.register_keras_serializable(package="custom_layers")
class AdaptiveAugment(layers.Layer):
    def __init__(self, initial_intensity=0.1, **kwargs):
        super().__init__(**kwargs)
        self.intensity = tf.Variable(initial_intensity, trainable=False, dtype=tf.float32)
        self.flip = layers.RandomFlip("horizontal_and_vertical")
        self.rotation = layers.RandomRotation(0.1)
        self.zoom = layers.RandomZoom(0.1)
        self.contrast = layers.RandomContrast(0.1)

    def call(self, x, training=None):
        if training:
            x = self.flip(x)
            aug_x = self.rotation(x)
            aug_x = self.zoom(aug_x)
            aug_x = self.contrast(aug_x)
            intensity = tf.cast(self.intensity, x.dtype)
            x = (1.0 - intensity) * x + intensity * aug_x
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"initial_intensity": float(tf.keras.backend.get_value(self.intensity))})
        return config

# ===============================================================
# 🏗️ MODEL BUILDING
# ===============================================================
def build_model(img_size, num_classes):
    input_shape = (*img_size, 3)
    adaptive_aug = AdaptiveAugment(initial_intensity=0.1)
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = adaptive_aug(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", kernel_regularizer=regularizers.l2(1e-4), dtype="float32")(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=FocalLoss(), metrics=["accuracy"])
    print("\n🧠 Model Summary:")
    model.summary()
    return model, base_model, adaptive_aug

# ===============================================================
# 🎚️ ADAPTIVE AUGMENT CALLBACK
# ===============================================================
class AdaptiveAugmentCallback(callbacks.Callback):
    def __init__(self, adaptive_layer, start=0.1, max_val=0.3, step=0.05, update_every=5):
        super().__init__()
        self.adaptive_layer = adaptive_layer
        self.start = start
        self.max_val = max_val
        self.step = step
        self.update_every = update_every

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.update_every == 0:
            new_val = min(self.adaptive_layer.intensity.numpy() + self.step, self.max_val)
            self.adaptive_layer.intensity.assign(new_val)
            print(f"\n🌱 Adaptive Augmentation intensity increased to: {new_val:.2f}")

# ===============================================================
# ⚡ FINE-TUNE AFTER 2 LR DROPS + VAL_ACC STALL
# ===============================================================
# ===============================================================
# ⚡ FINE-TUNE AFTER 2 LR DROPS + VAL_ACC STALL
# ===============================================================
class FineTuneAfterLRDropStall(callbacks.Callback):
    def __init__(self, model, base_model, train_ds, val_ds, adaptive_aug, class_weights,
                 fine_tune_at=100, fine_tune_epochs=20, min_epochs=10):
        super().__init__()
        self.lr_drop_count = 0
        self.model_ref = model
        self.base_model = base_model
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.adaptive_aug = adaptive_aug
        self.class_weights = class_weights
        self.fine_tune_at = fine_tune_at
        self.fine_tune_epochs = fine_tune_epochs
        self.min_epochs = min_epochs
        self.fine_tuned = False
        self.prev_lr = None
        self.prev_val_acc = None
        self.last_val_acc = 0.0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_acc = logs.get("val_accuracy", 0)
        optimizer = self.model_ref.optimizer

        # Handle mixed precision optimizers
        if hasattr(optimizer, "_optimizer"):
            lr = float(tf.keras.backend.get_value(optimizer._optimizer.learning_rate))
        else:
            lr = float(tf.keras.backend.get_value(optimizer.learning_rate))

        # Detect LR reduction
        if self.prev_lr is not None and lr < self.prev_lr:
            self.lr_drop_count += 1
            print(f"\n⚡ Learning rate reduced! Count: {self.lr_drop_count}/2")
            self.prev_val_acc = self.last_val_acc

        # Detect validation accuracy stall
        if val_acc <= self.last_val_acc:
            print(f"📉 Validation accuracy stalled or decreased (prev: {self.last_val_acc:.4f}, current: {val_acc:.4f})")

        # Trigger fine-tuning
        if (not self.fine_tuned and
            self.lr_drop_count >= 2 and
            (epoch + 1) >= self.min_epochs and
            self.prev_val_acc is not None and
            val_acc <= self.prev_val_acc):
            print(f"\n🚀 Conditions met — (2 LR drops + {self.min_epochs}+ epochs + val_acc stalled) — initiating fine-tuning!")
            fine_tune_model(
                self.model_ref,
                self.base_model,
                self.train_ds,
                self.val_ds,
                self.adaptive_aug,
                self.class_weights,
                fine_tune_at=self.fine_tune_at,
                fine_tune_epochs=self.fine_tune_epochs
            )
            self.fine_tuned = True

        # Update trackers
        self.prev_lr = lr
        self.last_val_acc = val_acc


# ===============================================================
# 🏋️ TRAINING
# ===============================================================
def train_model(model, base_model, train_ds, val_ds, epochs, adaptive_aug, class_weights):
    os.makedirs("checkpoints", exist_ok=True)
    callbacks_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint("checkpoints/best_model.keras", monitor="val_accuracy", save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1, min_lr=1e-7),
        AdaptiveAugmentCallback(adaptive_aug),
        FineTuneAfterLRDropStall(model, base_model, train_ds, val_ds, adaptive_aug, class_weights,
                                 fine_tune_at=100, fine_tune_epochs=20, min_epochs=10)
    ]
    print("\n🚀 Starting Training with Focal Loss + Class Weights...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks_list, class_weight=class_weights, verbose=1)
    return history

# ===============================================================
# 🔧 FINE-TUNING FUNCTION
# ===============================================================
def fine_tune_model(model, base_model, train_ds, val_ds, adaptive_aug, class_weights, fine_tune_at=100, fine_tune_epochs=20):
    print("\n🎯 Starting Fine-Tuning Phase...")
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    print(f"🔓 Unfroze top {len(base_model.layers) - fine_tune_at} layers of MobileNetV2")

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss=FocalLoss(), metrics=["accuracy"])
    callbacks_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint("checkpoints/best_finetuned_model.keras", monitor="val_accuracy", save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-7),
        AdaptiveAugmentCallback(adaptive_aug),
    ]
    history_fine = model.fit(train_ds, validation_data=val_ds, epochs=fine_tune_epochs, callbacks=callbacks_list, class_weight=class_weights, verbose=1)
    model.save("waste_classification_model_finetuned.keras")
    print("\n✅ Fine-tuning complete! Model saved as waste_classification_model_finetuned.keras")
    return history_fine

# ===============================================================
# 🧾 EVALUATION
# ===============================================================
def evaluate_model(model, test_ds, class_names):
    print("\n🔍 Evaluating Model...")
    loss, acc = model.evaluate(test_ds)
    print(f"✅ Test Accuracy: {acc*100:.2f}%")

    y_true, y_pred = [], []
    for imgs, labels in test_ds:
        preds = model.predict(imgs, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.close()
    print("📸 Confusion matrix saved as confusion_matrix.png")

# ===============================================================
# 🚦 MAIN EXECUTION
# ===============================================================
def main():
    print("="*65)
    print("♻️ WASTE CLASSIFICATION SYSTEM (Focal + Weighted + Adaptive)")
    print("="*65)

    config = {
        "data_dir": Path(r"C:\Users\DELL\OneDrive\Desktop\DESKTOP STUFF\AI_Waste_Management\Waste-Dataset\data"),
        "img_size": (224, 224),
        "batch_size": 32,
        "epochs": 50,
    }

    if not config["data_dir"].exists():
        print(f"❌ Data directory {config['data_dir']} not found.")
        return

    # Load datasets
    train_ds, val_ds, test_ds, class_names, num_classes, raw_train_ds = load_data(config["data_dir"], config["img_size"], config["batch_size"])
    class_weights = compute_weights(raw_train_ds, num_classes)

    # Resume from checkpoint if exists
    checkpoint_path = Path("checkpoints/best_model.keras")
    if checkpoint_path.exists():
        print("\n♻️ Found previous checkpoint — resuming training...")
        model = tf.keras.models.load_model(checkpoint_path, custom_objects={"FocalLoss": FocalLoss, "AdaptiveAugment": AdaptiveAugment})
        base_model = model.get_layer(index=2) if len(model.layers) > 2 else None
        adaptive_aug = next((l for l in model.layers if isinstance(l, AdaptiveAugment)), AdaptiveAugment(initial_intensity=0.1))
    else:
        print("\n🚀 No checkpoint found — building new model...")
        model, base_model, adaptive_aug = build_model(config["img_size"], num_classes)

    # Train
    history = train_model(model, base_model, train_ds, val_ds, config["epochs"], adaptive_aug, class_weights)

    # Evaluate & save final model
    evaluate_model(model, test_ds, class_names)
    model.save("waste_classification_model_final.keras")
    print("\n✅ Training complete! Model saved as waste_classification_model_final.keras")

if __name__ == "__main__":
    main()
