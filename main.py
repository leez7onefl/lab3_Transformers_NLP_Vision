import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers

# Enable memory growth for GPU if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# IMDB dataset preparation
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)

word_index = tf.keras.datasets.imdb.get_word_index()
index_to_word = {index + 3: word for word, index in word_index.items()}
index_to_word[0] = "[PAD]"
index_to_word[1] = "[START]"
index_to_word[2] = "[UNK]"
index_to_word[3] = "[UNUSED]"

x_train_text = [" ".join(index_to_word.get(i, "?") for i in review) for review in x_train]
x_test_text = [" ".join(index_to_word.get(i, "?") for i in review) for review in x_test]

# DistilBert tokenization
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
train_encodings = tokenizer(x_train_text, padding=True, truncation=True, max_length=128, return_tensors="tf")
test_encodings = tokenizer(x_test_text, padding=True, truncation=True, max_length=128, return_tensors="tf")

# Create datasets
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), y_train)).shuffle(1000).batch(8)
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), y_test)).batch(8)

with tf.device("/CPU:0"):
    model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2, from_pt=True)

optimizer = Adam(learning_rate=5e-5)
loss_fn = SparseCategoricalCrossentropy(from_logits=True)

# Training loop
epochs = 1

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    for step, (batch_inputs, batch_labels) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            outputs = model(batch_inputs, training=True)
            loss = loss_fn(batch_labels, outputs.logits)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss.numpy():.4f}")
    
    total_correct = 0
    total_samples = 0
    for batch_inputs, batch_labels in test_dataset:
        outputs = model(batch_inputs, training=False)
        predictions = tf.argmax(outputs.logits, axis=-1)
        total_correct += tf.reduce_sum(tf.cast(predictions == batch_labels, tf.int32)).numpy()
        total_samples += batch_labels.shape[0]
    accuracy = total_correct / total_samples
    print(f"Validation Accuracy: {accuracy:.4f}")

# Save model
# model.save_pretrained("distilbert-imdb")
# tokenizer.save_pretrained("distilbert-imdb")

# Load and evaluate
model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-imdb")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-imdb")

test_encodings = tokenizer(x_test_text, padding=True, truncation=True, max_length=128, return_tensors="tf")
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), y_test)).batch(8)

y_pred = []
total_correct = 0
total_samples = 0

for batch_inputs, batch_labels in tqdm(test_dataset, desc="Evaluating"):
    outputs = model(batch_inputs, training=False)
    predictions = tf.argmax(outputs.logits, axis=-1)
    y_pred.extend(predictions.numpy().tolist())
    total_correct += tf.reduce_sum(tf.cast(predictions == batch_labels, tf.int32)).numpy()
    total_samples += batch_labels.shape[0]

accuracy = total_correct / total_samples
print(f"Accuracy: {accuracy:.4f}")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# CIFAR-10 Vision Transformer implementation
num_classes = 10
input_shape = (32, 32, 3)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_train, y_test = y_train.flatten(), y_test.flatten()

image_size = 72
data_augmentation = keras.Sequential([
    layers.Resizing(image_size, image_size),
    layers.Normalization(),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.02),
    layers.RandomZoom(0.2, 0.2)
], name="data_augmentation")

data_augmentation.layers[1].adapt(x_train)

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        return {"patch_size": self.patch_size}

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch_vectors):
        projected = self.projection(patch_vectors)
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        positions = tf.expand_dims(positions, axis=0)
        return projected + self.position_embedding(positions)

    def get_config(self):
        return {"num_patches": self.num_patches, "projection_dim": self.position_embedding.output_dim}

patch_size = 6
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [projection_dim * 2, projection_dim]
transformer_layers = 8
mlp_head_units = [2048, 1024]

def create_vit_classifier():
    inputs = keras.Input(shape=input_shape)
    augmented = data_augmentation(inputs)
    patches = Patches(patch_size)(augmented)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    
    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        mlp_output = layers.Dense(transformer_units[0], activation=keras.activations.gelu)(x3)
        mlp_output = layers.Dropout(0.1)(mlp_output)
        mlp_output = layers.Dense(transformer_units[1], activation=keras.activations.gelu)(mlp_output)
        mlp_output = layers.Dropout(0.1)(mlp_output)
        encoded_patches = layers.Add()([mlp_output, x2])
    
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    
    features = representation
    for units in mlp_head_units:
        features = layers.Dense(units, activation=keras.activations.gelu)(features)
        features = layers.Dropout(0.5)(features)
    logits = layers.Dense(num_classes)(features)
    
    return keras.Model(inputs=inputs, outputs=logits, name="ViT_CIFAR10")

vit_model = create_vit_classifier()

optimizer = keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001)
vit_model.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)

epochs = 50
batch_size = 32
vit_model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1
)

test_loss, test_accuracy = vit_model.evaluate(x_test, y_test, batch_size=batch_size)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

vit_model.save("vit_cifar10.keras")

y_preds = vit_model.predict(x_test)
y_pred = np.argmax(y_preds, axis=1)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="hot", cbar=False, xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()