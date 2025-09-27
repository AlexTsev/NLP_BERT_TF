import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# Load IMDB dataset
ds_train, ds_test = tfds.load('imdb_reviews', split=['train', 'test'], as_supervised=True)

train_texts = []
train_labels = []
for text, label in tfds.as_numpy(ds_train):
    train_texts.append(text.decode('utf-8'))
    train_labels.append(label)

test_texts = []
test_labels = []
for text, label in tfds.as_numpy(ds_test):
    test_texts.append(text.decode('utf-8'))
    test_labels.append(label)

# Split train into train/validation
from sklearn.model_selection import train_test_split

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.2, stratify=train_labels, random_state=42
)

# -------------------------------------------------------------------------
# Hugging Face tokenizer (BERT uncased)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def encode(texts):
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="np"
    )
    return enc["input_ids"], enc["attention_mask"]

train_input_ids, train_masks = encode(train_texts)
val_input_ids, val_masks     = encode(val_texts)

AUTOTUNE = tf.data.AUTOTUNE
batch_size = 8

train_ds = tf.data.Dataset.from_tensor_slices(
    ((train_input_ids, train_masks), np.array(train_labels))
).shuffle(len(train_labels)).batch(batch_size).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices(
    ((val_input_ids, val_masks), np.array(val_labels))
).batch(batch_size).prefetch(AUTOTUNE)

# -------------------------------------------------------------------------
# BERT encoder from TF Hub
bert_encoder = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3",
    trainable=True,
    name="bert_encoder"
)

# Build model
input_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name="input_ids")
input_mask = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name="input_mask")

outputs = bert_encoder({
    "input_word_ids": input_ids,
    "input_mask": input_mask,
    "input_type_ids": tf.zeros_like(input_ids)
})

net = outputs["pooled_output"]
net = tf.keras.layers.Dropout(0.1)(net)
net = tf.keras.layers.Dense(1, activation="sigmoid")(net)  # binary classification

model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=net)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"]
)

# -------------------------------------------------------------------------
# Train
history = model.fit(train_ds, validation_data=val_ds, epochs=3)

# -------------------------------------------------------------------------
# Predict helper
def predict(text):
    ids, mask = encode([text])
    prob = model.predict([ids, mask])[0][0]
    return "Positive" if prob >= 0.5 else "Negative"

print(predict("This movie was amazing!"))
print(predict("I did not like this movie at all."))

# -------------------------------------------------------------------------
# Plots
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("IMDB_loss_plot.png")
plt.show()

plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("IMDB_accuracy_plot.png")
plt.show()

# -------------------------------------------------------------------------
# Predict emotions on validation set
print("\n--- Validation set predictions ---\n")

for i, ((ids, masks), label) in enumerate(val_ds.take(10)):  # take first 10 samples
    probs = model.predict([ids, masks])
    pred_labels = ["Negative" if p < 0.5 else "Positive" for p in probs.flatten()]
    true_labels = ["Negative" if l == 0 else "Positive" for l in label.numpy()]

    for t, p in zip(true_labels, pred_labels):
        print(f"True: {t}  -->  Predicted: {p}")
    print()


# -------------------------------------------------------------------------
# Create a visualization of validation texts with predicted sentiment
texts_to_show = val_texts[:10]      # first 10 texts
labels_true = val_labels[:10]

# Predict sentiments
pred_labels = []
for text in texts_to_show:
    ids, mask = encode([text])
    probs = model.predict([ids, mask])
    pred_labels.append("Positive" if probs[0][0] >= 0.5 else "Negative")

# Create figure
fig, ax = plt.subplots(figsize=(12, len(texts_to_show) * 1.2))
ax.axis('off')  # hide axes

# Display texts with predicted sentiment
for i, (text, pred) in enumerate(zip(texts_to_show, pred_labels)):
    color = 'green' if pred == 'Positive' else 'red'
    ax.text(0, len(texts_to_show) - i - 0.5, f"{text} --> {pred}", fontsize=10, color=color, wrap=True)

plt.tight_layout()
plt.savefig("IMDB_validation_texts_predictions.png", dpi=300)
plt.show()