import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# Load CSV
df = pd.read_csv("../data_augmentation/dataset_generated.csv")
texts  = df["text"].astype(str).tolist()
labels = df["label"].astype(int).tolist()

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=42
)

# -------------------------------------------------------------------------
# Hugging Face tokenizer (same vocab as BERT uncased)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def encode(texts):
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="np"
    )
    return enc["input_ids"], enc["attention_mask"]

train_input_ids, train_masks = encode(train_texts)
val_input_ids,   val_masks   = encode(val_texts)

AUTOTUNE = tf.data.AUTOTUNE
batch_size = 8

train_ds = tf.data.Dataset.from_tensor_slices(
    ((train_input_ids, train_masks), np.array(train_labels))
).shuffle(len(train_labels)).batch(batch_size).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices(
    ((val_input_ids, val_masks), np.array(val_labels))
).batch(batch_size).prefetch(AUTOTUNE)

# -------------------------------------------------------------------------
# BERT encoder from TF-Hub (no preprocess layer needed)
bert_encoder = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3",
    trainable=True,
    name="bert_encoder"
)

# Build model
input_ids = tf.keras.layers.Input(shape=(64,), dtype=tf.int32, name="input_ids")
input_mask = tf.keras.layers.Input(shape=(64,), dtype=tf.int32, name="input_mask")

outputs = bert_encoder({"input_word_ids": input_ids,
                        "input_mask": input_mask,
                        "input_type_ids": tf.zeros_like(input_ids)})

net = outputs["pooled_output"]
net = tf.keras.layers.Dropout(0.1)(net)
net = tf.keras.layers.Dense(3, activation="softmax")(net)

model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=net)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

# -------------------------------------------------------------------------
# Train
history = model.fit(train_ds, validation_data=val_ds, epochs=3)

# -------------------------------------------------------------------------
# Predict helper
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

def predict(text):
    ids, mask = encode([text])
    probs = model.predict([ids, mask])
    return label_map[int(tf.argmax(probs, axis=1).numpy()[0])]

print(predict("Αυτό είναι υπέροχο!"))

# -------------------------------------------------------------------------
# Plots
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend(); plt.show()

plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.legend(); plt.show()


# -------------------------------------------------------------------------
# Evaluate on validation (which is also our test set)
test_loss, test_acc = model.evaluate(val_ds)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# -------------------------------------------------------------------------
# Make predictions on some examples
examples = [
    "This movie was fantastic!",
    "I did not like the film.",
    "It was okay, nothing special."
]

for text in examples:
    ids, mask = encode([text])
    probs = model.predict([ids, mask])
    pred_label = label_map[int(tf.argmax(probs, axis=1).numpy()[0])]
    print(f"Text: {text}\nPredicted sentiment: {pred_label}\n")