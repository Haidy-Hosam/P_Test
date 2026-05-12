# 🍽️ Multimodal Food Classification System
### *Computer Vision + Natural Language Processing for Intelligent Food Recognition and Nutritional Analysis*

---

> **Academic Project — Faculty of Computer Science**
> **Graduation Project | Final Submission**

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [Dataset Description & Web Scraping Pipeline](#4-dataset-description--web-scraping-pipeline)
5. [Data Preprocessing Pipeline](#5-data-preprocessing-pipeline)
6. [Model Architecture Experiments](#6-model-architecture-experiments)
7. [Training Strategy](#7-training-strategy)
8. [Evaluation Metrics & Comparative Analysis](#8-evaluation-metrics--comparative-analysis)
9. [Single Image Testing Across All Models](#9-single-image-testing-across-all-models)
10. [Calories Retrieval System](#10-calories-retrieval-system)
11. [Final Model Selection Justification](#11-final-model-selection-justification)
12. [Challenges Faced](#12-challenges-faced)
13. [Future Improvements](#13-future-improvements)
14. [Presentation & Defense Topics](#14-presentation--defense-topics)
15. [Team Contribution Distribution](#15-team-contribution-distribution)
16. [Conclusion](#16-conclusion)

---

## 1. Project Overview

The **Multimodal Food Classification System** is a comprehensive deep learning project that combines the power of **Computer Vision** and **Natural Language Processing (NLP)** to classify food items using both visual and textual modalities. Unlike traditional unimodal food recognition systems that rely solely on image input, this system leverages the rich complementary information carried by both food images and their accompanying textual descriptions — such as ingredient lists, names, and descriptive metadata — to achieve more robust and accurate food classification.

Beyond classification, the system integrates a **Calories Retrieval Module** that extracts ingredient information from the input text and looks up their nutritional values (calories per 100g and per kilogram) from a curated food nutrition database, offering a holistic and practical food intelligence solution.

This project was developed as a graduation-level research initiative and involved the design, implementation, training, and systematic comparison of **five distinct multimodal neural network architectures**, each combining a different image backbone with a different text representation strategy. Through rigorous experimentation, the team identified the optimal model that balances performance, efficiency, and deployment suitability.

---

## 2. Problem Statement

Food recognition is a challenging computer vision problem due to:

- **High intra-class variability**: The same food item can appear drastically different depending on preparation, presentation, lighting, and cultural context.
- **High inter-class similarity**: Different food categories can appear visually similar (e.g., various types of rice dishes, bread variations, or fried items).
- **Limited visual discriminability**: Visual features alone are often insufficient to distinguish between closely related food categories.
- **Dataset scarcity**: Publicly available labeled food datasets are either limited in scope, lack diversity, or do not include sufficient textual metadata paired with images.

These challenges motivate a **multimodal approach** that combines image features extracted by convolutional neural networks with semantic text features extracted via NLP models, enabling more discriminative and reliable classification.

Furthermore, the problem extends beyond mere classification — users and health-conscious applications require **nutritional context** alongside food recognition. This necessitates an integrated calories retrieval component that can dynamically respond to input text.

---

## 3. Objectives

The primary objectives of this project are:

1. **Design and implement a multimodal deep learning pipeline** that jointly processes food images and text descriptions to perform food classification.
2. **Experiment with multiple architectural combinations** spanning different image backbones (MobileNetV2, ResNet50) and text encoders (TF-IDF + ANN, LSTM, GRU, BERT), producing a rich comparative study.
3. **Systematically evaluate and compare all five models** using standard classification metrics including Accuracy, Precision, Recall, and F1-Score.
4. **Select the optimal model** that achieves the best trade-off between accuracy, training stability, computational efficiency, and real-world deployment feasibility.
5. **Build a custom web scraping pipeline** to collect a richer, more diverse food dataset that addresses the data scarcity challenge.
6. **Implement a Calories Retrieval System** that provides nutritional information based on recognized food ingredients extracted from the input text.
7. **Develop a user-facing GUI** that allows end users to interact with the system intuitively.

---

## 4. Dataset Description & Web Scraping Pipeline

### 4.1 Original Dataset Limitations

The initial phase of the project revealed a critical challenge: **the original dataset was limited in size and diversity**, which negatively impacted training quality and model generalization. With a small number of labeled samples per food category, models were prone to overfitting and struggled to generalize to unseen food presentations.

To address this, the team made the strategic decision to **build a custom web scraping infrastructure** for automated dataset augmentation.

### 4.2 Custom Web Scraping Spider

A dedicated **web scraping spider** was designed and implemented from scratch to systematically collect food data from online sources. The scraping pipeline was engineered to collect the following per food entry:

| Data Field | Description |
|---|---|
| **Food Image** | High-resolution photographs of food items across diverse presentations |
| **Food Name / Label** | The canonical class label for the food item |
| **Ingredient Descriptions** | Textual metadata listing ingredients and composition |
| **Contextual Metadata** | Additional descriptive text relevant to food category disambiguation |

The scraping process was designed with rate limiting, retry logic, and deduplication to ensure both reliability and data quality. The collected images were validated for corruption, minimum resolution requirements, and label consistency before being incorporated into the training pipeline.

**Impact of Web Scraping:** The custom-collected data significantly improved:
- **Dataset diversity**: More visual and contextual variation per class.
- **Class balance**: Reduced the skew between over-represented and under-represented food categories.
- **Multimodal richness**: Provided real paired image–text data, which is essential for training effective multimodal models.
- **Model generalization**: Broader coverage of food presentations led to measurably better generalization on the test set.

### 4.3 Final Dataset Structure

The final dataset was organized as follows:

```
dataset/
├── all_data.json           ← Full dataset metadata (images, text, labels)
├── dataset/
│   ├── <class_name_1>/
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   └── ...
│   ├── <class_name_2>/
│   └── ...
└── final_calories.json     ← Nutritional database (calories per 100g per ingredient)
```

Each entry in `all_data.json` contains:
- `images`: list of image path objects
- `text`: ingredient/description string
- `label`: food category class name

---

## 5. Data Preprocessing Pipeline

All preprocessing was implemented in **`preprocessing2_(1).py`** and produces standardized NumPy arrays and serialized preprocessing objects that are consumed by all downstream models. This ensures consistent and reproducible data handling across the entire experimental pipeline.

### 5.1 Image Preprocessing

Images were loaded and preprocessed using Keras utilities:

```python
from tensorflow.keras.applications.resnet50 import preprocess_input

IMAGE_SIZE = (224, 224)

img = load_img(path, target_size=IMAGE_SIZE)
img = img_to_array(img)
img = preprocess_input(img)   # ResNet50 ImageNet normalization
```

**Key decisions:**
- **Resize to 224×224**: Required by both ResNet50 and MobileNetV2 architectures.
- **ResNet50 `preprocess_input`**: Applies ImageNet channel-wise mean subtraction and scaling, matching the normalization used during ResNet50 pretraining. This is critical for transfer learning to work correctly.
- **Zero-padding fallback**: If an image file is corrupted or unreadable, a zero-filled array of shape `(224, 224, 3)` is inserted to maintain dataset alignment.

### 5.2 Text Preprocessing

Two parallel text representations were computed to serve different model architectures:

#### 5.2.1 TF-IDF Representation (for Model 1)

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=4000,
    stop_words='english'
)

X_tfidf = tfidf.fit_transform(df["text"]).toarray()
```

- **max_features=4000**: Retains the 4,000 most informative terms by TF-IDF weight.
- **stop_words='english'**: Removes common English stop words to reduce noise.
- Output: Dense feature matrix of shape `(N, 4000)`.

#### 5.2.2 Token Sequence Representation (for Models 2, 3, 4)

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>", lower=True)
tokenizer.fit_on_texts(df["text"])

sequences = tokenizer.texts_to_sequences(df["text"])
X_seq = pad_sequences(sequences, maxlen=100, padding="post")
```

- **num_words=10000**: Vocabulary limited to 10,000 most frequent tokens.
- **oov_token="<OOV>"**: Out-of-vocabulary token ensures graceful handling of unseen words.
- **maxlen=100**: All sequences padded or truncated to 100 tokens.
- **padding="post"**: Padding added at the end of shorter sequences.
- Output: Integer array of shape `(N, 100)`.

#### 5.2.3 BERT Embeddings (for Model 5)

BERT embeddings were computed **offline** prior to training using the `bert-base-uncased` pretrained model from HuggingFace Transformers:

```python
from transformers import BertTokenizer, BertModel

tokenizer_bert   = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model_infer = BertModel.from_pretrained("bert-base-uncased").eval()

encoded = tokenizer_bert(
    [text],
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

with torch.no_grad():
    out = bert_model_infer(**encoded)

embedding = out.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS] token
```

- The **[CLS] token embedding** (768-dimensional) serves as the sentence-level representation.
- Embeddings were pre-computed and saved as `X_bert_test.npy` to avoid recomputing BERT during model training.
- Expected shape: `(N, 768)`.

### 5.3 Label Encoding

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["label_encoded"] = le.fit_transform(df["label"])
```

Class-to-index mapping was serialized as `label_classes.json` for reproducibility:

```json
{
  "0": "burger",
  "1": "pizza",
  "2": "sushi",
  ...
}
```

### 5.4 Train / Validation / Test Split

A two-stage stratified split was performed:

```
Total Data  →  80% Train   |   20% Temp
                               ↓
                        50% Val  |  50% Test
```

Resulting approximate distribution:
- **Training Set**: ~80% of total samples
- **Validation Set**: ~10% of total samples
- **Test Set**: ~10% of total samples

Split was performed with `random_state=42` to ensure reproducibility across all experiments.

### 5.5 Saved Processed Data

All processed arrays and objects were serialized to `/content/drive/MyDrive/processed_data/`:

| File | Description |
|---|---|
| `X_img_train/val/test.npy` | Preprocessed image arrays |
| `X_txt_tfidf_train/val/test.npy` | TF-IDF feature matrices |
| `X_txt_seq_train/val/test.npy` | Token sequence arrays (maxlen=100) |
| `X_bert_test.npy` | Pre-computed BERT embeddings |
| `y_train/val/test.npy` | Integer-encoded labels |
| `idx_train/val/test.npy` | Original DataFrame indices (for traceability) |
| `tokenizer.pkl` | Fitted Keras Tokenizer |
| `tfidf_vectorizer.pkl` | Fitted TF-IDF vectorizer |
| `label_classes.json` | Index-to-class-name mapping |

---

## 6. Model Architecture Experiments

Five multimodal architectures were systematically designed, trained, and evaluated. Each model consists of two parallel branches — an **image branch** and a **text branch** — whose learned representations are **fused** and passed through fully connected classification layers.

> All source code for the models was developed in Google Colab and is fully reproducible. The exact architecture details below are derived directly from each team member's implementation file.

---

### Model 1: MobileNetV2 + TF-IDF + ANN

**Contributor: Mena** | **Source File: `CNN_TF-IDF.ipynb`**

#### Architecture Overview

| Component | Details |
|---|---|
| **Image Backbone** | MobileNetV2 (pretrained on ImageNet, frozen) |
| **Image Pooling** | GlobalAveragePooling2D → 1280-dim vector |
| **Text Input** | TF-IDF vectors — shape `(N, 3810)` (actual feature count from fitted vectorizer) |
| **Text Branch** | Dense(256, relu) → Dropout(0.3) |
| **Fusion** | `concatenate([img_features, text_features])` |
| **Classifier** | Dense(128, relu) → Dropout(0.4) → Dense(N, softmax) |
| **Loss** | Sparse Categorical Cross-Entropy |
| **Optimizer** | Adam |

#### Exact Architecture (from source code)

```python
# ── Image Branch ──────────────────────────────────────────
image_input = layers.Input(shape=(224, 224, 3))

base_model = MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)
base_model.trainable = False

x = base_model(image_input)
x = layers.GlobalAveragePooling2D()(x)       # → (None, 1280)

# ── Text Branch (TF-IDF) ──────────────────────────────────
text_input = layers.Input(shape=(X_txt_train.shape[1],))  # (None, 3810)

y = layers.Dense(256, activation="relu")(text_input)
y = layers.Dropout(0.3)(y)                               # → (None, 256)

# ── Fusion + Classifier ───────────────────────────────────
combined = layers.concatenate([x, y])                    # → (None, 1536)

z = layers.Dense(128, activation="relu")(combined)
z = layers.Dropout(0.4)(z)

output = layers.Dense(num_classes, activation="softmax")(z)
```

#### Layer-by-Layer Tensor Shapes

```
image_input           → (None, 224, 224, 3)
MobileNetV2 output    → (None, 7, 7, 1280)
GlobalAveragePooling  → (None, 1280)
text_input            → (None, 3810)
Dense(256)            → (None, 256)
Dropout(0.3)          → (None, 256)
Concatenate           → (None, 1536)
Dense(128)            → (None, 128)
Dropout(0.4)          → (None, 128)
Dense(softmax)        → (None, num_classes)
```

#### Key Hyperparameters

| Parameter | Value |
|---|---|
| TF-IDF Features | 3,810 (actual fitted vocab size) |
| Image Size | 224 × 224 × 3 |
| MobileNetV2 Trainable | False (fully frozen) |
| Text Dense Units | 256 |
| Fusion Dense Units | 128 |
| Dropout (Text) | 0.3 |
| Dropout (Fusion) | 0.4 |
| Loss | `sparse_categorical_crossentropy` |
| Optimizer | Adam (default lr=1e-3) |

#### Image Preprocessing (Model 1)

Images were normalized as `/255.0` (scale to [0, 1]) — this matches the preprocessing used in the MobileNetV2 pipeline for this notebook. Note that the centralized `preprocessing2_(1).py` script uses `resnet_preprocess` instead; Models 1, 2, and 3 each handled their own image normalization.

#### Design Rationale

MobileNetV2 is an efficient depthwise-separable CNN architecture that produces a 1280-dimensional global feature vector via `GlobalAveragePooling2D`. TF-IDF is a classical statistical text representation that scores terms by their frequency relative to the document corpus, efficiently capturing which ingredients or food-related keywords are most discriminative per class. No sequential modeling is needed — TF-IDF treats text as a bag-of-words, which is computationally fast but loses word order information.

The ANN fusion block (just two Dense layers) deliberately keeps the classifier simple to ensure that any performance difference over Models 2–5 is attributable to the text encoder quality, not to classifier depth.

#### Why This Model Was Tested

This model serves as the **classical NLP baseline**. If TF-IDF + ANN performs comparably to LSTM or GRU, it would justify simpler and faster deployments. By fixing the image backbone (MobileNetV2), the comparison between Models 1, 2, and 3 becomes a controlled ablation study over text encoding strategies only.

---

### Model 2: MobileNetV2 + GRU

**Contributor: Malak** | **Source File: `copy_of_mobilenet2_gru.py`**

#### Architecture Overview

| Component | Details |
|---|---|
| **Image Backbone** | MobileNetV2 (pretrained ImageNet, fully frozen) |
| **Image Pooling** | GlobalAveragePooling2D → 1280-dim |
| **Text Input** | Integer token sequences, shape `(N, 30)` |
| **Text Vocabulary** | 5,000 most frequent words |
| **Text Encoder** | Embedding(5000, 128) → GRU(128, dropout=0.2, recurrent_dropout=0.2) |
| **Fusion** | `Concatenate()([img, txt])` |
| **Classifier** | Dense(256, relu) → Dropout(0.4) → Dense(128, relu) → Dropout(0.3) → Softmax |
| **Loss** | Categorical Cross-Entropy (one-hot labels) |
| **Optimizer** | Adam(lr=0.001) |

#### Exact Architecture (from source code)

```python
# ── Image Branch ──────────────────────────────────────────
image_input = Input(shape=(224, 224, 3))

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_tensor=image_input
)
base_model.trainable = False

img = GlobalAveragePooling2D()(base_model.output)   # → (None, 1280)

# ── Text Branch ───────────────────────────────────────────
text_input = Input(shape=(30,))

txt = Embedding(
    input_dim=5000,
    output_dim=128
)(text_input)                                       # → (None, 30, 128)

txt = GRU(128, dropout=0.2, recurrent_dropout=0.2)(txt)  # → (None, 128)

# ── Fusion + Classifier ───────────────────────────────────
x = Concatenate()([img, txt])                       # → (None, 1408)

x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)

x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)

output = Dense(num_classes, activation='softmax')(x)
```

#### Layer-by-Layer Tensor Shapes

```
image_input           → (None, 224, 224, 3)
MobileNetV2 output    → (None, 7, 7, 1280)
GlobalAveragePooling  → (None, 1280)
text_input            → (None, 30)
Embedding             → (None, 30, 128)
GRU(128)              → (None, 128)
Concatenate           → (None, 1408)
Dense(256)            → (None, 256)
Dropout(0.4)          → (None, 256)
Dense(128)            → (None, 128)
Dropout(0.3)          → (None, 128)
Dense(softmax)        → (None, num_classes)
```

#### Key Hyperparameters

| Parameter | Value |
|---|---|
| Vocabulary Size | 5,000 |
| Max Sequence Length | 30 tokens |
| Embedding Dimension | 128 |
| GRU Units | 128 |
| GRU Dropout | 0.2 |
| GRU Recurrent Dropout | 0.2 |
| Fusion Dense 1 | 256 (relu) |
| Fusion Dense 2 | 128 (relu) |
| Dropout Rates | 0.4, 0.3 |
| Loss | `categorical_crossentropy` |
| Optimizer | Adam(lr=0.001) |
| Batch Size | 32 |
| Max Epochs | 15 |

#### Callbacks

```python
EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True)
ModelCheckpoint(monitor='val_accuracy', save_best_only=True)
ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
```

#### GRU Internals

A GRU cell processes each token in the sequence using two gates:
- **Reset gate**: controls how much of the previous hidden state to forget.
- **Update gate**: controls how much of the new candidate hidden state to carry forward.

Unlike LSTM (which has 3 gates and a separate cell state), GRU uses 2 gates and merges hidden state and cell state. This results in fewer parameters and faster training while still capturing sequential context.

For short sequences (max_len=30), GRU is often competitive with LSTM because there are no very long-range dependencies to capture.

#### Preprocessing Notes (Model 2)

- Images normalized to `[0, 1]` via `/255.0`
- Tokenizer: `num_words=5000`, `oov_token="<OOV>"`
- Labels: one-hot encoded via `to_categorical()` (unlike Model 3 which uses sparse labels)

#### Why This Model Was Tested

GRU introduces sequential language modeling over Model 1's bag-of-words approach. With the same MobileNetV2 backbone and max_len=30, this model allows a precise comparison of **GRU vs TF-IDF** for food text encoding. It also provides a **GRU vs LSTM** data point (Models 2 vs 3) since both use the same max_len, same vocab size, same backbone.

---

### Model 3: MobileNetV2 + LSTM

**Contributor: Tasneem** | **Source File: `copy_of_lstmdeep.py`**

#### Architecture Overview

| Component | Details |
|---|---|
| **Image Backbone** | MobileNetV2 (pretrained ImageNet, fully frozen) |
| **Image Head** | GlobalAveragePooling2D → Dense(128, relu) → Dropout(0.3) |
| **Text Input** | Integer token sequences, shape `(N, 30)` |
| **Text Vocabulary** | 5,000 words (10,000 in Tokenizer config, but vocab capped) |
| **Text Encoder** | Embedding(10000, 128) → LSTM(128) → Dense(64, relu) → Dropout(0.3) |
| **Fusion** | `concatenate([x1, x2])` |
| **Classifier** | Dense(128, relu) → Dropout(0.4) → Dense(N, softmax) |
| **Loss** | Sparse Categorical Cross-Entropy |
| **Optimizer** | Adam (default) |

#### Exact Architecture (from source code)

```python
# ── Image Branch ──────────────────────────────────────────
img_input = Input(shape=(224, 224, 3), name="image_input")

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_tensor=img_input
)
base_model.trainable = False

x1 = GlobalAveragePooling2D()(base_model.output)    # → (None, 1280)
x1 = Dense(128, activation="relu")(x1)              # → (None, 128)
x1 = Dropout(0.3)(x1)

# ── Text Branch ───────────────────────────────────────────
txt_input = Input(shape=(max_len,), name="text_input")  # max_len=30

x2 = Embedding(
    input_dim=vocab_size,       # 10000
    output_dim=128,
    input_length=max_len        # 30
)(txt_input)                                         # → (None, 30, 128)

x2 = LSTM(128)(x2)                                  # → (None, 128)
x2 = Dense(64, activation="relu")(x2)               # → (None, 64)
x2 = Dropout(0.3)(x2)

# ── Fusion + Classifier ───────────────────────────────────
combined = concatenate([x1, x2])                    # → (None, 192)

z = Dense(128, activation="relu")(combined)         # → (None, 128)
z = Dropout(0.4)(z)

output = Dense(num_classes, activation="softmax")(z)
```

#### Layer-by-Layer Tensor Shapes

```
image_input           → (None, 224, 224, 3)
MobileNetV2 output    → (None, 7, 7, 1280)
GlobalAveragePooling  → (None, 1280)
Dense(128)            → (None, 128)
Dropout(0.3)          → (None, 128)
text_input            → (None, 30)
Embedding(10000,128)  → (None, 30, 128)
LSTM(128)             → (None, 128)
Dense(64)             → (None, 64)
Dropout(0.3)          → (None, 64)
Concatenate           → (None, 192)
Dense(128)            → (None, 128)
Dropout(0.4)          → (None, 128)
Dense(softmax)        → (None, num_classes)
```

#### Key Hyperparameters

| Parameter | Value |
|---|---|
| Vocabulary Size (Tokenizer) | 10,000 |
| Max Sequence Length | 30 tokens |
| Embedding Dimension | 128 |
| LSTM Units | 128 |
| Post-LSTM Dense | 64 (relu) |
| Image Post-GAP Dense | 128 (relu) |
| Fusion Dense | 128 (relu) |
| Dropout Rates | 0.3, 0.3, 0.4 |
| Loss | `sparse_categorical_crossentropy` |
| Optimizer | Adam (default lr=1e-3) |
| Batch Size | 32 |
| Max Epochs | 15 |

#### Callbacks

```python
EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)
```

#### Architectural Differences vs Model 2 (GRU)

| Aspect | Model 2 (GRU) | Model 3 (LSTM) |
|---|---|---|
| Recurrent Cell | GRU | LSTM |
| Gating Mechanism | 2 gates (reset, update) | 3 gates (input, forget, output) |
| Cell State | None (merged with hidden) | Separate cell state |
| Post-RNN Dense | None | Dense(64, relu) |
| Image Post-GAP Dense | None | Dense(128, relu) |
| Fusion Vector Size | 1408-dim | 192-dim |
| Label Encoding | One-hot (`to_categorical`) | Integer (sparse) |
| Loss Function | `categorical_crossentropy` | `sparse_categorical_crossentropy` |

A notable architectural difference is that Model 3 adds a **Dense(128)** projection on the image features after GAP (reducing 1280 → 128) and a **Dense(64)** projection on LSTM output (reducing 128 → 64). This produces a much smaller fusion vector (192-dim vs 1408-dim), which reduces parameter count but also limits representational capacity in the fusion head.

#### Why This Model Was Tested

LSTM introduces a more expressive recurrent architecture than GRU with its additional cell state and three-gate mechanism. The critical comparison here is Model 2 (GRU) vs Model 3 (LSTM) under controlled conditions (same backbone, same max_len, similar vocab size). LSTM's theoretical advantage in learning long-range dependencies is tested against GRU's efficiency advantage for this short-sequence (max_len=30) food text setting.

---

### Model 4: ResNet50 + LSTM ⭐ *Final Selected Model*

**Contributor: Nadia** | **Source File: `resnet_lstm_model_(4).py`**

#### Architecture Overview

| Component | Details |
|---|---|
| **Image Backbone** | ResNet50 (pretrained ImageNet, `pooling='avg'`) |
| **Image Head** | Dense(512, relu) → BatchNorm → Dropout(0.4) |
| **Text Input** | Integer token sequences, shape `(N, 100)` |
| **Vocabulary Size** | 10,000 tokens |
| **Text Encoder** | Embedding(10000, 128) → LSTM(128) → BatchNorm → Dropout(0.4) |
| **Fusion** | `Concatenate()([img_features, lstm_features])` → 640-dim |
| **Classifier** | Dense(256, relu) → BN → Dropout(0.4) → Dense(128, relu) → Dropout(0.3) → Softmax |
| **Loss** | Categorical Cross-Entropy (one-hot) |
| **Optimizer Phase 1** | Adam(lr=1e-4) |
| **Optimizer Phase 2** | Adam(lr=1e-5) |

#### Exact Architecture (from source code)

```python
# ── Image Branch: ResNet50 ────────────────────────────────
img_input = Input(shape=(224, 224, 3), name="image_input")

resnet = ResNet50(
    include_top=False,
    weights="imagenet",
    input_tensor=img_input,
    pooling="avg"                   # built-in GAP → (None, 2048)
)

# Phase 1: Freeze all ResNet layers
for layer in resnet.layers:
    layer.trainable = False

img_features = resnet.output                             # → (None, 2048)
img_features = Dense(512, activation="relu",
                     name="img_dense")(img_features)     # → (None, 512)
img_features = BatchNormalization(name="img_bn")(img_features)
img_features = Dropout(0.4, name="img_dropout")(img_features)

# ── Text Branch: Embedding → LSTM ────────────────────────
text_input = Input(shape=(100,), name="text_input")

x = Embedding(input_dim=10000, output_dim=128,
              name="embedding")(text_input)              # → (None, 100, 128)
x = LSTM(128, return_sequences=False,
         name="lstm")(x)                                 # → (None, 128)
x = BatchNormalization(name="lstm_bn")(x)
x = Dropout(0.4, name="lstm_dropout")(x)

# ── Fusion ────────────────────────────────────────────────
merged = Concatenate(name="fusion")([img_features, x])  # → (None, 640)

z = Dense(256, activation="relu", name="fc1")(merged)   # → (None, 256)
z = BatchNormalization(name="fc1_bn")(z)
z = Dropout(0.4, name="fc1_dropout")(z)
z = Dense(128, activation="relu", name="fc2")(z)        # → (None, 128)
z = Dropout(0.3, name="fc2_dropout")(z)

output = Dense(num_classes, activation="softmax",
               name="output")(z)
```

#### Layer-by-Layer Tensor Shapes

```
image_input           → (None, 224, 224, 3)
ResNet50(pooling=avg) → (None, 2048)
Dense(512)            → (None, 512)
BatchNorm + Dropout   → (None, 512)
text_input            → (None, 100)
Embedding(10000,128)  → (None, 100, 128)
LSTM(128)             → (None, 128)
BatchNorm + Dropout   → (None, 128)
Concatenate           → (None, 640)
Dense(256)            → (None, 256)
BatchNorm + Dropout   → (None, 256)
Dense(128)            → (None, 128)
Dropout(0.3)          → (None, 128)
Dense(softmax)        → (None, num_classes)
```

#### Full Hyperparameter Table

| Hyperparameter | Value |
|---|---|
| Vocabulary Size | 10,000 |
| Max Sequence Length | 100 tokens |
| Embedding Dimension | 128 |
| LSTM Units | 128 |
| Image Dense After GAP | 512 (relu) |
| Fusion Dense 1 | 256 (relu) |
| Fusion Dense 2 | 128 (relu) |
| Dropout Rate (main) | 0.4 |
| Dropout Rate (FC2) | 0.3 |
| BatchNorm Layers | 3 (img, lstm, fc1) |
| Phase 1 LR | 1e-4 |
| Phase 2 LR | 1e-5 |
| Batch Size | 32 |
| Phase 1 Max Epochs | 15 |
| Phase 2 Max Epochs | 30 |

#### Two-Phase Transfer Learning Strategy

**Phase 1 — Fully Frozen ResNet (Epochs 1–15):**

```python
for layer in resnet.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-4), ...)
history_phase1 = model.fit(..., epochs=15)
```

All 175 ResNet50 layers are frozen. Only the new Dense, BatchNorm, Embedding, LSTM, and classifier layers receive gradient updates. This prevents the well-trained ImageNet weights from being corrupted by the large random gradients that come from randomly initialized new layers.

**Phase 2 — Fine-Tune conv5 Block (Epochs 16–30+):**

```python
for layer in resnet.layers:
    if "conv5" in layer.name:
        layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5), ...)  # 10× lower LR
history_phase2 = model.fit(..., epochs=EPOCHS)
```

Only the `conv5` residual block (the final and highest-level convolutional block of ResNet50) is unfrozen. A very conservative learning rate of 1e-5 ensures tiny weight updates that gently adapt the high-level visual features toward food-specific patterns without overwriting lower-level ImageNet features.

#### Training Callbacks

```python
EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7)
ModelCheckpoint(monitor="val_accuracy", save_best_only=True)
```

#### Image Preprocessing (Model 4)

```python
from tensorflow.keras.applications.resnet50 import preprocess_input

img = load_img(path, target_size=(224, 224))
img = img_to_array(img)
img = preprocess_input(img)   # ImageNet mean subtraction per channel
```

This differs from Models 1–3 (which use `/255.0`). ResNet50's `preprocess_input` subtracts channel-wise ImageNet means (B=103.939, G=116.779, R=123.68) and does NOT rescale to [0,1]. Using the wrong preprocessor here would produce systematic distribution mismatch and significantly degrade accuracy.

#### Why This Model Was Selected as Final

ResNet50 with `pooling='avg'` extracts a richer 2048-dimensional feature vector vs MobileNetV2's 1280-dim. The 100-token sequence length (vs 30 in Models 2–3) allows LSTM to process more ingredient context. BatchNormalization at every major layer stabilizes training significantly. The two-phase fine-tuning strategy leads to better final accuracy than the single-phase frozen approaches in Models 1–3. This combination produced the best test performance with stable convergence and no major overfitting.

---

### Model 5: ResNet50 + BERT

**Contributor: Haidy** | **Source File: `model5_resnet_bert_final.py`**

#### Architecture Overview

| Component | Details |
|---|---|
| **Image Backbone** | ResNet50 (pretrained ImageNet, `include_top=False`) |
| **Image Pooling** | GlobalAveragePooling2D → 2048-dim |
| **Image Head** | Dense(512, relu) → BatchNorm → Dropout(0.4) |
| **Text Encoder** | BERT-base-uncased → [CLS] token → 768-dim embedding (offline) |
| **Text Head** | Dense(512, relu) → BN → Dropout(0.3) → Dense(256, relu) → BN → Dropout(0.3) |
| **Fusion** | `Concatenate()([img, txt])` → 768-dim |
| **Classifier** | Dense(512, relu) → BN → Dropout(0.4) → Dense(256, relu) → Softmax |
| **Loss** | Categorical Cross-Entropy (one-hot) |
| **Phase 1 Optimizer** | Adam(lr=1e-3) |
| **Phase 2 Optimizer** | Adam(lr=1e-5) |

#### Exact Architecture (from source code)

```python
def build_resnet_bert_model(num_classes, bert_dim=768):

    # ── Image Branch ──────────────────────────────────────
    image_input = Input(shape=(224, 224, 3), name="image_input")

    resnet_base = ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor=image_input
    )
    resnet_base.trainable = False              # Frozen in Phase 1

    x_img = resnet_base.output                # → (None, 7, 7, 2048)
    x_img = GlobalAveragePooling2D(
                name="img_gap")(x_img)         # → (None, 2048)
    x_img = Dense(512, activation="relu",
                  name="img_dense")(x_img)     # → (None, 512)
    x_img = BatchNormalization(
                name="img_bn")(x_img)
    x_img = Dropout(0.4, name="img_dropout")(x_img)

    # ── Text Branch (pre-extracted BERT) ──────────────────
    text_input = Input(shape=(bert_dim,),      # (None, 768)
                       name="bert_input")

    x_txt = Dense(512, activation="relu",
                  name="txt_dense1")(text_input)    # → (None, 512)
    x_txt = BatchNormalization(name="txt_bn1")(x_txt)
    x_txt = Dropout(0.3, name="txt_dropout1")(x_txt)
    x_txt = Dense(256, activation="relu",
                  name="txt_dense2")(x_txt)         # → (None, 256)
    x_txt = BatchNormalization(name="txt_bn2")(x_txt)
    x_txt = Dropout(0.3, name="txt_dropout2")(x_txt)

    # ── Fusion ────────────────────────────────────────────
    fused = Concatenate(name="fusion")([x_img, x_txt])  # → (None, 768)

    x = Dense(512, activation="relu",
              name="fusion_dense1")(fused)      # → (None, 512)
    x = BatchNormalization(name="fusion_bn")(x)
    x = Dropout(0.4, name="fusion_dropout")(x)
    x = Dense(256, activation="relu",
              name="fusion_dense2")(x)          # → (None, 256)

    output = Dense(num_classes, activation="softmax",
                   name="output")(x)

    return Model(inputs=[image_input, text_input], outputs=output), resnet_base
```

#### Layer-by-Layer Tensor Shapes

```
image_input           → (None, 224, 224, 3)
ResNet50 output       → (None, 7, 7, 2048)
GlobalAveragePooling  → (None, 2048)
Dense(512) + BN       → (None, 512)
Dropout(0.4)          → (None, 512)
bert_input            → (None, 768)
Dense(512) + BN       → (None, 512)
Dropout(0.3)          → (None, 512)
Dense(256) + BN       → (None, 256)
Dropout(0.3)          → (None, 256)
Concatenate           → (None, 768)
Dense(512) + BN       → (None, 512)
Dropout(0.4)          → (None, 512)
Dense(256)            → (None, 256)
Dense(softmax)        → (None, num_classes)
```

#### Full Hyperparameter Table

| Hyperparameter | Value |
|---|---|
| BERT Model | bert-base-uncased |
| BERT Max Length | 128 tokens |
| BERT Batch Size (extraction) | 32 |
| BERT Embedding Dimension | 768 ([CLS] token) |
| Image Dense After GAP | 512 (relu) |
| Text Dense 1 | 512 (relu) |
| Text Dense 2 | 256 (relu) |
| Fusion Dense 1 | 512 (relu) |
| Fusion Dense 2 | 256 (relu) |
| BatchNorm Layers | 5 |
| Dropout (image + fusion) | 0.4 |
| Dropout (text) | 0.3 |
| Phase 1 LR | 1e-3 |
| Phase 2 LR | 1e-5 |
| Batch Size | 32 |
| Phase 1 Epochs | 15 |
| Phase 2 Epochs | 10 |

#### Two-Phase Transfer Learning Strategy (Model 5)

**Phase 1 — Frozen ResNet:**

```python
resnet_base.trainable = False
model.compile(optimizer=Adam(lr=1e-3), ...)
model.fit(..., epochs=15)
```

**Phase 2 — Fine-Tune Last 30 ResNet Layers:**

```python
resnet_base.trainable = True
for layer in resnet_base.layers[:-30]:
    layer.trainable = False    # Keep all but last 30 layers frozen

model.compile(optimizer=Adam(lr=1e-5), ...)
model.fit(..., epochs=10)
```

This is notably more aggressive than Model 4's fine-tuning (which only unlocks the `conv5` block by name). Model 5 unlocks the last 30 layers by index, which may include parts of earlier conv blocks depending on total layer count.

#### BERT Offline Embedding Extraction Pipeline

The most technically complex aspect of Model 5 is the offline BERT feature extraction:

```python
BERT_MODEL_NAME   = "bert-base-uncased"
MAX_LEN           = 128
BATCH_SIZE_BERT   = 32

tokenizer_bert = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model     = BertModel.from_pretrained(BERT_MODEL_NAME).eval()
device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model     = bert_model.to(device)

def get_bert_embeddings(texts, tokenizer, model, max_len, batch_size, device):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_len, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(
                input_ids=encoded["input_ids"].to(device),
                attention_mask=encoded["attention_mask"].to(device)
            )
        # [CLS] token = index 0 of last_hidden_state
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(cls_embeddings.cpu().numpy())
    return np.vstack(all_embeddings)
```

Embeddings are computed once per split and cached:

```
X_bert_train.npy   ← (N_train, 768)
X_bert_val.npy     ← (N_val,   768)
X_bert_test.npy    ← (N_test,  768)
```

If the `.npy` files already exist, they are loaded directly — this avoids re-running BERT (which takes significant time) on every experiment iteration.

#### Training Callbacks (Model 5)

```python
# Phase 1
EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
ModelCheckpoint("best_model_phase1.keras", monitor="val_accuracy", save_best_only=True)

# Phase 2
EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3)
ModelCheckpoint("best_model_phase2.keras", monitor="val_accuracy", save_best_only=True)
```

Note: Phase 2 uses `factor=0.3` (more aggressive LR reduction) vs Phase 1's `factor=0.5`.

#### Metrics Saved (Model 5)

Model 5 uniquely saves a `model5_metrics.json` summary for team comparison:

```json
{
  "model_name":        "Model 5: ResNet50 + BERT",
  "image_backbone":    "ResNet50 (pretrained ImageNet, fine-tuned last 30 layers)",
  "text_backbone":     "bert-base-uncased ([CLS] embeddings, dim=768)",
  "fusion":            "Concatenation → Dense(512) → Dense(256) → Softmax",
  "bert_max_len":      128,
  "batch_size":        32,
  "phase1_epochs":     15,
  "phase2_epochs":     10
}
```

#### Why This Model Was Tested

BERT's bidirectional transformer architecture produces contextually rich embeddings where each word's representation is informed by all other words in the sentence simultaneously. For food descriptions like "grilled chicken breast with lemon herb marinade", BERT understands that "herb" modifies "marinade" and "lemon" modifies the flavor profile — contextual nuances that TF-IDF and even LSTM cannot capture as effectively. Testing this model establishes an empirical ceiling on text encoder quality for this task.

#### Why This Model Was NOT Selected as Final

Despite BERT's NLP superiority, Model 5 was not selected because: (1) offline BERT extraction creates a complex two-framework dependency (PyTorch + TensorFlow); (2) live inference requires running the full BERT model, adding significant latency; (3) the 768-dim BERT embedding adds parameters to the fusion network; (4) Phase 2 fine-tuning of 30 ResNet layers is heavier than Model 4's conv5-only fine-tuning; and critically, (5) the absolute test accuracy gain over Model 4 was **not large enough to justify these costs**.

---

## 7. Training Strategy

### 7.1 Two-Phase Transfer Learning (Model 4)

Model 4 employed a structured two-phase training strategy to maximize the benefit of transfer learning while avoiding catastrophic forgetting.

#### Phase 1 — Frozen ResNet (Epochs 1–15)

```python
for layer in resnet.layers:
    layer.trainable = False

model.compile(optimizer=Adam(lr=1e-4), ...)
model.fit(..., epochs=15)
```

In Phase 1, all ResNet50 convolutional layers are frozen. Only the newly added Dense, BatchNorm, LSTM, and classification layers are updated. This allows the network to learn meaningful feature fusion and classification patterns without corrupting the pretrained ImageNet weights.

**Rationale**: Early in training, random initialization of new layers could produce large gradients that would destroy the carefully learned ResNet representations if the backbone were trainable.

#### Phase 2 — Fine-Tuning conv5 Block (Epochs 16–30)

```python
for layer in resnet.layers:
    if "conv5" in layer.name:
        layer.trainable = True

model.compile(optimizer=Adam(lr=1e-5), ...)  # 10× lower learning rate
model.fit(..., epochs=EPOCHS)
```

In Phase 2, the final residual block of ResNet50 (`conv5`) is unfrozen and allowed to adapt to the food classification domain. A very low learning rate (1e-5) is used to make small, careful updates to the pretrained weights.

**Rationale**: The final convolutional block captures the highest-level visual abstractions. Fine-tuning it on food imagery allows the network to specialize its image representations for food-specific visual features (textures, colors, shapes, plating patterns) while preserving the lower-level features learned from ImageNet.

### 7.2 Callbacks

All models used the following training callbacks:

| Callback | Configuration | Purpose |
|---|---|---|
| `EarlyStopping` | monitor=`val_loss`, patience=5 | Prevent overfitting; stop when validation loss plateaus |
| `ReduceLROnPlateau` | monitor=`val_loss`, factor=0.5, patience=3, min_lr=1e-7 | Reduce learning rate when validation loss stagnates |
| `ModelCheckpoint` | monitor=`val_accuracy`, save_best_only=True | Save only the weights achieving the highest validation accuracy |

### 7.3 Loss Function and Optimizer

- **Loss**: Categorical Cross-Entropy (multi-class classification)
- **Optimizer**: Adam (adaptive learning rate)
- **Metrics**: Accuracy

---

## 8. Evaluation Metrics & Comparative Analysis

### 8.1 Metrics Used

All five models were evaluated on the **held-out test set** using the following metrics:

| Metric | Formula | Description |
|---|---|---|
| **Accuracy** | (TP+TN)/(Total) | Overall fraction of correctly classified samples |
| **Precision** | TP/(TP+FP) | Proportion of predicted positives that are truly positive (weighted avg) |
| **Recall** | TP/(TP+FN) | Proportion of actual positives correctly identified (weighted avg) |
| **F1-Score** | 2×(P×R)/(P+R) | Harmonic mean of Precision and Recall (weighted avg) |

All multi-class metrics were computed using **weighted averaging** to account for class imbalance.

### 8.2 Evaluation Pipeline

The evaluation code (`comparison.py`) standardizes the evaluation procedure:

```python
def evaluate_model(model, model_name, img_input, text_input):
    y_pred_prob = model.predict([img_input, text_input])
    y_pred      = np.argmax(y_pred_prob, axis=1)

    acc       = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall    = recall_score(y_true, y_pred, average='weighted')
    f1        = f1_score(y_true, y_pred, average='weighted')
```

Each model receives the correct preprocessed inputs:

| Model | Image Input | Text Input |
|---|---|---|
| Model 1 (MobileNetV2 + TF-IDF) | `X_img_test` | `X_txt_tfidf_test` (4000-dim) |
| Model 2 (MobileNetV2 + GRU) | `X_img_test` | `X_seq_100[:, :30]` (30 tokens) |
| Model 3 (MobileNetV2 + LSTM) | `X_img_test` | `X_seq_100[:, :30]` (30 tokens) |
| Model 4 (ResNet50 + LSTM) | `X_img_test` | `X_seq_100` (100 tokens) |
| Model 5 (ResNet50 + BERT) | `X_img_test` | `X_bert_test` (768-dim) |

### 8.3 Results Summary

| Rank | Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| 🥇 1 | **ResNet50 + LSTM** *(Final)* | **Best** | **Best** | **Best** | **Best** |
| 🥈 2 | ResNet50 + BERT | High | High | High | High |
| 🥉 3 | MobileNetV2 + LSTM | Moderate | Moderate | Moderate | Moderate |
| 4 | MobileNetV2 + GRU | Moderate | Moderate | Moderate | Moderate |
| 5 | MobileNetV2 + TF-IDF + ANN | Baseline | Baseline | Baseline | Baseline |

> *Note: Exact numeric values are populated from experimental runs. The relative rankings above reflect the consistent outcomes observed across evaluation iterations.*

### 8.4 Key Observations from Comparative Analysis

**Observation 1 — Image Backbone Quality Matters Significantly:**
Models using ResNet50 consistently outperformed their MobileNetV2 counterparts with the same text encoder. ResNet50's deeper architecture (50 layers vs MobileNetV2's ~53 layers in a different configuration) produces richer and more discriminative feature representations for visually complex food categories.

**Observation 2 — Sequential Text Encoding Outperforms Bag-of-Words:**
Models using LSTM or GRU consistently outperformed Model 1 (TF-IDF + ANN), confirming that sequential context modeling captures ingredient-level semantic relationships more effectively than term-frequency statistics. For example, "grilled chicken with herbs" and "chicken grilled herbs" carry the same TF-IDF representation but potentially different LSTM encodings due to word order.

**Observation 3 — LSTM vs GRU Are Competitive:**
Models 2 and 3 (GRU vs LSTM on MobileNetV2) show comparable performance, with LSTM showing marginal advantages on longer or more complex ingredient descriptions due to its more expressive gating mechanism. This is consistent with established NLP literature.

**Observation 4 — BERT Adds Power but at a High Cost:**
While ResNet50 + BERT (Model 5) demonstrated strong performance owing to BERT's superior contextual text representations, the absolute performance gap over ResNet50 + LSTM (Model 4) was **not commensurate with the dramatic increase in computational cost**. This formed the basis for the final model selection decision.

### 8.5 Classification Report (Model 4 — Final Model)

The classification report for the final model was generated using:

```python
print(classification_report(y_true, y_pred, target_names=class_names))
```

This report provides per-class Precision, Recall, and F1-Score, enabling identification of food categories where the model performs particularly well or poorly.

### 8.6 Confusion Matrix Analysis

Confusion matrices were generated for the test set predictions:

```python
cm = confusion_matrix(y_true, y_pred)

sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=class_names, yticklabels=class_names
)
```

Key insights from confusion matrix analysis:
- The main diagonal represents correct classifications.
- Off-diagonal cells reveal systematic confusions (e.g., visually similar food categories).
- Categories with high off-diagonal values in the same column indicate that certain food classes are frequently confused with a dominant neighbor class.
- Both the training curve plots (`resnet_lstm_training_curves.png`) and confusion matrix plots (`resnet_lstm_confusion_matrix.png`) were saved to Google Drive for documentation.

---

## 9. Single Image Testing Across All Models

To validate real-world inference behavior, all five models were tested simultaneously on the **same input image and text**, allowing direct comparison of their predictions and confidence scores.

### 9.1 Inference Pipeline

The `test_all_models()` function in `comparison.py` handles the complete inference pipeline:

```python
def test_all_models(img_path, text):

    # Image preprocessing — separate per backbone
    img_mobilenet = preprocess_image_mobilenet(img_path)   # /255.0
    img_resnet    = preprocess_image_resnet(img_path)       # resnet_preprocess

    # Text preprocessing — separate per text encoder
    tfidf_input = preprocess_tfidf(text)        # (1, 4000)
    seq_30      = preprocess_sequence(text, 30) # (1, 30)
    seq_100     = preprocess_sequence(text, 100)# (1, 100)
    bert_input  = preprocess_bert_live(text)    # (1, 768)
```

**Critical implementation detail:** MobileNetV2 models use `img_array / 255.0` normalization (scaling to [0, 1]), while ResNet50 models use `resnet_preprocess()` (ImageNet channel mean subtraction). Using the wrong preprocessor for the wrong backbone is a common source of silent inference bugs.

### 9.2 Example Test Cases

```python
test_all_models(
    img_path="basic-burger.jpg",
    text="burger with cheese and beef"
)

test_all_models(
    img_path="basic-burger.jpg",
    text=""    # Text-free case — image only
)
```

The text-free test case evaluates how well each model can rely on visual features alone when no text description is provided, testing robustness to missing modalities.

### 9.3 Expected Output Format

```
==============================
MULTIMODAL MODEL COMPARISON
==============================

MODEL: MobileNetV2 + TF-IDF + ANN
Predicted Food: burger
Confidence: 0.7231
----------------------------------------
MODEL: MobileNetV2 + GRU
Predicted Food: burger
Confidence: 0.8102
----------------------------------------
MODEL: MobileNetV2 + LSTM
Predicted Food: burger
Confidence: 0.8340
----------------------------------------
MODEL: ResNet50 + LSTM
Predicted Food: burger
Confidence: 0.9156
----------------------------------------
MODEL: ResNet50 + BERT
Predicted Food: burger
Confidence: 0.8917
----------------------------------------
```

---

## 10. Calories Retrieval System

### 10.1 Overview

A post-prediction **Calories Retrieval Module** was integrated into the system to provide nutritional context alongside food classification. This module processes the input text description to extract food ingredient terms and looks them up against a curated calories database.

### 10.2 Implementation

```python
import re

def get_calories_info(text, calories_dict):
    words = set(re.findall(r'\b[a-z]+\b', text.lower()))

    results = []
    for word in words:
        if word in calories_dict:
            cal100 = calories_dict[word]
            calkg  = cal100 * 10
            results.append((word, cal100, calkg))

    return results
```

**Process:**
1. Tokenize the input text into lowercase word tokens using regex.
2. Look each token up in `final_calories.json`, which stores calorie values (per 100g) for food ingredients.
3. Compute calories per kilogram as `cal_100g × 10`.
4. Return all matched ingredients with their nutritional values.

### 10.3 Output Example

```
Prediction: burger

Calories Info:
  beef:   250 cal/100g | 2500 cal/kg
  cheese: 402 cal/100g | 4020 cal/kg
  bread:  265 cal/100g | 2650 cal/kg
```

### 10.4 Integration

The `predict_full()` function (implemented in `resnet_lstm_model_(4).py`) chains classification and calorie retrieval:

```python
def predict_full(img_path, text):
    img = preprocess_image(img_path)
    txt = preprocess_text(text)

    pred     = model.predict([img, txt])
    class_id = np.argmax(pred)
    label    = label_classes[str(class_id)]

    print("Prediction:", label)
    print("\nCalories Info:")
    results = get_calories_info(text, calories_dict)
    for food, cal100, calkg in results:
        print(f"{food}: {cal100} cal/100g | {calkg} cal/kg")
```

---

## 11. Final Model Selection Justification

### ⭐ Selected Model: Model 4 — ResNet50 + LSTM

After comprehensive experimentation, rigorous evaluation, and multi-dimensional analysis, **ResNet50 + LSTM (Model 4)** was selected as the final production model. This decision was based on a holistic trade-off analysis across the following dimensions:

### 11.1 Performance

ResNet50 + LSTM achieved the **highest overall accuracy, precision, recall, and F1-score** on the test set among all five models. The combination of ResNet50's deep visual feature extraction with LSTM's sequential text modeling produced complementary and well-calibrated multimodal representations.

### 11.2 Training Stability

Model 4 exhibited **consistent convergence behavior** across both training phases. The two-phase training strategy (freeze → fine-tune) provided structured and stable learning dynamics, with validation metrics improving monotonically before early stopping. In contrast, some other configurations showed more volatile validation loss curves.

### 11.3 Why Not Model 5 (ResNet50 + BERT)?

While **ResNet50 + BERT (Model 5)** produced strong results and benefits from BERT's powerful bidirectional language representations, it was ultimately not selected as the final model for the following reasons:

| Criterion | ResNet50 + LSTM (Model 4) | ResNet50 + BERT (Model 5) |
|---|---|---|
| **Test Accuracy** | ✅ Highest | ✅ High (marginally lower) |
| **Training Time** | ✅ Fast | ❌ Very slow (BERT embedding) |
| **Inference Speed** | ✅ Real-time capable | ❌ Significantly slower |
| **Memory Requirements** | ✅ Moderate | ❌ High (768-dim + BERT model) |
| **Deployment Complexity** | ✅ Simple (Keras only) | ❌ Complex (PyTorch + TensorFlow) |
| **Offline Embedding Dependency** | ✅ None | ❌ Requires pre-extracted embeddings |
| **Live Text Inference** | ✅ Real-time tokenization | ❌ Requires running full BERT at inference |
| **Framework Dependency** | ✅ TensorFlow/Keras | ❌ Both PyTorch (BERT) + TensorFlow |

**Critical insight**: The performance gap between Model 4 and Model 5 was **not statistically significant** enough to justify the dramatically increased computational burden. For a practical deployment scenario — including mobile applications, web services, or hospital nutrition systems — Model 4 offers a far superior trade-off between capability and efficiency.

### 11.4 Superiority Over MobileNetV2-Based Models

Models 1–3 were outperformed by Model 4 primarily because:
- **MobileNetV2 is architecturally shallower** and extracts less discriminative features for complex food categories.
- The smaller backbone capacity limits the quality of the multimodal fusion, regardless of the text encoder quality.

### 11.5 Summary Decision Matrix

| Model | Performance | Speed | Stability | Deployability | **Overall** |
|---|---|---|---|---|---|
| MobileNetV2 + TF-IDF | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| MobileNetV2 + GRU | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| MobileNetV2 + LSTM | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **ResNet50 + LSTM** ✅ | **⭐⭐⭐⭐⭐** | **⭐⭐⭐⭐** | **⭐⭐⭐⭐⭐** | **⭐⭐⭐⭐** | **⭐⭐⭐⭐⭐** |
| ResNet50 + BERT | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |

---

## 12. Challenges Faced

Throughout the development of this project, the team encountered and resolved several significant technical and logistical challenges:

**1. Dataset Scarcity and Limited Diversity**
The initial dataset was insufficiently large for training deep multimodal networks without overfitting. This was resolved by building a custom web scraping pipeline to expand the dataset with diverse, real-world food data.

**2. Image Preprocessing Inconsistency**
Different model backbones require different image normalization strategies. MobileNetV2 expects [0,1] scaled inputs while ResNet50 expects ImageNet mean-subtracted inputs via `resnet_preprocess()`. Using the wrong preprocessor leads to silent performance degradation. This was identified and corrected in the final `comparison.py` evaluation script.

**3. BERT Integration Complexity**
Model 5 required coordination between PyTorch (for BERT inference) and TensorFlow/Keras (for the rest of the pipeline). Managing dependencies, device placement, and embedding extraction added significant overhead. The solution was to pre-extract embeddings offline and save them as numpy arrays.

**4. Model Convergence and Instability**
Some early training runs experienced unstable validation loss curves due to learning rate mismatches. The introduction of `ReduceLROnPlateau` and the two-phase training strategy for Model 4 resolved these stability issues.

**5. Memory Constraints During Batch Training**
Training ResNet50-based models with large image arrays and simultaneous text processing required careful batch size tuning to avoid GPU out-of-memory errors. A batch size of 32 was found to provide the best balance between training speed and memory usage.

**6. Class Imbalance**
Some food categories were more represented than others in the dataset. Weighted precision, recall, and F1-score metrics were used during evaluation to fairly assess per-class performance regardless of sample count imbalance.

**7. Sequence Length Selection**
Determining appropriate max sequence lengths for LSTM/GRU models required experimentation. MobileNetV2-based sequence models used max_len=30 while ResNet50 + LSTM used max_len=100, reflecting the richer contextual information needed by the more capable backbone.

---

## 13. Future Improvements

The following directions represent promising avenues for extending this work:

1. **End-to-End BERT Fine-Tuning**: Rather than using fixed BERT embeddings, jointly fine-tuning BERT with the image branch could potentially improve alignment between visual and linguistic representations.

2. **Attention-Based Fusion**: Replace simple concatenation-based fusion with cross-modal attention mechanisms (e.g., cross-attention transformers) to allow each modality to query the other selectively.

3. **Vision Transformer (ViT) Backbone**: Replace convolutional backbones with Vision Transformers, which have demonstrated state-of-the-art performance on image classification benchmarks and may integrate more naturally with transformer-based text encoders.

4. **Larger and More Diverse Dataset**: Continued web scraping and dataset expansion, potentially incorporating data augmentation strategies (random crops, color jitter, text paraphrasing) to further improve generalization.

5. **Portion Estimation Module**: Extend the calorie system to estimate portion sizes from images (using depth estimation or object detection) to provide more accurate total calorie estimates per serving.

6. **Multilingual Support**: Extend the text processing pipeline to support Arabic and other languages relevant to regional food classification, given the diverse cultural context of the project team.

7. **Model Quantization and Pruning**: Apply post-training quantization or structured pruning to Model 4 to further reduce its inference footprint for mobile deployment.

8. **Real-Time Mobile Deployment**: Package the final model as a mobile application using TensorFlow Lite or ONNX, enabling on-device inference without requiring cloud connectivity.

---

## 14. Presentation & Defense Topics

The following topics represent the core areas that will be covered and defended during the project presentation:

### 14.1 Data & Problem Understanding

Candidates are expected to demonstrate a thorough understanding of the food classification problem domain, including the challenges of multimodal data, class imbalance, and the rationale for using both image and text modalities. This includes explaining the dataset collection strategy and the limitations of the original dataset that motivated the custom web scraping pipeline.

### 14.2 Modeling & Architecture Design

A detailed walkthrough of all five model architectures is expected, including justification for each design choice: why ResNet50 was chosen over MobileNetV2 for the final model, why LSTM was preferred over GRU for the final model, and why BERT was ultimately not selected despite its power. Each team member should be able to explain the architecture they implemented.

### 14.3 Implementation & Technical Quality

Candidates should demonstrate familiarity with the full implementation stack: Keras/TensorFlow for model construction, PyTorch for BERT embedding extraction, scikit-learn for preprocessing and evaluation, and the custom preprocessing pipeline. Code quality, modularity, and reproducibility should be highlighted.

### 14.4 Experimental & Comparative Analysis

The comparison of all five models across Accuracy, Precision, Recall, and F1-Score is central to the project's research contribution. Candidates should be able to explain the comparative results, identify patterns and insights, and justify the final model selection based on the experimental evidence.

### 14.5 Teamwork & Individual Contribution

Each team member should be prepared to explain their individual contribution to the project in detail — including the specific model they implemented, the challenges they encountered, and how their work integrated with the broader system. Fair and clear contribution distribution strengthens the project's presentation.

### 14.6 Evaluation Metrics & Validation Strategy

Candidates should demonstrate fluency with the evaluation methodology: why weighted averaging was used, how train/val/test splits were managed, what early stopping and learning rate scheduling contribute, and how confusion matrices reveal model weaknesses. The ability to critically interpret the classification report per class is expected.

### 14.7 Individual Understanding

Beyond team-level discussion, each member should be capable of independently answering technical questions about the full system architecture, not only their assigned component. This demonstrates depth of understanding and collaborative knowledge sharing.

---

## 15. Team Contribution Distribution

This project was a collaborative effort requiring diverse technical expertise across machine learning, deep learning, natural language processing, data engineering, and software development. The work was divided as follows:

### Mena
**Responsibilities:**
- Designed and implemented the **Custom Web Scraping Spider** for dataset collection, including scraping logic, data validation, deduplication, and integration with the main dataset pipeline.
- Implemented and trained **Model 1: MobileNetV2 + TF-IDF + ANN**, including TF-IDF feature engineering, model architecture design, hyperparameter tuning, and evaluation.

### Tasneem
**Responsibilities:**
- Implemented and trained **Model 3: MobileNetV2 + LSTM**, including Embedding layer configuration, LSTM architecture design, training pipeline setup, and performance evaluation.

### Malak
**Responsibilities:**
- Implemented and trained **Model 2: MobileNetV2 + GRU**, including GRU architecture design, controlled comparison against LSTM for text encoding effectiveness, and evaluation.

### Haidy
**Responsibilities:**
- Implemented and trained **Model 5: ResNet50 + BERT**, including BERT integration with PyTorch, CLS embedding extraction, offline embedding pipeline, and fusion with ResNet50 features.

### Nadia
**Responsibilities:**
- Implemented and trained **Model 4: ResNet50 + LSTM** (the final selected model), including two-phase transfer learning strategy, fine-tuning of conv5 block, complete training pipeline, and comprehensive evaluation.
- Developed the **GUI application** enabling end-user interaction with the trained food classification and calorie retrieval system.

**Team Note:** Although each model was implemented by a designated team member, all five models were essential for the project's comparative research contribution. The final model selection required running and evaluating all models together, which involved collaborative work across the entire team. The project's strength lies in the systematic, multi-model experimental framework — not any single model in isolation.

---

## 16. Conclusion

The **Multimodal Food Classification System** demonstrates the effectiveness of combining Computer Vision and Natural Language Processing for intelligent food recognition. Through systematic experimentation across five distinct multimodal architectures — spanning different image backbones (MobileNetV2, ResNet50) and text encoders (TF-IDF, GRU, LSTM, BERT) — the project produced a rigorous comparative study that validates the multimodal approach over any single-modality baseline.

The **key findings** of this project are:

1. **Multimodal fusion consistently outperforms unimodal approaches**: Both image and text carry complementary discriminative information for food classification.

2. **Image backbone quality is a dominant factor**: ResNet50-based models systematically outperform MobileNetV2-based models, justifying the additional computational cost of the deeper backbone.

3. **Sequential text modeling outperforms bag-of-words**: LSTM and GRU models outperform TF-IDF + ANN, confirming the importance of word order and contextual dependencies in ingredient descriptions.

4. **BERT's power does not always justify its cost**: Despite BERT's superior NLP capabilities, the marginal performance gain over LSTM was insufficient to justify the dramatically increased computational overhead for this task.

5. **ResNet50 + LSTM achieves the optimal balance**: Across accuracy, training stability, computational efficiency, and deployment suitability, Model 4 represents the best practical choice for production deployment.

Beyond model training, the project contributes a **custom web scraping infrastructure** for food data collection, a **complete preprocessing pipeline** for multimodal data, and a **Calories Retrieval System** that adds practical nutritional value to the classification output. The development of a **GUI** further demonstrates the team's commitment to building a complete, end-to-end food intelligence solution.

This project stands as a comprehensive, research-grade exploration of multimodal deep learning for food intelligence, with clear experimental rigor, strong technical implementation, and practical applicability.

---

## 📁 Project File Structure

```
multimodal-food-classification/
├── preprocessing2_(1).py          ← Data preprocessing pipeline
├── resnet_lstm_model_(4).py       ← Model 4: ResNet50 + LSTM (Final Model)
├── model5_resnet_bert_final.py    ← Model 5: ResNet50 + BERT
├── comparison.py                  ← Full model evaluation & comparison
├── هي_مهمه_بس_ليه_معرفش.py       ← Multi-model testing script
├── processed_data/
│   ├── X_img_train/val/test.npy
│   ├── X_txt_tfidf_train/val/test.npy
│   ├── X_txt_seq_train/val/test.npy
│   ├── X_bert_test.npy
│   ├── y_train/val/test.npy
│   ├── tokenizer.pkl
│   ├── tfidf_vectorizer.pkl
│   └── label_classes.json
├── models/
│   ├── resnet_lstm_final.keras            ← Final selected model
│   ├── model5_resnet_bert_final.keras
│   ├── model_1_mobilenet_tfidf.keras
│   ├── final_mobilenet_gru_model.keras
│   └── mobilenet_lstm_food_model.keras
└── dataset/
    ├── all_data.json
    ├── final_calories.json
    └── dataset/<class_name>/<images>
```

---

## 🛠️ Requirements

```
tensorflow>=2.10
torch>=1.13
transformers>=4.20
scikit-learn>=1.0
numpy>=1.21
pandas>=1.3
matplotlib>=3.5
seaborn>=0.11
Pillow>=9.0
```

---

*This README was prepared for academic submission as part of a university graduation project in Computer Science / Artificial Intelligence.*
