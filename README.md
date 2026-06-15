# Decoding Italian Music: A Text Mining Breakdown via Topic Modeling and Classification

## Authors
* **Mattia Bertolotti** (866362)
* **Gabriele Giorgio** (925678)

## Installation and Setup
### Prerequisites
* Python
* Jupyter Notebook

### Required Libraries
To run the notebooks in a local setup, simply install the dependencies from the requirements file:
```bash
pip install -r requirements.txt
```

To manage the heavy computational requirements of the project, the entire experimental framework was executed via the Windows Subsystem for Linux (WSL). Operating within a dedicated [rapids25.06_python3.12](https://rapids.ai/) environment allowed us to seamlessly interface with the CUDA architecture, thereby fully exploiting the NVIDIA GPU for hardware acceleration.

## Project Overview
This project investigates the linguistic and thematic patterns within over 20,000 Italian song lyrics. We tackled the complex challenge of lemmatizing Italian lyrics, used topic modeling algorithms to extract core themes, and applied text classification to verify if the artists' lexical choices actually align with user-assigned musical genres. Finally, we analyzed how Italian artists approach and express the historically central theme of love.

For a detailed explanation of the methodology, results, and conclusions, please refer to the project documentation:
* **[Project Report (PDF)](report/Text_mining_report_Bertolotti_Giorgio.pdf)**
* **[Presentation Slides (PDF)](report/Text_mining_presentation_Bertolotti_Giorgio.pdf)**

## Methodology

### 1. Dataset Creation
Notebook: `1_dataset_creation.ipynb`

The data collection strategy combined several APIs to build a comprehensive corpus:
* **Spotify API:** Implemented a snowball sampling approach, starting from the artist 'Calcutta', to retrieve an initial network of relevant artists.
* **Last.fm API:** Retrieved the top 5 user-voted genres, total listeners, and playcounts for each artist to establish target labels.
* **Genius API:** Extracted the raw lyrics for the 20 most popular tracks per artist. The extraction pipeline included an API key rotation system and checkpointing mechanisms to bypass strict rate limits.

### 2. Text Pre-Processing & Representation
Notebook: `2_text_preprocessing.ipynb`

* **Language Detection:** Implemented a granular, line-by-line language detection strategy using the `Lingua` library to handle the phenomenon of "code-switching" (mixing multiple languages in a single song) effectively.
* **Tokenization & Lemmatization:** Transitioned from heuristic models to Stanford's `Stanza` neural pipeline. This computationally intensive step was necessary to achieve the morphological precision required to correctly resolve ambiguities in conjugated Italian verbs.
* **Grammatical Pruning:** Utilized Stanza's POS tagger as an advanced stop-word remover, retaining only tokens with high semantic weight: Nouns, Verbs, Adjectives, and Adverbs.
* **Vectorization:** Represented the textual corpus mathematically using Bag-of-Words (BoW) and TF-IDF models. The TF-IDF weight captures the global rarity of terms across the corpus.

### 3. Topic Modeling
Notebook: `3_topic_modeling.ipynb`

To understand the main themes in the Italian music scene, we compared traditional and neural models:
* **Traditional Models:** Evaluated LDA, NMF, and LSA via Grid Search. We selected **NMF (K=6)** as the best linear baseline because it provided the optimal trade-off between semantic coherence and vocabulary diversity.
* **Neural VAE:** Built a custom Variational Autoencoder using PyTorch, fixing the latent space to exactly 6 topics to directly compare it against NMF. The Neural VAE demonstrated vastly superior semantic depth, successfully isolating specific cultural tropes.

### 4. Text Classification
Notebook: `4_text_calssification.ipynb`

Because music is intrinsically fluid, we treated genre prediction as a **multi-label classification** task. We mapped over 1,110 noisy user tags into 8 standardized macro-genres using a custom rule-based heuristic dictionary. 
We benchmarked distinct architectures:
* **SVM via One-vs-Rest:** Proved to be the Best Global Model, achieving the highest Macro F1-Score (0.50). Its success was driven by superior robustness and resilience on difficult, minority genres like Electronic and Jazz.
* **Classifier Chains:** Evaluated linking binary SVM classifiers and Extreme Gradient Boosting (XGBoost) trees to actively learn label correlations and contextual dependencies.
* **Transformer BERT (UmBERTo):** Fine-tuned a RoBERTa-based model trained on Italian web data. We engineered a custom "Head+Tail" tokenization strategy (extracting the first and last 150 words) to bypass Transformer memory limits without losing narrative-rich segments. While UmBERTo dominated highly populated mainstream genres, it produced more false positives overall.

#### Sentiment Analysis
We investigated the underlying emotional valence of lyrics by feeding tracks clustered under the "Love" (Amore) topic into `FEEL-IT`, an advanced Transformer fine-tuned for Italian emotion detection.
* **A Melancholic Majority:** 64.5% of tracks exhibit a Negative sentiment.
* **The Joyful Minority:** Only 35.5% display a Positive sentiment.
The analysis illustrates a strong artistic tendency in the Italian music scene to utilize music as an emotional outlet for the suffering and complexities associated with romance.

## Reproducibility
To ensure full transparency and research reproducibility, the entire computational framework has been made available in this repository. The codebase is organized into interactive notebooks, where each logical block is accompanied by explanatory comments.

---
*Project developed for the Text Mining and Search course (A.Y. 2025/2026), CdLM in Theory and Technology of Communication, Università degli Studi di Milano-Bicocca.*
