# Semantic Textual Similarity (STS) Project

## Overview

This project explores various approaches to Semantic Textual Similarity (STS), the task of quantifying the degree to which two text snippets share the same meaning. Given a pair of sentences, the models assign a similarity score on a continuous scale from 0 (no semantic overlap) to 1 (full equivalence). The performance of each model is evaluated by calculating the Pearson's correlation between its similarity scores and human judgments.

## Team

* Abhinav Reddy Boddu 
* Rohit Gowlapalli 
* Sasidhar Chavali

## Problem Statement

Semantic Textual Similarity (STS) aims to quantify the semantic equivalence between two text snippets. The goal is to develop a model that can assign a similarity score between 0 and 1 to a given pair of sentences, reflecting their semantic overlap. Model performance is assessed by correlating its scores with human-annotated similarity judgments.

## Datasets Used

Initially, the project considered four datasets, but ultimately focused on two due to issues with binary classification datasets biasing continuous score learning. The final merged dataset was split into 70% training, 10% validation, and 20% testing.

1.  **STS Benchmark**: This dataset comprises 8,628 sentence pairs across news, image captions, and forum discussions. It was curated from English datasets used in SemEval STS tasks (2012-2017). A key strength is its human-annotated similarity scores, ensuring high-quality labels.

2.  **SICK Dataset**: The Sentences Involving Compositional Knowledge (SICK) dataset is designed for compositional distributional semantics. It features numerous sentence pairs with diverse lexical, syntactic, and semantic features. Each pair is annotated with relatedness and entailment labels. Original relatedness scores (1 to 5) were normalized to the [0, 1] range for this task.

You can find all relevant datasets and data releases on the project's [GitHub Releases page](https://github.com/Abhinavreddy-B/INLP-Project/releases).

## Methodology

The project explored a range of methods, from statistical techniques to advanced neural networks:

### N-gram Based Statistical Methods

These methods compare text based on overlapping words or n-grams. Synsets (cognitive synonyms) for tokens in each sentence are retrieved and compared, with semantic alignments made if synset similarity exceeds a threshold (0.3). An example highlights how different words can be aligned semantically (e.g., "began" and "started"). The similarity score is calculated using a formula involving token counts and alignment scores. Pearson's correlation is then calculated.

**Results (Pearson's Correlation)**:
* n=1: 0.655
* n=2: 0.661
* n=3: 0.661
* n=4: 0.661
* n=5: 0.661

Performance slightly improves from unigrams to bigrams, suggesting capture of more semantic information. A plateau is observed from n=2 onwards, indicating diminishing returns for higher-order n-grams due to data sparsity and the nature of meaningful phrases. Optimal balance is achieved at n=2.

### Neural Network Models

#### CNN (Convolutional Neural Network)

* **Preprocessing & Feature Engineering**: Sentences are cleaned (punctuation removed, lowercased) and tokenized using NLTK. Words are mapped to pre-trained GloVe embeddings, with unknown words assigned zero vectors. Sentences are padded to 30 tokens. Handcrafted features, such as word appearance flags, numeric match flags, and one-hot encoded POS tags (derived using NLTK), are appended to embeddings.
* **CNN Architecture**: A one-layer 1D CNN with 300 filters (filter length matching word vector dimension) generates sentence embeddings. ReLU activation and max pooling are applied. Early stopping is used to prevent overfitting.
* **Semantic Similarity Scoring**: A semantic difference vector is created by concatenating the element-wise absolute difference and Hadamard product of sentence embeddings.
* **Fully Connected Neural Network (FCNN)**: The 600-dimension difference vector is fed into a two-layer FCNN. The first layer has 300 units with tanh activation, and the second has 6 output units with softmax activation for probability distribution over similarity labels.

**Results (Pearson's Correlation)**: 0.7423

#### RNN (Recurrent Neural Network)

* **Preprocessing & Feature Engineering**: Similar to CNN, including NLTK tokenization, GloVe embeddings (Common Crawl, 840B tokens), zero vectors for unknown words, padding to 30 tokens, and appending handcrafted features (word/numeric match, POS tags).
* **RNN Architecture**: An RNN encodes sequential information, processing words in sequence to produce context-aware embeddings. The final hidden state serves as the sentence representation.
* **Semantic Similarity Scoring**: Element-wise addition and multiplication of sentence embeddings are performed, then concatenated to form a combined vector capturing shared and contrasting features.
* **FCNN**: The combined vector is passed to a two-layer FCNN, with 300 tanh units in the first layer and 6 softmax output units for similarity label prediction.

**Results (Pearson's Correlation)**: 0.6666

#### LSTM (Long Short-Term Memory)

* **Preprocessing & Feature Engineering**: Identical to CNN and RNN setups, involving cleaning, NLTK tokenization, GloVe embeddings (Common Crawl, 840B tokens) with zero vectors for OOV words, padding to 30 tokens, and appending handcrafted features.
* **LSTM Architecture**: An LSTM replaces the vanilla RNN to capture long-range dependencies and contextual meaning. It processes sentences word by word, outputting a final hidden state as a dense, context-rich embedding. Its memory cell helps retain relevant information.
* **Semantic Similarity Scoring**: Element-wise addition and multiplication of LSTM-generated embeddings are performed and concatenated to form the semantic difference vector.
* **FCNN**: The semantic difference vector is fed into a two-layer FCNN, with 300 tanh units in the first layer and 6 softmax output units for similarity score prediction.

**Results (Pearson's Correlation)**: 0.7113

#### Siamese Network - BiLSTM

* **Word Embeddings**: Uses pre-trained Google News Word2Vec (300-dimensional). A vocabulary is built from training sentences, mapping words to indices, with an "unk" token for out-of-vocabulary words.
* **Model Architecture**: Consists of two identical BiLSTM branches (Siamese structure) with an attention mechanism.
    * **Embedding Layer**: Initialized with frozen pre-trained Word2Vec embeddings, mapping input word indices to 300-dimensional vectors.
    * **Bidirectional LSTM (BiLSTM)**: Hidden Size: 50, Layers: 2. Processes sequences bidirectionally to capture contextual dependencies.
    * **Attention Mechanism**: Computes attention weights using a feed-forward layer followed by softmax, producing a weighted sum of BiLSTM outputs to emphasize relevant words.
    * **Similarity Scoring**: Concatenates the final representations of both sentences and passes them through a fully connected layer with sigmoid activation to produce a similarity score in [0, 1].

#### Doc2Vec

* **Model Architecture**: Utilizes Doc2Vec (Paragraph Vector) for sentence embeddings.
    * **Normalized Cosine Similarity**: $Score = (1 - cosine(v_1, v_2) + 1) \times 2.5$, where $v_1, v_2$ are Doc2Vec embeddings.
    * **BiLSTM Regression**: A bidirectional LSTM processes concatenated embeddings ($v_1 \oplus v_2$) to predict similarity, trained with MSE loss.
* **Key Steps**:
    * **Embedding Generation**: Sentences are tokenized and converted to embeddings using Doc2Vec (vector size = 25, window = 6). Embeddings are inferred via `model.infer_vector()`.
    * **Training**: BiLSTM trained for 10 epochs (batch size = 10, LR = 0.001).

**Results (Pearson's Correlation)**:
* Train: 0.82
* Val/Test: ~0.40
* Overall: 0.47

#### BERT

* **Model Architecture**: Leverages BERT (Bidirectional Encoder Representations from Transformers) fine-tuned for sequence classification to predict continuous similarity scores.
    * **BERT-Base-Uncased**: A pre-trained transformer model with 12 layers, 768 hidden dimensions, and 110M parameters.
    * **Fine-Tuning Setup**:
        * **Input Processing**: Sentences are tokenized and truncated/padded to a maximum length (95th percentile of training sentence lengths).
        * **Classification Head**: A regression layer on top of BERT outputs a single similarity score.
        * **Loss Function**: Mean Squared Error (MSE) for regression.
        * **Optimizer**: Adam with a learning rate of 1e-5 and betas (0.5, 0.99).

**Results (Pearson's Correlation)**: 0.7903

### Knowledge Distillation

This approach uses a "teacher" model to guide a smaller "student" model, balancing hard and soft loss.

* **Teacher Model**: A large pre-trained BERT model fine-tuned on the similarity task, providing rich contextual representations but being computationally heavy.
* **Student Model**: `microsoft/MiniLM-L12-H384-uncased` with a sequence classification head (single regression output). MiniLM is a distilled model designed to be lightweight and fast, retaining much of the teacher's power, with 12 layers and 384 hidden size.
* **Loss Function**: A custom distillation loss that balances:
    * **Hard loss**: MSE between student predictions and ground-truth labels.
    * **Soft loss**: MSE between student predictions and soft labels (teacher predictions).
    * The combined loss: $Loss = \alpha \times hardLoss + (1 - \alpha) \times softLoss$, with $\alpha = 0.5$ (found to give best results).

**Why Distillation Outperforms BERT**:
* **Regularization**: It acts as regularization, allowing the student to learn richer relational knowledge from the teacher's probability distributions, leading to a smoother loss landscape and better generalization.
* **MiniLM Design**: MiniLM's lightweight architecture captures attention patterns of larger models, making it efficient for tasks like STS that don't require deep linguistic reasoning.
* **Mitigating Teacher Bias**: Distillation helps mitigate potential noise or overfitting from the teacher, as the student focuses on relevant patterns, learning a cleaner representation.

**Results (Pearson's Correlation)**: 0.8943

## Results Summary

| Method                 | Pearson's Correlation (Test) |
| :--------------------- | :--------------------------- |
| Doc2Vec                | 0.4017                       |
| RNN                    | 0.6666                       |
| LSTM                   | 0.7113                       |
| CNN                    | 0.7423                       |
| BERT                   | 0.7903                       |
| Knowledge Distillation | 0.8943                       |

*(Refer to the "Results" section in the original report for detailed training/validation losses and additional metrics like Spearman and MAE/MSE by method.)*

## Conclusion

* **Neural Networks vs. Statistical Methods**: Neural architectures (RNN, LSTM, CNN) significantly outperform classical N-Gram and Doc2Vec methods, demonstrating superior ability to learn contextual and sequential relationships.
* **LSTM**: Shows high capacity for context modeling with the highest training correlation, but a slight test performance drop suggests potential overfitting.
* **CNN**: Exhibits robust generalization, surpassing RNN and LSTM on the test set due to localized pattern recognition.
* **BERT**: Provides strong performance from deep contextual understanding, capturing fine-grained semantic relationships but is computationally intensive.
* **Knowledge Distillation (MiniLM)**: Achieves the highest Pearson correlation on test data, demonstrating excellent generalization despite its reduced size. The dual-loss strategy enables the student model to approximate BERT's behavior while learning from data-driven corrections, resulting in an efficient and highly accurate model.

**Summary**: Knowledge Distillation strikes the optimal balance, delivering near BERT-level accuracy with faster inference and lower memory usage, making it the most practical model for real-world STS deployment.
