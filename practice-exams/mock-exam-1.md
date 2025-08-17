# NVIDIA Certified Associate Mock Exam 1
## Generative AI with LLMs (NCA-GENL)

**Instructions:**
- 50 questions, 60 minutes
- Multiple choice (A, B, C, D)
- Mark answers clearly
- Review marked questions at end

---

### Question 1
What is the primary purpose of positional encoding in transformer models?
A) To reduce computational complexity
B) To provide information about token order in sequences
C) To compress input representations
D) To handle variable-length sequences

**Correct Answer:** B
**Explanation:** Transformers process all tokens in parallel (unlike RNNs which process sequentially), so they lack inherent knowledge of token order. Positional encoding adds information about each token's position in the sequence, enabling the model to understand relationships based on position.

---

### Question 2
In the self-attention mechanism, which matrix represents the token seeking relevant information?
A) Key (K)
B) Value (V)
C) Query (Q)
D) Output (O)

**Correct Answer:** C
**Explanation:** In attention mechanism, Query (Q) represents the token that is "asking" for information. Key (K) is used to compute compatibility with other tokens, and Value (V) contains the actual information to be retrieved. Think of it as Q="what am I looking for?", K="what do I have?", V="the actual content."

---

### Question 3
When comparing two LLMs using BLEU score, a higher BLEU indicates:
A) The model has a broader vocabulary
B) The model understands complex sentence structures better
C) The model produces text more similar to reference text
D) The model has higher computational efficiency

**Correct Answer:** C
**Explanation:** BLEU (Bilingual Evaluation Understudy) measures n-gram overlap between generated text and reference text. Higher BLEU specifically indicates greater lexical similarity to reference, NOT broader vocabulary or deeper understanding. It's a surface-level similarity metric.

---

### Question 4
What is the main benefit of model quantization?
A) Increasing model accuracy
B) Reducing memory usage and inference latency
C) Improving training stability
D) Enhancing model interpretability

**Correct Answer:** B
**Explanation:** Quantization reduces numerical precision (e.g., from FP32 to INT8), which significantly reduces memory footprint and speeds up inference. This comes with a small trade-off in accuracy but massive gains in efficiency and cost reduction for deployment.

---

### Question 5
Which NVIDIA tool is specifically designed for high-performance model inference serving?
A) NeMo Framework
B) CUDA Toolkit
C) Triton Inference Server
D) TensorRT

**Correct Answer:** C
**Explanation:** Triton Inference Server is NVIDIA's dedicated platform for deploying AI models at scale in production. It handles multiple frameworks, dynamic batching, model versioning, and monitoring. TensorRT optimizes models, NeMo is for development, CUDA is low-level programming.

---

### Question 6
In A/B testing for LLM deployment, what is the recommended traffic distribution approach?
A) Send all traffic to the new model immediately
B) Deploy both models and serve traffic equally
C) Use only internal testing
D) Gradually increase traffic over months

**Correct Answer:** B
**Explanation:** Equal (50/50) traffic distribution ensures fair comparison between models and enables statistically significant performance measurement. This approach minimizes bias and provides reliable data for decision-making about which model to fully deploy.

---

### Question 7
What is the primary advantage of transfer learning in LLM development?
A) Eliminates need for labeled data
B) Reduces training time and computational requirements
C) Guarantees better performance
D) Removes all biases from models

**Correct Answer:** B
**Explanation:** Transfer learning leverages knowledge learned from large-scale pre-training, dramatically reducing the time and computational resources needed for task-specific fine-tuning. It doesn't eliminate the need for labeled data entirely or guarantee performance, but makes training much more efficient.

---

### Question 8
Which evaluation metric is most commonly used for text summarization tasks?
A) BLEU
B) ROUGE
C) Perplexity
D) F1-Score

**Correct Answer:** B
**Explanation:** ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is specifically designed for evaluating summarization quality. It measures overlap between generated and reference summaries. BLEU is for translation, perplexity for language modeling, F1 for classification.

---

### Question 9
What does masked multi-head attention in BERT enable?
A) Faster training speed
B) Bidirectional context understanding
C) Reduced memory usage
D) Better long sequence handling

**Correct Answer:** B
**Explanation:** BERT's masked attention allows tokens to attend to all other tokens in both directions (past and future), unlike GPT which only attends to previous tokens. This bidirectional context enables better understanding tasks but prevents autoregressive generation.

---

### Question 10
Why is Exploratory Data Analysis (EDA) essential when fine-tuning LLMs?
A) To increase model parameters
B) To uncover patterns and anomalies in the dataset
C) To reduce training time
D) To eliminate pre-training requirements

**Correct Answer:** B
**Explanation:** EDA helps understand data distribution, identify biases, assess data quality, and detect anomalies before fine-tuning. This analysis is crucial for determining appropriate preprocessing steps and ensuring successful adaptation to the target domain.

---

### Question 11
What is the main purpose of layer normalization in transformers?
A) To reduce overfitting
B) To stabilize training and enable deeper networks
C) To compress model size
D) To speed up inference

**Correct Answer:** B
**Explanation:** Layer normalization normalizes inputs to each layer, reducing internal covariate shift and stabilizing gradients. This enables training of very deep networks (like transformers with many layers) and often leads to faster convergence.

---

### Question 12
Which architecture type is GPT based on?
A) Encoder-only transformer
B) Decoder-only transformer
C) Encoder-decoder transformer
D) Recurrent neural network

**Correct Answer:** B
**Explanation:** GPT uses only the decoder part of the transformer, with masked self-attention that prevents looking at future tokens. This autoregressive design enables text generation by predicting the next token based on previous tokens.

---

### Question 13
What is the key difference between BERT and GPT training objectives?
A) BERT uses autoregressive modeling, GPT uses masked language modeling
B) BERT uses masked language modeling, GPT uses autoregressive modeling
C) Both use the same training objective
D) BERT uses supervised learning, GPT uses unsupervised learning

**Correct Answer:** B
**Explanation:** BERT masks random tokens and predicts them using bidirectional context (Masked Language Modeling). GPT predicts the next token given previous tokens (autoregressive). This difference makes BERT better for understanding tasks, GPT better for generation.

---

### Question 14
In Named Entity Recognition (NER), what type of information is typically extracted?
A) Sentiment polarity
B) Topic categories
C) Person, organization, location names
D) Grammar structures

**Correct Answer:** C
**Explanation:** NER identifies and classifies named entities in text into predefined categories like PERSON (John Smith), ORGANIZATION (Google), LOCATION (New York), DATE, etc. It's a key information extraction task used in search, content analysis, and privacy protection.

---

### Question 15
What is the purpose of the attention mask in transformer models?
A) To hide irrelevant tokens during computation
B) To reduce computational complexity
C) To prevent attention to padding tokens or future tokens
D) To compress attention weights

**Correct Answer:** C
**Explanation:** Attention masks ensure tokens don't attend to padding tokens (in batched sequences) or future tokens (in autoregressive models like GPT). This maintains the intended information flow and prevents the model from "cheating" by seeing future information during training.

---

### Question 16
Which NVIDIA framework provides pre-trained models for NLP tasks?
A) Triton
B) TensorRT
C) NeMo
D) CUDA

**Correct Answer:** C
**Explanation:** NeMo (Neural Modules) is NVIDIA's framework for building AI applications, providing pre-trained models for NLP, speech, and vision tasks with easy customization and fine-tuning capabilities. Triton serves models, TensorRT optimizes them, CUDA is low-level programming.

---

### Question 17
What is Retrieval-Augmented Generation (RAG)?
A) A training technique for transformers
B) A method combining retrieval and generation for responses
C) A model compression technique
D) An evaluation metric for language models

**Correct Answer:** B
**Explanation:** RAG combines information retrieval with text generation. It first retrieves relevant documents from a knowledge base, then uses that context to generate more accurate, up-to-date responses. This reduces hallucination and enables access to external knowledge beyond training data.

---

### Question 18
In multi-head attention, why are multiple attention heads used?
A) To reduce computational cost
B) To focus on different types of relationships simultaneously
C) To handle longer sequences
D) To improve memory efficiency

**Correct Answer:** B
**Explanation:** Multiple attention heads allow the model to attend to different types of relationships simultaneously - one head might focus on syntactic relationships, another on semantic similarity, etc. Each head learns different attention patterns, providing richer representations.

---

### Question 19
What is the typical range of BLEU scores?
A) -1 to 1
B) 0 to 100
C) 0 to 1
D) 1 to 10

**Correct Answer:** C
**Explanation:** BLEU scores range from 0 to 1, where 0 means no overlap with reference text and 1 means perfect match. Scores are often reported as percentages (0-100), but the underlying scale is 0-1. Higher scores indicate better similarity to reference text.

---

### Question 20
Which technique is most effective for reducing LLM inference costs in production?
A) Increasing batch size only
B) Using larger models
C) Model quantization and optimization
D) Adding more GPUs

**Correct Answer:** C
**Explanation:** Model quantization and optimization techniques (like TensorRT) reduce memory usage and computational requirements, directly lowering infrastructure costs. While batch size helps throughput, and more GPUs increase capacity, optimization reduces the fundamental resource requirements per inference.

---

### Question 21
What is the main advantage of Parameter-Efficient Fine-tuning (PEFT) methods like LoRA?
A) Better accuracy than full fine-tuning
B) Reduces computational requirements while maintaining performance
C) Eliminates need for pre-trained models
D) Faster inference speed

**Correct Answer:** B
**Explanation:** PEFT methods like LoRA (Low-Rank Adaptation) fine-tune only a small subset of parameters, dramatically reducing memory and computational requirements while achieving performance comparable to full fine-tuning. This makes fine-tuning accessible with limited resources.

---

### Question 22
In transformer architecture, what connects different layers?
A) Attention mechanisms only
B) Feed-forward networks only
C) Residual connections
D) Pooling layers

**Correct Answer:** C
**Explanation:** Residual connections (skip connections) add the input of each sub-layer to its output, enabling gradient flow in deep networks and preventing vanishing gradients. They're combined with layer normalization in the "Add & Norm" operations throughout the transformer.

---

### Question 23
What is the purpose of the softmax function in attention computation?
A) To normalize attention weights to sum to 1
B) To increase computational speed
C) To reduce memory usage
D) To handle negative values

**Correct Answer:** A
**Explanation:** Softmax converts raw attention scores into a probability distribution where all weights sum to 1. This ensures that attention is properly distributed across tokens and creates a weighted average when applied to values. The normalization is essential for stable training.

---

### Question 24
Which model format is supported by TensorRT for optimization?
A) Only PyTorch models
B) Only TensorFlow models
C) Multiple formats including ONNX
D) Only custom NVIDIA formats

**Correct Answer:** C
**Explanation:** TensorRT supports multiple model formats including ONNX (Open Neural Network Exchange), TensorFlow, PyTorch, and others. ONNX is particularly important as it enables model portability across different frameworks and deployment platforms.

---

### Question 25
What is the key benefit of dynamic batching in Triton Inference Server?
A) Reduces model size
B) Optimizes throughput by combining requests
C) Improves model accuracy
D) Reduces latency for all requests

**Correct Answer:** B
**Explanation:** Dynamic batching automatically combines multiple incoming requests into batches to maximize GPU utilization and overall throughput. While individual request latency might increase slightly, the system can handle many more requests per second.

---

### Question 26
In text classification tasks, what does F1-score measure?
A) Only precision
B) Only recall
C) Harmonic mean of precision and recall
D) Arithmetic mean of precision and recall

**Correct Answer:** C
**Explanation:** F1-score is the harmonic mean of precision and recall: F1 = 2 × (precision × recall) / (precision + recall). It provides a single metric that balances both precision (correct positive predictions) and recall (finding all positive cases), especially useful for imbalanced datasets.

---

### Question 27
What is the main challenge that positional encoding solves in transformers?
A) Handling variable sequence lengths
B) Lack of inherent sequential order in parallel processing
C) Memory limitations
D) Computational complexity

**Correct Answer:** B
**Explanation:** Unlike RNNs that process tokens sequentially, transformers process all tokens simultaneously in parallel. This parallel processing loses information about token order, which is crucial for language understanding. Positional encoding restores this positional information.

---

### Question 28
Which training paradigm involves learning from massive unlabeled text corpora?
A) Supervised learning
B) Pre-training
C) Fine-tuning
D) Transfer learning

**Correct Answer:** B
**Explanation:** Pre-training involves learning general language representations from large-scale unlabeled text (like web pages, books) using self-supervised objectives like predicting masked tokens or next tokens. This creates foundation models that can then be fine-tuned for specific tasks.

---

### Question 29
What is the purpose of teacher forcing in sequence generation?
A) To speed up inference
B) To use ground truth tokens during training
C) To reduce model size
D) To improve model interpretability

**Correct Answer:** B
**Explanation:** Teacher forcing feeds the ground truth token at each time step during training instead of the model's own prediction. This stabilizes training and speeds convergence, though it can create a mismatch between training and inference (exposure bias).

---

### Question 30
In question answering tasks, what distinguishes extractive from generative approaches?
A) Model architecture used
B) Whether answers are extracted from text or generated
C) Training data requirements
D) Computational complexity

**Correct Answer:** B
**Explanation:** Extractive QA finds and extracts the answer span from the given passage (like highlighting text). Generative QA creates new text as the answer, potentially combining information from multiple sources or paraphrasing. Both can use similar architectures but have different output strategies.

---

### Question 31
What is the main advantage of using pre-trained embeddings?
A) Faster training speed only
B) Capturing semantic relationships learned from large corpora
C) Reduced memory usage
D) Better interpretability

**Correct Answer:** B
**Explanation:** Pre-trained embeddings (like Word2Vec, GloVe) capture semantic relationships between words learned from large text corpora. Words with similar meanings have similar embeddings, providing rich representations that improve downstream task performance, especially with limited training data.

---

### Question 32
In the context of LLMs, what does "few-shot learning" refer to?
A) Training with very few parameters
B) Learning tasks with minimal examples
C) Fast inference speed
D) Short sequence processing

**Correct Answer:** B
**Explanation:** Few-shot learning enables LLMs to perform new tasks by providing just a few examples in the input prompt, without additional training or fine-tuning. For example, showing 2-3 examples of sentiment classification in the prompt to classify new text.

---

### Question 33
What is the role of the feed-forward network in each transformer layer?
A) To compute attention weights
B) To apply non-linear transformations to representations
C) To normalize layer outputs
D) To handle positional information

**Correct Answer:** B
**Explanation:** The feed-forward network applies non-linear transformations (typically two linear layers with ReLU activation) to each position independently. This adds modeling capacity and non-linearity to the otherwise linear attention operations, enabling complex pattern learning.

---

### Question 34
Which metric is most appropriate for evaluating machine translation quality?
A) ROUGE
B) F1-score
C) BLEU
D) Perplexity

**Correct Answer:** C
**Explanation:** BLEU (Bilingual Evaluation Understudy) was specifically designed for machine translation evaluation. It measures n-gram precision between translated text and reference translations. ROUGE is for summarization, F1 for classification, perplexity for language modeling.

---

### Question 35
What is the purpose of beam search in sequence generation?
A) To speed up generation
B) To explore multiple candidate sequences simultaneously
C) To reduce memory usage
D) To handle longer sequences

**Correct Answer:** B
**Explanation:** Beam search maintains multiple candidate sequences (beams) during generation, exploring different paths to find higher-quality outputs. It balances between greedy search (fast but potentially suboptimal) and exhaustive search (optimal but computationally prohibitive).

---

### Question 36
In distributed training, what is the purpose of gradient accumulation?
A) To speed up training
B) To simulate larger batch sizes with limited memory
C) To improve model accuracy
D) To reduce communication overhead

**Correct Answer:** B
**Explanation:** Gradient accumulation allows training with larger effective batch sizes when memory is limited. Instead of updating parameters after each small batch, gradients are accumulated over multiple mini-batches before the parameter update, simulating a larger batch.

---

### Question 37
What is the main difference between autoregressive and autoencoding language models?
A) Model size
B) Training data requirements
C) Generation vs. understanding focus
D) Computational complexity

**Correct Answer:** C
**Explanation:** Autoregressive models (like GPT) predict the next token sequentially, making them excellent for generation tasks. Autoencoding models (like BERT) use bidirectional context to understand masked tokens, making them better for understanding tasks like classification and question answering.

---

### Question 38
Which NVIDIA technology is specifically designed for inference optimization?
A) NeMo
B) TensorRT
C) DGX
D) CUDA

**Correct Answer:** B
**Explanation:** TensorRT is NVIDIA's inference optimization library that accelerates deep learning models through techniques like layer fusion, precision calibration (INT8/FP16), and kernel auto-tuning. It specifically focuses on making inference faster and more efficient.

---

### Question 39
What is the purpose of dropout in neural networks?
A) To speed up training
B) To prevent overfitting by randomly setting neurons to zero
C) To reduce model size
D) To improve interpretability

**Correct Answer:** B
**Explanation:** Dropout randomly sets a percentage of neurons to zero during training, preventing the model from becoming too dependent on specific neurons. This regularization technique reduces overfitting and improves generalization to unseen data.

---

### Question 40
In the context of attention mechanisms, what does "scaled" refer to in scaled dot-product attention?
A) Scaling by sequence length
B) Scaling by square root of key dimension
C) Scaling by number of heads
D) Scaling by batch size

**Correct Answer:** B
**Explanation:** The attention scores (QKᵀ) are divided by √d_k (square root of the key dimension) to prevent extremely large values that would cause the softmax to have very small gradients. This scaling ensures stable training.

---

### Question 41
What is the main advantage of using ONNX format for models?
A) Better accuracy
B) Faster training
C) Model portability across frameworks
D) Reduced model size

**Correct Answer:** C
**Explanation:** ONNX (Open Neural Network Exchange) is a standard format that enables models trained in one framework (PyTorch, TensorFlow, etc.) to be deployed in another framework or runtime. This portability simplifies deployment and enables optimization tools like TensorRT.

---

### Question 42
In fine-tuning scenarios, what is catastrophic forgetting?
A) Model losing pre-trained knowledge during fine-tuning
B) Model forgetting current task
C) Memory overflow during training
D) Gradient explosion problem

**Correct Answer:** A
**Explanation:** Catastrophic forgetting occurs when fine-tuning on new tasks causes the model to lose previously learned knowledge from pre-training or other tasks. This is why techniques like parameter-efficient fine-tuning (PEFT) and careful learning rate scheduling are important.

---

### Question 43
What is the purpose of nucleus sampling (top-p) in text generation?
A) To speed up generation
B) To control randomness by sampling from top probability mass
C) To reduce memory usage
D) To handle longer sequences

**Correct Answer:** B
**Explanation:** Nucleus sampling (top-p) samples from the smallest set of tokens whose cumulative probability exceeds p (e.g., 0.9). This dynamically adjusts the candidate pool size based on the probability distribution, balancing diversity and quality better than fixed top-k sampling.

---

### Question 44
Which component of BERT is responsible for handling different sentence pairs?
A) Multi-head attention
B) Positional encoding
C) Segment embeddings
D) Layer normalization

**Correct Answer:** C
**Explanation:** Segment embeddings (also called token type embeddings) distinguish between different sentences in tasks like Next Sentence Prediction. They add learned embeddings to indicate whether each token belongs to sentence A or sentence B in sentence pair tasks.

---

### Question 45
What is the main purpose of knowledge distillation in model compression?
A) To transfer knowledge from larger to smaller models
B) To improve training speed
C) To reduce inference latency
D) To handle longer sequences

**Correct Answer:** A
**Explanation:** Knowledge distillation trains a smaller "student" model to mimic the behavior of a larger "teacher" model by learning from the teacher's soft outputs (probability distributions) rather than just hard labels. This creates compact models with better performance than training from scratch.

---

### Question 46
In the context of LLMs, what does "hallucination" refer to?
A) Model generating factually incorrect information
B) Model processing images
C) Model compression artifacts
D) Training instability

**Correct Answer:** A
**Explanation:** Hallucination occurs when LLMs generate plausible-sounding but factually incorrect information. This happens because models learn patterns from training data but may not truly "understand" facts, leading to confident-sounding but false statements.

---

### Question 47
What is the key innovation of the Transformer architecture over RNNs?
A) Better memory efficiency
B) Parallel processing capability
C) Smaller model size
D) Faster inference only

**Correct Answer:** B
**Explanation:** Transformers can process all tokens in a sequence simultaneously (parallel processing), unlike RNNs which must process tokens sequentially. This parallelization enables much faster training and better utilization of modern hardware like GPUs.

---

### Question 48
Which technique is commonly used to handle out-of-vocabulary words in modern NLP?
A) Word-level tokenization only
B) Subword tokenization (BPE, WordPiece)
C) Character-level processing only
D) Dictionary lookup

**Correct Answer:** B
**Explanation:** Subword tokenization (like BPE - Byte Pair Encoding, WordPiece) breaks words into smaller subword units, allowing models to handle unseen words by combining known subword pieces. This reduces vocabulary size while maintaining semantic meaning.

---

### Question 49
What is the purpose of warm-up in learning rate scheduling?
A) To speed up training
B) To gradually increase learning rate at training start
C) To reduce memory usage
D) To prevent overfitting

**Correct Answer:** B
**Explanation:** Learning rate warm-up gradually increases the learning rate from a small value to the target value over the first few epochs/steps. This helps stabilize training of large models by preventing large gradient updates when parameters are randomly initialized.

---

### Question 50
In production LLM deployment, what is the most critical consideration for cost optimization?
A) Model accuracy only
B) Balancing performance, latency, and computational costs
C) Using the largest available model
D) Maximum throughput regardless of cost

**Correct Answer:** B
**Explanation:** Production deployment requires balancing multiple factors: model performance (accuracy), latency requirements (user experience), and computational costs (infrastructure expenses). The optimal solution rarely maximizes just one dimension but finds the best trade-off for business requirements.

---

## Answer Key
1. B  2. C  3. C  4. B  5. C  6. B  7. B  8. B  9. B  10. B
11. B  12. B  13. B  14. C  15. C  16. C  17. B  18. B  19. C  20. C
21. B  22. C  23. A  24. C  25. B  26. C  27. B  28. B  29. B  30. B
31. B  32. B  33. B  34. C  35. B  36. B  37. C  38. B  39. B  40. B
41. C  42. A  43. B  44. C  45. A  46. A  47. B  48. B  49. B  50. B

## Scoring
- 45-50 correct (90-100%): Excellent preparation, ready for exam
- 40-44 correct (80-89%): Good preparation, review weak areas
- 35-39 correct (70-79%): Moderate preparation, need more study
- Below 35 correct (<70%): Significant study required