# NVIDIA Certified Associate Quick Reference Sheet
## Last-Minute Review Guide for NCA-GENL Exam

---

## Exam Format Quick Facts
- **50-51 questions, 60 minutes**
- **1.2 minutes per question average**
- **Multiple choice (A, B, C, D)**
- **Online, remotely proctored**
- **Aim for 80%+ on practice tests**

---

## Transformer Architecture Essentials

### Core Components
```
Input → Embedding + Positional Encoding → Multi-Head Attention → 
Add & Norm → Feed-Forward → Add & Norm → Output
```

### Attention Formula
```
Attention(Q,K,V) = softmax(QK^T / √d_k)V
```

### Q, K, V Matrices (MEMORIZE!)
- **Query (Q):** Token seeking information
- **Key (K):** Compatibility measure with other tokens  
- **Value (V):** Information to be retrieved

### Positional Encoding Purpose
- Transformers process in parallel (no inherent order)
- Adds sequence position information
- Uses sine/cosine functions or learned embeddings

---

## Model Types Quick Comparison

| Model | Architecture | Training | Best For |
|-------|-------------|----------|----------|
| **BERT** | Encoder-only | Masked LM + NSP | Understanding tasks |
| **GPT** | Decoder-only | Autoregressive | Text generation |
| **T5** | Encoder-Decoder | Text-to-text | Versatile tasks |

---

## Evaluation Metrics (HIGH PRIORITY)

### BLEU Score
- **Purpose:** Text quality evaluation
- **Range:** 0 to 1 (higher = better)
- **Measures:** N-gram overlap with reference
- **Higher BLEU = More similar to reference text**
- **NOT broader vocabulary or better understanding**

### ROUGE Score  
- **Purpose:** Summarization evaluation
- **Types:** ROUGE-N, ROUGE-L, ROUGE-W
- **Focus:** Recall-oriented evaluation

### Model Quantization Benefits
1. **Reduced memory usage**
2. **Faster inference**
3. **Lower computational requirements**
4. **Reduced energy consumption**

---

## NVIDIA Tools (CRITICAL KNOWLEDGE)

### Triton Inference Server
- **Purpose:** High-performance model serving
- **Features:** Dynamic batching, multi-framework support
- **Key benefit:** Optimizes throughput by combining requests

### TensorRT
- **Purpose:** Inference optimization
- **Techniques:** Layer fusion, precision calibration
- **Supports:** Multiple formats including ONNX

### NeMo Framework
- **Purpose:** Building AI applications
- **Features:** Pre-trained models, distributed training
- **Domains:** NLP, speech, vision

### ONNX
- **Purpose:** Model portability across frameworks
- **Benefit:** Framework interoperability

---

## Transfer Learning & Fine-tuning

### Transfer Learning Benefits
1. **Reduces training time and computational requirements**
2. Leverages knowledge from large datasets
3. Enables few-shot and zero-shot learning
4. Does NOT eliminate need for labeled data entirely

### EDA Importance
- **Purpose:** Uncover patterns and anomalies in dataset
- Essential for successful fine-tuning
- Helps identify biases and data quality issues

### Catastrophic Forgetting
- Model loses pre-trained knowledge during fine-tuning
- Common challenge in sequential learning

---

## A/B Testing (FREQUENTLY TESTED)

### Best Practice for LLM Deployment
- **Deploy both models simultaneously**
- **Serve traffic equally (50/50 split)**
- Monitor performance metrics
- Enable statistical significance testing

---

## Layer Normalization
- **Purpose:** Stabilize training, enable deeper networks
- **Formula:** LayerNorm(x) = γ × (x - μ) / σ + β
- Reduces internal covariate shift

---

## Advanced Concepts

### Few-Shot Learning
- Learning tasks with minimal examples in prompt
- No additional fine-tuning required

### RAG (Retrieval-Augmented Generation)
- Combines retrieval + generation
- Reduces hallucination
- Provides access to external knowledge

### Beam Search
- Explores multiple candidate sequences
- Better quality than greedy search
- Maintains k best sequences (beams)

### Diffusion Models
- **Forward:** Add noise to data
- **Reverse:** Learn to remove noise
- Used for image/text generation

---

## Common Exam Traps

### BLEU Score Misconceptions
❌ Higher BLEU = broader vocabulary  
❌ Higher BLEU = better understanding  
✅ Higher BLEU = more similar to reference

### Attention Matrix Confusion
❌ Query = information to retrieve  
❌ Key = token seeking information  
✅ Query = token seeking information  
✅ Value = information to retrieve

### Quantization Benefits
❌ Increases model size  
❌ Improves training accuracy  
✅ Reduces memory and latency

---

## Time Management Strategy

### First Pass (30-35 minutes)
- Answer confident questions immediately
- Mark uncertain questions for review
- Don't spend more than 1 minute per question

### Second Pass (20-25 minutes)  
- Review marked questions
- Use elimination strategy
- Apply educated guessing

### Final Check (5 minutes)
- Verify answer sheet completeness
- Double-check flagged questions

---

## Key Formulas to Remember

```
Self-Attention: Attention(Q,K,V) = softmax(QK^T / √d_k)V

Layer Norm: LayerNorm(x) = γ × (x - μ) / σ + β

Positional Encoding:
PE(pos, 2i) = sin(pos/10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
```

---

## Last-Minute Checklist

### Technical Concepts ✓
- [ ] Transformer architecture components
- [ ] Q, K, V matrix purposes  
- [ ] BLEU vs ROUGE differences
- [ ] Quantization benefits
- [ ] Layer normalization purpose

### NVIDIA Ecosystem ✓
- [ ] Triton = inference serving
- [ ] TensorRT = optimization
- [ ] NeMo = AI application framework
- [ ] ONNX = model portability

### Practical Knowledge ✓
- [ ] A/B testing (50/50 traffic split)
- [ ] EDA importance for fine-tuning
- [ ] Transfer learning benefits
- [ ] Few-shot learning definition

---

## Emergency Review (10 minutes before exam)

1. **BLEU measures similarity to reference, not vocabulary**
2. **Query matrix seeks information, Value contains information**
3. **Triton serves models, TensorRT optimizes them**
4. **A/B testing uses equal traffic distribution**
5. **Quantization reduces memory and latency**
6. **EDA uncovers patterns and anomalies**
7. **Positional encoding adds sequence order information**

---

## Confidence Boosters

- You have 60 minutes for 50 questions - that's manageable
- Mark uncertain questions and return to them
- Trust your preparation and first instincts
- Eliminate obviously wrong answers first
- Look for NVIDIA-specific terminology in questions

**Good luck! You've got this! 🚀**