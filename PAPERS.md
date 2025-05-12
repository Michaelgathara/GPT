# Foundational Concepts & Further Reading

This page lists key research papers that are foundational to understanding the concepts and architectures implemented in this project. While not mandatory for running the code, exploring these can provide a deeper understanding of *why* things are built the way they are.

## Core Transformer and GPT Architecture

1.  **"Attention Is All You Need"** (Vaswani et al., 2017)
    * **Link:** [PDF](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
    * **Relevance:** Introduced the original Transformer architecture, which is the basis for GPT models. Understanding self-attention, multi-head attention, and positional encoding from this paper is crucial for grasping the model in this repository.

2.  **"Language Models are Unsupervised Multitask Learners"** (Radford et al., OpenAI - GPT-2 Paper)
    * **Link:** [PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
    * **Relevance:** Details the GPT-2 model, a direct predecessor and significant inspiration for the architecture and training approach used in this project. It highlights the power of large-scale unsupervised pre-training for language models.

## Training Techniques and Components

3.  **"GLU Variants Improve Transformer"** (Shazeer, 2020)
    * **Link:** [arXiv PDF](https://arxiv.org/pdf/2002.05202.pdf)
    * **Relevance:** This paper explores variants of Gated Linear Units (GLU), including SwiGLU, which is used in the feed-forward network of this project's Transformer model. It demonstrates performance improvements over standard ReLU or GELU activations.

4.  **"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"** (Dao et al., 2022)
    * **Link:** [arXiv PDF](https://arxiv.org/pdf/2205.14135.pdf)
    * **Relevance:** Describes FlashAttention, an optimized attention algorithm that significantly speeds up computation and reduces memory usage by being IO-aware. This project incorporates FlashAttention for improved training and inference efficiency on compatible hardware. (A follow-up, FlashAttention-2, further improves upon this: [arXiv PDF](https://arxiv.org/pdf/2307.08691.pdf))

5.  **"Adam: A Method for Stochastic Optimization"** (Kingma & Ba, 2014)
    * **Link:** [arXiv PDF](https://arxiv.org/pdf/1412.6980.pdf)
    * **Relevance:** Introduces the Adam optimization algorithm. This project uses AdamW, a variant of Adam that incorporates weight decay differently, which is a standard choice for training Transformers.

6.  **"Why Warmup the Learning Rate? Underlying Mechanisms and Improvements"** (Xiao et al., 2024)
    * **Link:** [arXiv PDF](https://arxiv.org/pdf/2406.09405)
    * **Relevance:** Explores the learning rate warmup strategy, a common technique used in training large neural networks (including the one in this project) to improve stability and overall performance, especially in the early stages of training.
