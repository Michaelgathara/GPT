1. Attention is all you need - https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf

2. GPT2 - https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

3. Why Warmup the Learning Rate? Underlying Mechanisms and Improvements - https://arxiv.org/pdf/2406.09405

4. Multiheaded Latent Attention: Lecture - https://www.youtube.com/watch?v=xPGvxE_hGd4&t=1899s : Slides - https://github.com/SanDiegoMachineLearning/talks/blob/main/papers/ML_Paper_Review_202411_multi-head_latent_attn.pdf

5. Deepseek V3 - https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py#L164

6. DeepSeek-V3 Explained 1: Multi-head Latent Attention - https://medium.com/data-science/deepseek-v3-explained-1-multi-head-latent-attention-ed6bee2a67c4

7. Deepseek prompts:
    * "can you explain what q_lora_rank is for your application of MLA on transformers"
    * "is it the query vector or the kv vectors that use latent_dim for multi-headed latent attention"
    * "what does pre-norm attention do?"
    * "how is my original implementation considered post-norm" - Given additional context by adding code from the Block class found in the transformer.py file found under transformer_setup