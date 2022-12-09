---
layout:  post
title:  "Fantastic In-context Examples and Where to Find Them (MIT 6.S898 final project blog post)"
tags:  [language models,  few-shot learning]
authors: Pan, Elton
---

Recent developments in large causal language models (eg. GPT-3) have enabled many tasks (eg. entity recognition, extractive Q&A) to be performed using few-shot learning. However, the few-shot performance of GPT models on downstream tasks is often sensitive to in-context training examples in the prompt i.e. the downstream performance could range from SOTA to random guessing depending on the training examples. In this blog, we aim to illustrate what makes an in-context example good (or not). We do this by showing one can get better downstream performance from GPT by selecting in-context training examples based on their proximity to test examples in embedding space.


# Large language models and few-shot learning
In machine learning, **few-shot learning** involves a model learning to perform a task with only a **few labels**. In other words, the objective of the model is to generalize to unseen examples both *quickly* and *efficiently* with **few** training datapoints.

In recent years, OpenAI (and many companies) have released large language models (LLMs) with ever increasing number of parameters. For example, GPT-3 has 175 billion parameters (> 1000x BERT released by Google). This is due to the empirical observation that by scaling up model and data size, models start to exhibit **emergent phenomena** such as few-shot, or even zero-shot learning (model performing task with no training labels!). Therefore, few-shot learning has garnered considerable interest in both academia and industry as one no longer needs as many labels to perform a task we need. This is especially relevant to label-scarce domains - think of scientific/biomedical/clinical domains where expert labels are hard/expensive to procure (domain-expertise is rare)!

While this sounds all good, there are multiple barriers toward the adoption of LLM in academia and industry:
- LLMs are slow and guzzle up lots of compute
- LLMs (often being black boxes) are hard to explain
- LLMs few-shot learning **performance is highly sensitive to in-context examples**

 <img src="{{ site.url }}/public/images/2022-11-30-GPT-in-context/llm_instability.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />
 ***Table 1:** Results of GPT-3 on the SST-2 sentiment analysis dataset. Five different examples are randomly selected from the training set for each trial. Different contexts induce different accuracies on the test set. Source: Liu et al., 2021. What Makes Good In-Context Examples for GPT-3?. arXiv preprint arXiv:2101.06804.*

Small variations in the LLM prompt can have a huge influence on its performance on a downstream task. Specifically, it has been reported that accuracies of a LLM classifier on a sentiment analysis tasks can vary drastically depending on the **choice of in-context examples** in the prompt (Table 1), hence raising the important question of:

>  ***What makes good in-context examples for GPT/LLM?***

# What makes good in-context examples for GPT and how to find them: a k-nearest neighbor approach
 <img src="{{ site.url }}/public/images/2022-11-30-GPT-in-context/kate.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />
***Figure 1:** GPT-3 in-context example selection. Blue star: test example, white points: unused training samples, grey points: randomly sampled training samples, red points: training samples selected by the k-nearest neighbors in the embedding space of an encoder. Source: Liu et al., 2021. What Makes Good In-Context Examples for GPT-3?. arXiv preprint arXiv:2101.06804.*


A solution to the said instability of performance is presented in paper **"What Makes Good In-Context Examples for GPT-3?"**. As presented in Fig. 1, the central idea is that the in-context examples that are closer to the test sample (red points) in the embedding space will give rise to stronger performance relative to the further ones.

Based on this simple intuition, the authors developed a strategy that first uses a sentence encoder to encode examples in both training and test set to vector representations. During inference on a test point, they retrieve its **k nearest neighbors** from the training set according to a distance metric (L2 or cosine) as the in-context examples for few-shot learning. Results show that this is a generalizable method of increasing and stabilizing GPT-3's performance in sentiment analysis, table-to-text generation and question answering.

**TLDR:** *Semantically similar* training examples (in an embedding space) make good in-context examples for GPT

# New experiments: a k-means approach
Since the paper from the previous section  is a retrieval-based method, it requires a sizeable set of ***labeled*** data (>1000 human labels), which is often **not realistic** in *label-scarce* NLP fields like the scientific domain (labels from domain experts are expensive!).

Building on the previous experiments in the paper, we decided to flip the switch and test this hypothesis under a different scenario that is more suitable for label-scarce NLP fields:

**The research question is:** 

> Given an *unlabeled* dataset, which **one datapoint** does the human label for **1-shot** learning?

Note, we emphasize again, this is for 1-shot learning, meaning we learn to perform the NLP task with just **one example**.

Our experiments show that the 1-shot performance (on a materials extraction dataset) of a GPT-J model is correlated with the distance between training example to test examples in some embedding space. For example, in Fig. 2, we encoded each datapoint using an encoder (eg. MatSciBERT). Each *grey* point is a *test* example. Each *colored* point is a *training* example with its color corresponding to its 1-shot performance when we used it as the in-context training example.

In Fig. 2 (left), we observe that *in-distribution* training datapoints (near the centre of grey distribution) tend to give better performance (see yellow/orange vs purple) compared to *out-of-distribution* training datapoints (at the fringe). Great! Our findings aligns with the intuition presented in the paper!

 <img src="{{ site.url }}/public/images/2022-11-30-GPT-in-context/gpt-k-means.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />
***Figure 2:** **Left:** SVD of MatSciBERT embeddings of training  (grey) and test datapoints (colored) in materials extraction dataset.  Colored points:  Each point is a single training datapoint, with its color corresponding to GPT 1-shot F1 score on materials extraction task when conditioned on this single training datapoint  as an in-context example.  Grey points:  Test datapoints. Notice that in-distribution  training datapoints (near the centre of grey distribution) tend to give better performance (yellow/orange) compared to out-of-distribution training datapoints (at the fringe) on the materials extraction task. This offers a prior in selecting optimal in-context  examples for 1-shot learning. **Right:** 1-shot performance (metric: F1-Rouge-L  ↑)  on materials extraction task of random baseline (grey) and our approach (green) using various encoders. Here, the error bars correspond to the std of the top 3 closest examples to centroid determined by  k-means in L2 (Euclidean) space. Note that the high variance of random baseline and open-domain encoders BERT and Longformer.  Importantly, domain-specific encoders (SciBERT, MatSciBERT) gave the best performance and resulted in lower variance  in performance. Source: Pan et al. Unpublished work*

Now, motivated by this finding, we present a novel k-means approach approach which aims to find the **optimal datapoint** for GPT. First, we **encode** each datapoint using an encoder. Next, we use k-means to find the **centroid** of all datapoints. Subsequently, we find the nearest datapoint to this centroid according to some similarity metric. This selected datapoint is our optimal datapoint, which is then labeled and is our **in-context example** for 1-shot learning.

**Results** In Fig. 2 (right), the baseline consists of evaluating 1-shot performance on the test set by conditioning GPT-J on a single example randomly sampled from the training set. We can observe that, as expected, random choice of in-context example (grey) **suffers from *high variance*** in 1-shot performance with an average F1-Rouge-L of 0.589. We compare our approach to the baseline by evaluating 1-shot performance on the test set by conditioning the LM on a single example selected from the training set based on its proximity to test examples in L2 (Euclidean) space. Clearly, using various encoders (Longformer, SciBERT, MatSciBERT in green), **our method showed marked improvements** on average top-3 performance (0.611, 0.674, 0.687 respectively) over the baseline on the 1-shot learning task  with the exception of BERT. In particular, SciBERT and MatSciBERT gave the best average 1-shot performance and also has the low variance compared to the rest of the methods.

**TLDR:** Using a simple k-means approach, our approach is an *unsupervised* approach to select *good in-context examples* for GPT.

# Conclusion
NLP has been revolutionized by the inception of LLMs like GPT-3, especially in industry, where LLM model performance largely depends on the prompt, hence making the prompt (instead of the model) the trade secret! As such, being able to select high quality in-context examples is crucial in how we perform NLP now, and in the future. Hope this work can inspire NLP practitioners to use prompts that are better than random guessing!