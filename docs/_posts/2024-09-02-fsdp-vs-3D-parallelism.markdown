---
layout: post
title:  "Benchmarking FSDP vs. 3D Paralellism "
date:   2024-09-02 10:45:13 +0200
categories: Research findings
---

# Benchmarking FSDP with LLaMA Factory vs. 3D Parallelism with Nanotron

In my recent experiments, I compared two distributed training frameworks: [**LLaMA Factory**](https://github.com/hiyouga/LLaMA-Factory) using FSDP (Fully Sharded Data Parallelism) with `accelerate` and [**Nanotron**](https://github.com/huggingface/nanotron/tree/main) using 3D parallelism. Below, I summarize the benchmark results and share my observations. The model used in my experiments is [meta-llama/Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B).

## Context
I came across a similar benchmark in a LightOn [blog](https://www.lighton.ai/lighton-blogs/passing-the-torch-training-a-mamba-model-for-smooth-handover), where they trained a model with 2 billion parameters on a single node with 4x A100-64GB GPUs. The results are summarized in the table below:

| Parallelism Type | Model | Batch Size | Block-wise Activation Recomputation | Throughput | TFLOPs |
|------------------|-------|------------|-------------------------------------|------------|--------|
| FSDP             | 1.6B  | 5          | Yes                                 | 11,000     | 96.25  |
| 3D (DP=2, PP=2, TP=1) | 1.6B  | 2          | Not supported                       | 4,880      | 51     |

I found these results very interesting. However, they could benefit from more comprehensive experiments for two key reasons:
- First, the training was conducted on only one node, which doesn’t account for the overhead introduced by inter-node communication.
- Second, the comparison was made using different global batch sizes, which makes the throughput not strictly comparable as the convergence rate could vary.

For these reasons, I conducted a more elaborate, though still incomplete, benchmark.

## Benchmark Setup

### Setup:
- **H100 80GB GPUs**
- **GPUs per Node:** 4
- **Examples per GPU:** 4
- **Cutoff Length:** 4096

### Results:

| #  | Framework        | Settings          | Nodes | Gradient Accum. (global bs) | Throughput (tokens/s) | 
|----|------------------|-------------------|-------|-----------------------------|-----------------------|
| 1  | **LLaMA Factory**| accelerate + FSDP | 4     | 1   (64)                     | 93.6K                 | 
| 2  | **Nanotron**     | DP=2, PP=1, TP=8  | 4     | 8   (64)                     | 83.3K                 | 
| 3  | **Nanotron**     | DP=4, PP=1, TP=4  | 4     | 4   (64)                     | 128K                  |
| 4  | **Nanotron**     | DP=4, PP=2, TP=2  | 4     | 1   (16)                     | 49.2K                 |
| 5  | **Nanotron**     | DP=4, PP=1, TP=4  | 4     | 1   (16)                     | 108K                  |
| 6  | **Nanotron**     | DP=8, PP=1, TP=4  | 8     | 4   (128)                    | 253K                  |
| 7  | **Nanotron**     | DP=4, PP=1, TP=8  | 8     | 8   (128)                    | 154K                  |





## Observations
- Comparing lines 1 and 3, 3D(2D)-parallelism is 1.36 times more efficient than FSDP.
- Lines 1 and 5 demonstrate that even with lower gradient accumulation, 3D(2D)-parallelism can still achieve higher throughput.
- The comparisons between lines 2 and 3, as well as lines 6 and 7, suggest that the optimal TP is 4, which matches the number of GPUs per node. 
- The throughput scales predictably with the number of nodes. As a rule of thumb, throughput can be approximated as `nnodes` * 23,405 tokens/s.
- When using Nanotron with TP=4 and PP=1, throughput appears to scale as `nnodes` * 32,000 tokens/s when DP is increased.
- There is a noticeable drop in throughput when using `PP > 1`. For example, with DP=4, PP=2, TP=2, throughput significantly drops to 49.2K tokens/s.
- Increasing DP while holding TP=4 and PP=1 constant yields better throughput. With DP=8, throughput reached 253K tokens/s on 8 nodes.
  
## Known Issues with Nanotron
While Nanotron shows promise in certain configurations, there are currently some known issues:

1. **Gradient Accumulation with PP > 1**: There is a bug preventing gradient accumulation when using multinodes if `PP > 1`. You can track this issue [here](https://github.com/huggingface/nanotron/issues/209).
2. **Checkpoint Resumption with PP > 1**: Another bug exists that prevents resumption from checkpoints when `PP > 1`. This issue is being tracked [here](https://github.com/huggingface/nanotron/issues/221).

## Acknowledgment
I would like to thank ChatGPT for helping refine the writing in this post.
