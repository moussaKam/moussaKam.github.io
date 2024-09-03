---
layout: post
title:  "Evaluating LLMs"
date:   2024-09-03 12:31:34 +0200
categories: Research findings
---
I used LLMs to generate summaries for french conversations. I considered 6 models: [`Meta-Llama-3-70B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct), [`Meta-Llama-3.1-405B-Instruct-FP8`](https://huggingface.co/meta-llama/Meta-Llama-3.1-405B-Instruct-FP8), [`Meta-Llama-3.1-70B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct), [`Mistral-Large-Instruct-2407`](https://huggingface.co/mistralai/Mistral-Large-Instruct-2407), [`Qwen2-72B-Instruct`](https://huggingface.co/Qwen/Qwen2-72B-Instruct), [`c4ai-command-r-plus`](CohereForAI/c4ai-command-r-plus). The results are interesting and counter-intuitive and that' why I decided to share them.

To evaluate these models I used four LLM judges: `Meta-Llama-3-70B-Instruct`, `Meta-Llama-3.1-70B-Instruct`, `Mistral-Large-Instruct-2407`, `Qwen2-72B-Instruct`. 

So first I started by generating summaries for 1000 conversations extracted randomy from a internal dataset using all the 6 models. To generate summaries I used the following prompt: 

    prompt = f"Générez un résumé abstractif en français de la conversation suivante : {conversation}. Le résumé doit comporter environ {int(len(conversation.split()) * 0.2)} mots. Ne générez que le résumé, sans ajouter de texte supplémentaire comme 'Voici le résumé de la conversation'. 

Then for simlicity I prompted the judge LLMs to perform a pair-wise comparision to choose 
