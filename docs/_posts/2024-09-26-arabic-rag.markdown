---
layout: post
title:  "Building a Retrieval-Augmented Generation System with Wikipedia and Language Models"
date:   2024-09-03 12:31:34 +0200
categories: Research findings
---

In this post, I will walk you through a simple Retrieval-Augmented Generation (RAG) system that leverages Wikipedia content for language generation using a combination of multi-threaded web scraping, text chunking, document encoding, and large language models. Retrieval-Augmented Generation is a powerful technique that allows a model to generate more accurate and contextually relevant responses by retrieving information from a corpus and integrating it into its answer.

The full code can be found here: 

## why do we need RAG ?
Even if LLMs are pretrained on extremely large corpora covering almost all human knowledge, they still suffer from hallucinations and factual inaccuracies. The frequency of these hallucinations increases significantly when the size of the LLM decreases or when dealing with languages that are not extensively covered during the pretraining phase. Often, users must choose between expensive, very large models with limited hallucinations and smaller models with a high rate of inaccuracies. This is where Retrieval-Augmented Generation (RAG) comes into play, bridging the gap between large and smaller models by incorporating a retrieval component. This component retrieves relevant documents with factual information and enriches the user’s query, enabling the LLM to produce more accurate and reliable outputs.

To better understand the limitations that RAG addresses, take a ~1.5B parameter model (e.g., `Qwen/Qwen2.5-1.5B-Instruct`) and ask it questions about a specific historical topic in Arabic (for example, `الحرب العالمية الثانية` or "World War II"). Then, ask the same question again, but this time provide context that directly or indirectly contains the answer, and compare the outputs in both cases. I tested this approach, and the results are shown in the screenshots below.

No Context         | With Context
:-------------------------:|:-------------------------:
![image](/assets/images/2024-09-26-arabic-rag/no_context.png)| ![image](/assets/images/2024-09-26-arabic-rag/with_context.png)




In summary, RAG automates this process. Instead of manually finding and injecting the relevant context, RAG uses a retriever model to scan a database of documents, retrieve the top documents relevant to the query, and incorporate them into the prompt for the LLM to produce a more accurate response.


This tutorial will cover:
1. Extracting Wikipedia content using Wikipedia's API.
2. Efficiently handling large-scale data retrieval with multithreading.
3. Chunking text for better handling by a language model.
4. Building a document retriever using sentence embeddings.
5. Using Gradio to create a simple interface for querying the system.

Let's dive into the core code and see how this system works step-by-step.

---

## Extracting Wikipedia Content Using Wikipedia's API

The first step is to extract content from a given Wikipedia page and its related links. For this, I used the `wikipediaapi` library along with `requests` for making API calls. This helps in creating a custom corpus for your RAG system.

Here's a quick look at the code used for fetching Wikipedia links and extracting the content:

```python
def extract_wikipedia_content(title, language_code="ar"):
    url = f"https://{language_code}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "format": "json",
        "explaintext": True,
        "titles": "_".join(title.split()),
    }
    response = requests.get(url, params=params)
    data = response.json()
    # Extract the page content
    page = next(iter(data["query"]["pages"].values()))
    extract = page.get("extract", "No extract available")
    return extract
```

I also implemented a function to retrieve all the links from a given Wikipedia page:

```python
def get_wikipedia_links(page_title, language_code="ar"):
    wiki = wikipediaapi.Wikipedia(language=language_code)
    page = wiki.page(page_title)
    return [str(el) for el in page.links] if page.exists() else []
```

This allows us to gather a list of related Wikipedia links to fetch content from.

## Using Multithreading to Speed Up Content Retrieval

Fetching large amounts of data from Wikipedia can be time-consuming. To optimize this, I employed the `ThreadPoolExecutor` for multithreading, significantly speeding up the retrieval process.

```python
def extract_contents_multithreading(links, max_workers=5, language_code="ar"):
    contents = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_link = {
            executor.submit(extract_wikipedia_content, link, language_code=language_code): link
            for link in links
        }
        for future in tqdm(as_completed(future_to_link), total=len(links)):
            link = future_to_link[future]
            try:
                contents.append(future.result())
            except Exception as exc:
                print(f"An error occurred while processing link {link}: {exc}")
    return contents
```

## Chunking Text for Efficient Language Model Input

Once we have the Wikipedia content, it’s crucial to chunk the text into smaller segments. This ensures that the language model can process the input efficiently without hitting token limits.

```python
def chunk_text(text, chunk_size=700, overlap=150):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i : i + chunk_size]
        chunks.append(" ".join(chunk))
    return chunks
```

By using overlapping chunks, we preserve context between chunks, improving retrieval results when using a language model.

## Building a Simple Document Retriever with Sentence Embeddings

For the retrieval component, I used a sentence-transformer model to encode each chunk. This way, I can search for the most relevant chunks based on the query.

```python
retriever = Retriever(args.embedding_model, device=args.device)
retriever.encode_documents(chunks)
```

The retriever model will then allow us to rank the chunks and find the top-k most relevant segments based on the query.

## Creating a User Interface with Gradio

Finally, I built a simple Gradio UI for the RAG system. Gradio is a fantastic library for creating easy-to-use interfaces for machine learning demos. Users can input a query, and the system will display the generated response alongside the supporting context.

```python
with gr.Blocks() as demo:
    query = gr.Textbox(lines=3, max_lines=8, interactive=True, label="query", rtl=True)
    answer = gr.Textbox(placeholder="", label="Answer", elem_id="q-input", lines=5, interactive=False, rtl=True)
    context = gr.Textbox(placeholder="", label="Answer", elem_id="q-input", lines=5, interactive=False, rtl=True, visible=False)
    submit = gr.Button("Submit")
    submit.click(instruct, inputs=query, outputs=[answer, context])
```

## Conclusion

The complete code ties together web scraping, text processing, and language model inference into a cohesive system. Such a setup can be applied to various use cases, including building chatbots, generating summaries, or creating custom knowledge bases from structured data sources like Wikipedia.

You can find the full code [here](link-to-your-github).

I hope this guide helps you understand how to build your own RAG system. Feel free to reach out with any questions or improvements!

---

This outline covers the main technical points of your code and presents it in an accessible manner for blog readers. Feel free to expand each section with more details or add visual aids like code snippets and diagrams to make it more engaging.