# Building A Billion Scale Image Vector Database With SkyPilot 

### Context: Semantic Search at Billion Scale 
Retrieval-Augmented Generation (RAG) is an advanced AI technique that enhances large language models (LLMs) by integrating external data sources into their responses.

### Step 0: Set Up The Environment
Setup Huggingface token in `~/.env`
```
HF_TOKEN=hf_xxxxx
```

To run the experiments 
```
sky launch clip.yaml --env-file ~/.env
sky jobs launch clip.yaml --env-file ~/.env
```

To construct the database from embeddings: 
```
sky jobs launch build_vectordb.yaml 
```

To query the constructed database: 
```
sky launch serve_vectordb.yaml
```