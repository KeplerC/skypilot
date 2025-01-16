

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
python build_vectordb.py \
  --bucket kych-clip-embeddings \
  --prefix embeddings \
  --collection-name clip_embeddings \
  --persist-dir /tmp/chroma_db \
  --batch-size 1000
```

To query the constructed database: 
```
python query_vectordb.py \
  --text "a photo of cloud" \
  --collection-name clip_embeddings \
  --persist-dir ./chroma_db \
  --n-results 5
```