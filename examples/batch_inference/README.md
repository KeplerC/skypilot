

To run the experiments 
```
sky launch clip.yaml
sky jobs launch clip.yaml
```

To construct the database from embeddings: 
```
python build_vectordb.py \
  --bucket kych-clip-embeddings \
  --prefix embeddings \
  --collection-name clip_embeddings \
  --persist-dir ./chroma_db \
  --batch-size 1000
```

To query the constructed database: 
```
python query_vectordb.py \
  --text "a photo of a black and white cat" \
  --collection-name clip_embeddings \
  --persist-dir ./chroma_db \
  --n-results 5
```