# LightRAG Deferred Embedding

This feature extension allows you to build knowledge graphs without immediately embedding them into vector databases, and later load and embed the graph data when needed.

## Why Deferred Embedding?

Building a knowledge graph with LightRAG involves several steps:

1. Chunking documents
2. Extracting entities and relationships
3. Building the graph structure
4. Embedding everything into vector databases

The embedding step (4) can be the most time-consuming, especially with large documents or slow embedding models. Deferred embedding lets you:

- Build the graph once and embed it multiple times with different embedding models
- Save time when initially processing documents
- Focus on graph quality before committing to vector embedding
- Distribute the workload (build graph on one machine, embed on another)

## Usage Options

### Option 1: Disable Vector DB Upsert Globally

When initializing the LightRAG instance, set `enable_vdb_upsert=False`:

```python
rag = LightRAG(
    working_dir="./my_graph",
    embedding_func=my_embedding_func,
    llm_model_func=my_llm_func,
    enable_vdb_upsert=False  # Disable all vector DB operations
)

# Process documents (no embedding will happen)
rag.insert("My document text")

# Save graph data for later embedding
rag.save_graph_data_sync("./saved_graph_data")

# Later, load and embed the graph
rag.load_and_embed_graph_sync(
    input_dir="./saved_graph_data",
    embedding_func=my_embedding_func,
    embedding_name="my_embedding"
)
```

### Option 2: Temporarily Disable Embedding for Specific Operations

Use the `enable_embedding` parameter in the `insert()` method:

```python
rag = LightRAG(
    working_dir="./my_graph",
    embedding_func=my_embedding_func,
    llm_model_func=my_llm_func
)

# Process documents without embedding for this operation only
rag.insert("My document text", enable_embedding=False)

# Save graph data
rag.save_graph_data_sync("./saved_graph_data")

# Load and embed the graph
rag.load_and_embed_graph_sync(
    input_dir="./saved_graph_data",
    embedding_func=my_embedding_func,
    embedding_name="my_embedding"
)
```

## Using Multiple Embedding Models

One key advantage of deferred embedding is the ability to use multiple embedding models with the same graph:

```python
# Save the graph data
graph_data_dir = rag.save_graph_data_sync()

# Load and embed with first embedding model
rag.load_and_embed_graph_sync(
    input_dir=graph_data_dir,
    embedding_func=embedding_model_1,
    embedding_name="model1"
)

# Load and embed with another embedding model
rag.load_and_embed_graph_sync(
    input_dir=graph_data_dir,
    embedding_func=embedding_model_2,
    embedding_name="model2"
)
```

## Async API

All methods also have async versions:

```python
await rag.save_graph_data()
await rag.load_and_embed_graph(input_dir="./saved_graph_data")
```

## Complete Example

See `graph_build_and_embed_demo.py` for a complete working example of deferred embedding with LightRAG.
