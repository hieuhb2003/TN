### Query Param

```python
class QueryParam:
    mode: Literal["local", "global", "hybrid", "naive", "mix"] = "global"
    """Specifies the retrieval mode:
    - "local": Focuses on context-dependent information.
    - "global": Utilizes global knowledge.
    - "hybrid": Combines local and global retrieval methods.
    - "naive": Performs a basic search without advanced techniques.
    - "mix": Integrates knowledge graph and vector retrieval. Mix mode combines knowledge graph and vector search:
        - Uses both structured (KG) and unstructured (vector) information
        - Provides comprehensive answers by analyzing relationships and context
        - Supports image content through HTML img tags
        - Allows control over retrieval depth via top_k parameter
    """
    only_need_context: bool = False
    """If True, only returns the retrieved context without generating a response."""
    response_type: str = "Multiple Paragraphs"
    """Defines the response format. Examples: 'Multiple Paragraphs', 'Single Paragraph', 'Bullet Points'."""
    top_k: int = 60
    """Number of top items to retrieve. Represents entities in 'local' mode and relationships in 'global' mode."""
    max_token_for_text_unit: int = 4000
    """Maximum number of tokens allowed for each retrieved text chunk."""
    max_token_for_global_context: int = 4000
    """Maximum number of tokens allocated for relationship descriptions in global retrieval."""
    max_token_for_local_context: int = 4000
    """Maximum number of tokens allocated for entity descriptions in local retrieval."""
    ids: list[str] | None = None # ONLY SUPPORTED FOR PG VECTOR DBs
    """List of ids to filter the RAG."""
    ...
```

<details>
<summary> <b>Using Open AI-like APIs</b> </summary>

- LightRAG also supports Open AI-like chat/embeddings APIs:

```python
async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        "solar-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("UPSTAGE_API_KEY"),
        base_url="https://api.upstage.ai/v1/solar",
        **kwargs
    )

async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embed(
        texts,
        model="solar-embedding-1-large-query",
        api_key=os.getenv("UPSTAGE_API_KEY"),
        base_url="https://api.upstage.ai/v1/solar"
    )

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=4096,
            max_token_size=8192,
            func=embedding_func
        )
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag
```

</details>

<details>
<summary> <b>Using Hugging Face Models</b> </summary>

- If you want to use Hugging Face models, you only need to set LightRAG as follows:

See `lightrag_hf_demo.py`

```python
# Initialize LightRAG with Hugging Face model
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=hf_model_complete,  # Use Hugging Face model for text generation
    llm_model_name='meta-llama/Llama-3.1-8B-Instruct',  # Model name from Hugging Face
    # Use Hugging Face embedding function
    embedding_func=EmbeddingFunc(
        embedding_dim=384,
        max_token_size=5000,
        func=lambda texts: hf_embed(
            texts,
            tokenizer=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"),
            embed_model=AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        )
    ),
)
```

</details>

<details>
<summary> <b>Separate Keyword Extraction</b> </summary>

We've introduced a new function `query_with_separate_keyword_extraction` to enhance the keyword extraction capabilities. This function separates the keyword extraction process from the user's prompt, focusing solely on the query to improve the relevance of extracted keywords.

##### How It Works?

The function operates by dividing the input into two parts:

- `User Query`
- `Prompt`

It then performs keyword extraction exclusively on the `user query`. This separation ensures that the extraction process is focused and relevant, unaffected by any additional language in the `prompt`. It also allows the `prompt` to serve purely for response formatting, maintaining the intent and clarity of the user's original question.

##### Usage Example

This `example` shows how to tailor the function for educational content, focusing on detailed explanations for older students.

```python
rag.query_with_separate_keyword_extraction(
    query="Explain the law of gravity",
    prompt="Provide a detailed explanation suitable for high school students studying physics.",
    param=QueryParam(mode="hybrid")
)
```

</details>
<details>
  <summary> <b> Batch Insert </b></summary>

```python
# Basic Batch Insert: Insert multiple texts at once
rag.insert(["TEXT1", "TEXT2",...])

# Batch Insert with custom batch size configuration
rag = LightRAG(
    working_dir=WORKING_DIR,
    addon_params={
        "insert_batch_size": 20  # Process 20 documents per batch
    }
)

rag.insert(["TEXT1", "TEXT2", "TEXT3", ...])  # Documents will be processed in batches of 20
```

The `insert_batch_size` parameter in `addon_params` controls how many documents are processed in each batch during insertion. This is useful for:

- Managing memory usage with large document collections
- Optimizing processing speed
- Providing better progress tracking
- Default value is 10 if not specified

</details>

<details>
  <summary> <b> Insert with ID </b></summary>

If you want to provide your own IDs for your documents, number of documents and number of IDs must be the same.

```python
# Insert single text, and provide ID for it
rag.insert("TEXT1", ids=["ID_FOR_TEXT1"])

# Insert multiple texts, and provide IDs for them
rag.insert(["TEXT1", "TEXT2",...], ids=["ID_FOR_TEXT1", "ID_FOR_TEXT2"])
```

</details>

## Entity Merging

<details>
<summary> <b>Merge Entities and Their Relationships</b> </summary>

LightRAG now supports merging multiple entities into a single entity, automatically handling all relationships:

```python
# Basic entity merging
rag.merge_entities(
    source_entities=["Artificial Intelligence", "AI", "Machine Intelligence"],
    target_entity="AI Technology"
)
```

With custom merge strategy:

```python
# Define custom merge strategy for different fields
rag.merge_entities(
    source_entities=["John Smith", "Dr. Smith", "J. Smith"],
    target_entity="John Smith",
    merge_strategy={
        "description": "concatenate",  # Combine all descriptions
        "entity_type": "keep_first",   # Keep the entity type from the first entity
        "source_id": "join_unique"     # Combine all unique source IDs
    }
)
```

With custom target entity data:

```python
# Specify exact values for the merged entity
rag.merge_entities(
    source_entities=["New York", "NYC", "Big Apple"],
    target_entity="New York City",
    target_entity_data={
        "entity_type": "LOCATION",
        "description": "New York City is the most populous city in the United States.",
    }
)
```

Advanced usage combining both approaches:

```python
# Merge company entities with both strategy and custom data
rag.merge_entities(
    source_entities=["Microsoft Corp", "Microsoft Corporation", "MSFT"],
    target_entity="Microsoft",
    merge_strategy={
        "description": "concatenate",  # Combine all descriptions
        "source_id": "join_unique"     # Combine source IDs
    },
    target_entity_data={
        "entity_type": "ORGANIZATION",
    }
)
```

When merging entities:

- All relationships from source entities are redirected to the target entity
- Duplicate relationships are intelligently merged
- Self-relationships (loops) are prevented
- Source entities are removed after merging
- Relationship weights and attributes are preserved

</details>

## LightRAG init parameters

<details>
<summary> Parameters </summary>

| **Parameter**                           | **Type**        | **Explanation**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | **Default**                                                                                                 |
| --------------------------------------- | --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **working_dir**                         | `str`           | Directory where the cache will be stored                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | `lightrag_cache+timestamp`                                                                                  |
| **kv_storage**                          | `str`           | Storage type for documents and text chunks. Supported types:`JsonKVStorage`, `OracleKVStorage`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | `JsonKVStorage`                                                                                             |
| **vector_storage**                      | `str`           | Storage type for embedding vectors. Supported types:`NanoVectorDBStorage`, `OracleVectorDBStorage`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | `NanoVectorDBStorage`                                                                                       |
| **graph_storage**                       | `str`           | Storage type for graph edges and nodes. Supported types:`NetworkXStorage`, `Neo4JStorage`, `OracleGraphStorage`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | `NetworkXStorage`                                                                                           |
| **chunk_token_size**                    | `int`           | Maximum token size per chunk when splitting documents                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | `1200`                                                                                                      |
| **chunk_overlap_token_size**            | `int`           | Overlap token size between two chunks when splitting documents                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | `100`                                                                                                       |
| **tiktoken_model_name**                 | `str`           | Model name for the Tiktoken encoder used to calculate token numbers                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | `gpt-4o-mini`                                                                                               |
| **entity_extract_max_gleaning**         | `int`           | Number of loops in the entity extraction process, appending history messages                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | `1`                                                                                                         |
| **entity_summary_to_max_tokens**        | `int`           | Maximum token size for each entity summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | `500`                                                                                                       |
| **node_embedding_algorithm**            | `str`           | Algorithm for node embedding (currently not used)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | `node2vec`                                                                                                  |
| **node2vec_params**                     | `dict`          | Parameters for node embedding                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | `{"dimensions": 1536,"num_walks": 10,"walk_length": 40,"window_size": 2,"iterations": 3,"random_seed": 3,}` |
| **embedding_func**                      | `EmbeddingFunc` | Function to generate embedding vectors from text                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | `openai_embed`                                                                                              |
| **embedding_batch_num**                 | `int`           | Maximum batch size for embedding processes (multiple texts sent per batch)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | `32`                                                                                                        |
| **embedding_func_max_async**            | `int`           | Maximum number of concurrent asynchronous embedding processes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | `16`                                                                                                        |
| **llm_model_func**                      | `callable`      | Function for LLM generation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | `gpt_4o_mini_complete`                                                                                      |
| **llm_model_name**                      | `str`           | LLM model name for generation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | `meta-llama/Llama-3.2-1B-Instruct`                                                                          |
| **llm_model_max_token_size**            | `int`           | Maximum token size for LLM generation (affects entity relation summaries)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | `32768`（default value changed by env var MAX_TOKENS)                                                       |
| **llm_model_max_async**                 | `int`           | Maximum number of concurrent asynchronous LLM processes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | `16`（default value changed by env var MAX_ASYNC)                                                           |
| **llm_model_kwargs**                    | `dict`          | Additional parameters for LLM generation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |                                                                                                             |
| **vector_db_storage_cls_kwargs**        | `dict`          | Additional parameters for vector database, like setting the threshold for nodes and relations retrieval.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | cosine_better_than_threshold: 0.2（default value changed by env var COSINE_THRESHOLD)                       |
| **enable_llm_cache**                    | `bool`          | If `TRUE`, stores LLM results in cache; repeated prompts return cached responses                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | `TRUE`                                                                                                      |
| **enable_llm_cache_for_entity_extract** | `bool`          | If `TRUE`, stores LLM results in cache for entity extraction; Good for beginners to debug your application                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | `TRUE`                                                                                                      |
| **addon_params**                        | `dict`          | Additional parameters, e.g.,`{"example_number": 1, "language": "Simplified Chinese", "entity_types": ["organization", "person", "geo", "event"], "insert_batch_size": 10}`: sets example limit, output language, and batch size for document processing                                                                                                                                                                                                                                                                                                                                                                                                                                   | `example_number: all examples, language: English, insert_batch_size: 10`                                    |
| **convert_response_to_json_func**       | `callable`      | Not used                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | `convert_response_to_json`                                                                                  |
| **embedding_cache_config**              | `dict`          | Configuration for question-answer caching. Contains three parameters:`<br>`- `enabled`: Boolean value to enable/disable cache lookup functionality. When enabled, the system will check cached responses before generating new answers.`<br>`- `similarity_threshold`: Float value (0-1), similarity threshold. When a new question's similarity with a cached question exceeds this threshold, the cached answer will be returned directly without calling the LLM.`<br>`- `use_llm_check`: Boolean value to enable/disable LLM similarity verification. When enabled, LLM will be used as a secondary check to verify the similarity between questions before returning cached answers. | Default:`{"enabled": False, "similarity_threshold": 0.95, "use_llm_check": False}`                          |

</details>

## Insert

Phần mới có code chạy được chứ chưa oke, nối nhầm nhiều
Code ví dụ:
LIGHTRAG\index_graph_and_retrieval\build_zalo_graph_vi.py

code này build wiki thì chạy phà phà nma sang chạy zalo thì lỗi API liên tục ?????

Hiện tại insert đồ thị 1 ngôn ngữ thì matching_method = None

```python
# Sử dụng LLM (mặc định)
lightrag.insert("Nội dung tiếng Việt", language="Vietnamese")
lightrag.insert("English content", language="English")

# Chỉ dùng embedding
lightrag.insert("Ô tô là phương tiện giao thông phổ biến", language="Vietnamese", matching_method="embedding")

# Chỉ dùng LLM
lightrag.insert("Ô tô là phương tiện giao thông phổ biến", language="Vietnamese", matching_method="llm")

# Dùng cả hai
lightrag.insert("Ô tô là phương tiện giao thông phổ biến", language="Vietnamese", matching_method="hybrid")

matching_method = [embedding,llm,hybrid,None]
```

## Query

```python
ll_chunk_list, hl_chunk_list = rag.retrieval(
    query,
    param=QueryParam(
        mode="hybrid",
        only_need_context=True,
        top_k=10
    )
)
```

retrieval là hàm mới viết base trên hàm lấy context để gen câu trả lời, trả ra kết quả tuple (ll_chunk_list, hl_chunk_list) tùy vào mode lựa chọn để truy xuất. Local thì hl_chunk_list = [] và ngược lại

### Thay vector db bằng embedding mới

lightrag.create_db_with_new_embedding(embedding_func, embedding_name)

Thì nó sẽ tính embedding và lưu ra là vdb*chunks*<embedding_name>.json ví dụ embedding_name là "bge-m3" -> file tên là vdb_chunks_bge-m3.json
Sau khi tạo xong thì lúc ông tạo graph thì ông sẽ truyền tembedding_name vào attributes "embedding_func_name" thì nó sẽ load đúng cái file vector db cho embedding đó

Nếu ông không truyền vào hoặc truyền tên mà không có file vector thì nó sẽ tự động load cái file vector gốc ví dụ như là vdb_chunks.json

### raw data Label train embedding

/home/hungpv/projects/TN/data/raw_label_data

## Decoupled Graph Building and Vector DB Embedding

LightRAG now supports decoupling the graph building process from the vector database embedding. This is particularly useful when:

- You need to build a large graph but want to save the intermediate results before embedding
- You want to use the same graph with different embedding models
- Your embedding operation is computationally expensive and you want to run it separately

The workflow consists of two steps:

### Step 1: Build and serialize the graph without embedding

```python
from lightrag import LightRAG

# Create LightRAG instance with your embedding and LLM functions
rag = LightRAG(
    working_dir="./my_graph_data",
    embedding_func=my_embedding_func,
    llm_model_func=my_llm_func
)

# Build the graph without embedding to vector DB
rag.build_graph_only_sync(my_document_text)
```

### Step 2: Load the graph and embed it with different models

```python
# Later, you can load the graph and embed it with different models
rag.load_graph_and_embed_sync(embedding_func=bge_embedding_func, embedding_name="bge")

# You can use the same graph with another embedding model
rag.load_graph_and_embed_sync(embedding_func=openai_embedding_func, embedding_name="openai")
```

### Full Example

For a complete example, see the `decoupled_embedding_example.py` file in the repository.

### Using Different Embeddings for Retrieval

Once you've created multiple embeddings, you can specify which embedding to use when initializing LightRAG:

```python
# Use the BGE embeddings
rag = LightRAG(
    working_dir="./my_graph_data",
    embedding_func=bge_embedding_func,
    embedding_func_name="bge"
)

# Query with BGE embeddings
results = rag.retrieval(query, param=QueryParam(mode="hybrid"))
```

This feature offers greater flexibility in how you build and use your knowledge graphs, allowing for more efficient workflows with large datasets.
