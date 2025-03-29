from __future__ import annotations

import asyncio
import os
from datetime import datetime
from dataclasses import asdict

from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.utils import compute_mdhash_id, logger
from cyclic_entity_extraction import cyclic_entity_extraction

# Extend the LightRAG class with improved duo insertion method
async def improved_ainsert_duo(
    self: LightRAG,
    data_original,
    data_translated=None,
    source_language="Vietnamese",
    target_language="English",
    store_translations=True,
    translation_db_path=None
):
    """
    Async insert a document in both its original language and translated version,
    with improved cyclic entity and relation extraction.
    
    Args:
        data_original: Original document data
        data_translated: Translated document data (if None, will be generated using LLM)
        source_language: Source language name 
        target_language: Target language for translation if data_translated is None
        store_translations: Whether to store entity and relation translations
        translation_db_path: Path to store translation mappings (defaults to working_dir/translations.json)
    
    Returns:
        Tuple of (original_doc_id, translated_doc_id)
    """
    if translation_db_path is None:
        translation_db_path = os.path.join(self.working_dir, "translations.json")
    
    # First process the original document
    logger.info(f"Processing original document in duo insertion")
    original_doc_id = compute_mdhash_id(data_original.strip(), prefix="doc-")
    
    # Store original document in doc status first
    await self.doc_status.upsert({
        original_doc_id: {
            "content": data_original,
            "content_summary": self._get_content_summary(data_original),
            "content_length": len(data_original),
            "status": DocStatus.PENDING,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "language": source_language,
        }
    })
    
    # Store data in full_docs
    await self.full_docs.upsert({original_doc_id: {"content": data_original.strip()}})
    
    # Create chunks for original document
    original_chunks = {
        compute_mdhash_id(dp["content"], prefix="chunk-"): {
            **dp,
            "full_doc_id": original_doc_id,
            "language": source_language
        }
        for dp in self.chunking_func(
            data_original,
            None,
            False,
            self.chunk_overlap_token_size,
            self.chunk_token_size,
            self.tiktoken_model_name,
        )
    }
    
    # If translated data is not provided, use LLM to translate
    if data_translated is None or data_translated == "":
        logger.info(f"Generating translation for document in {target_language}")
        
        # Use a translation prompt
        translation_prompt = f"""
        You are a professional translator from {source_language} to {target_language}.
        
        Translate the following text into {target_language}, preserving the original meaning, tone,
        and all named entities as closely as possible:
        
        {data_original}
        
        Provide ONLY the translated text without any explanations or additional comments.
        """
        
        # Get LLM response
        data_translated = await self.llm_model_func(translation_prompt)
    
    # Process translated document
    logger.info(f"Processing translated document in duo insertion")
    translated_doc_id = compute_mdhash_id(data_translated.strip(), prefix="doc-")
    
    # Store translated document in doc status
    await self.doc_status.upsert({
        translated_doc_id: {
            "content": data_translated,
            "content_summary": self._get_content_summary(data_translated),
            "content_length": len(data_translated),
            "status": DocStatus.PENDING,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "language": target_language,
        }
    })
    
    # Store data for translated document
    await self.full_docs.upsert({translated_doc_id: {"content": data_translated.strip()}})
    
    # Create chunks for translated document
    translated_chunks = {
        compute_mdhash_id(dp["content"], prefix="chunk-"): {
            **dp,
            "full_doc_id": translated_doc_id,
            "language": target_language
        }
        for dp in self.chunking_func(
            data_translated,
            None,
            False,
            self.chunk_overlap_token_size,
            self.chunk_token_size,
            self.tiktoken_model_name,
        )
    }
    
    # Store chunks in vector database and text chunks storage
    await asyncio.gather(
        self.chunks_vdb.upsert(original_chunks),
        self.text_chunks.upsert(original_chunks),
        self.chunks_vdb.upsert(translated_chunks),
        self.text_chunks.upsert(translated_chunks),
    )
    
    # Update status for both documents to processing
    await asyncio.gather(
        self.doc_status.upsert({
            original_doc_id: {
                "status": DocStatus.PROCESSING,
                "updated_at": datetime.now().isoformat(),
            }
        }),
        self.doc_status.upsert({
            translated_doc_id: {
                "status": DocStatus.PROCESSING,
                "updated_at": datetime.now().isoformat(),
            }
        })
    )
    
    # Perform cyclic entity extraction (this is the key improvement)
    logger.info("Performing cyclic entity extraction to match entities across languages")
    knowledge_graph, source_entities, source_relations, target_entities, target_relations = await cyclic_entity_extraction(
        original_chunks,
        translated_chunks,
        self.chunk_entity_relation_graph,
        self.entities_vdb,
        self.relationships_vdb,
        asdict(self),
        self.llm_response_cache,
        source_language,
        target_language
    )
    
    # Update our knowledge graph with the results
    self.chunk_entity_relation_graph = knowledge_graph
    
    # Create cross-lingual edges between entities
    if len(source_entities) > 0 and len(target_entities) > 0:
        try:
            await self._create_cross_lingual_edges(
                source_entities,
                target_entities,
                source_language=source_language,
                target_language=target_language
            )
            logger.info(f"Successfully created cross-lingual edges between {len(source_entities)} source entities and {len(target_entities)} target entities")
        except Exception as e:
            logger.error(f"Error creating cross-lingual edges: {e}")
    else:
        logger.warning(f"Cannot create cross-lingual edges: source entities: {len(source_entities)}, target entities: {len(target_entities)}")
    
    # Store translation pairs if requested
    if store_translations and len(source_entities) > 0 and len(target_entities) > 0:
        try:
            await self._store_translation_pairs(
                source_entities,
                target_entities,
                source_relations,
                target_relations,
                target_language,
                translation_db_path
            )
            logger.info(f"Successfully stored translation pairs")
        except Exception as e:
            logger.error(f"Error storing translation pairs: {e}")
    elif store_translations:
        logger.warning(f"Cannot store translation pairs: source entities: {len(source_entities)}, target entities: {len(target_entities)}")
    
    # Update status for both documents to processed
    await asyncio.gather(
        self.doc_status.upsert({
            original_doc_id: {
                "status": DocStatus.PROCESSED,
                "chunks_count": len(original_chunks),
                "updated_at": datetime.now().isoformat(),
            }
        }),
        self.doc_status.upsert({
            translated_doc_id: {
                "status": DocStatus.PROCESSED,
                "chunks_count": len(translated_chunks),
                "updated_at": datetime.now().isoformat(),
            }
        })
    )
    
    # Save changes to all storages
    await self._insert_done()
    
    # Log success info
    logger.info(f"Duo insertion completed successfully with {len(source_entities)} source entities and {len(target_entities)} target entities")
    
    return original_doc_id, translated_doc_id


# Function to install the method into the LightRAG class
def install_improved_ainsert_duo():
    """Install the improved_ainsert_duo method into the LightRAG class."""
    LightRAG.ainsert_duo = improved_ainsert_duo
    
    # Also need to update the sync version to use our async version
    old_insert_duo = LightRAG.insert_duo
    def improved_insert_duo(self, *args, **kwargs):
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.ainsert_duo(*args, **kwargs))
    
    LightRAG.insert_duo = improved_insert_duo
    
    logger.info("Successfully installed improved ainsert_duo into LightRAG class")
    
    return True 