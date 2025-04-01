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
    Async insert a document in both its original language and translated version.
    Ensures exact 1:1 mapping between entities and relationships with maximized parallel execution.
    
    Args:
        data_original: Original document data (Vietnamese)
        data_translated: Translated document data (if None, will be generated using LLM)
        source_language: Source language (default: "Vietnamese")
        target_language: Target language (default: "English")
        store_translations: Whether to store entity and relation translations
        translation_db_path: Path to store translation mappings (defaults to working_dir/translations.json)
    
    Returns:
        Tuple of (original_doc_id, translated_doc_id)
    """
    if translation_db_path is None:
        translation_db_path = os.path.join(self.working_dir, "translations.json")
    
    logger.info(f"Starting duo insertion: {source_language} and {target_language}")
    
    # Compute document IDs for original and translated docs
    original_doc_id = compute_mdhash_id(data_original.strip(), prefix="doc-")
    
    # Check if document already exists and is processed
    doc_exists = await self.doc_status.get_by_id(original_doc_id)
    if doc_exists and doc_exists.get("status") == DocStatus.PROCESSED:
        translated_doc_id = compute_mdhash_id((data_translated or "").strip(), prefix="doc-")
        logger.info(f"Document {original_doc_id} already processed, skipping duo insertion")
        return original_doc_id, translated_doc_id
    
    # If translated data is not provided, generate it
    if data_translated is None or data_translated == "":
        logger.info(f"Generating translation for document in {target_language}")
        data_translated = await self._translate_preserving_structure(
            data_original, 
            source_language,
            target_language
        )
    
    # Compute translated document ID
    translated_doc_id = compute_mdhash_id(data_translated.strip(), prefix="doc-")
    
    # Create chunks for both original and translated documents
    original_chunks = {
        compute_mdhash_id(dp["content"], prefix="chunk-"): {
            **dp,
            "full_doc_id": original_doc_id,
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
    
    translated_chunks = {
        compute_mdhash_id(dp["content"], prefix="chunk-"): {
            **dp,
            "full_doc_id": translated_doc_id,
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
    
    # Initialize all document metadata and storage operations in parallel
    init_tasks = [
        # Store document metadata
        self.doc_status.upsert({
            original_doc_id: {
                "content": data_original,
                "content_summary": self._get_content_summary(data_original),
                "content_length": len(data_original),
                "status": DocStatus.PENDING,
                "language": source_language,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
        }),
        self.doc_status.upsert({
            translated_doc_id: {
                "content": data_translated,
                "content_summary": self._get_content_summary(data_translated),
                "content_length": len(data_translated),
                "status": DocStatus.PENDING,
                "language": target_language,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
        }),
        # Store full document content
        self.full_docs.upsert({original_doc_id: {"content": data_original.strip()}}),
        self.full_docs.upsert({translated_doc_id: {"content": data_translated.strip()}}),
        # Store chunks in vector and text storage
        self.chunks_vdb.upsert(original_chunks),
        self.text_chunks.upsert(original_chunks),
        self.chunks_vdb.upsert(translated_chunks),
        self.text_chunks.upsert(translated_chunks),
    ]
    
    # Execute all initialization tasks in parallel
    await asyncio.gather(*init_tasks)
    
    # Update status to processing
    processing_tasks = [
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
    ]
    
    await asyncio.gather(*processing_tasks)
    
    # Extract entities and relations from original document with concurrency
    logger.info(f"Extracting entities and relations from {source_language} document")
    
    # We can run the entity extraction and getting the entities/relations in parallel
    extraction_tasks = [
        extract_entities(
            original_chunks,
            knowledge_graph_inst=self.chunk_entity_relation_graph,
            entity_vdb=self.entities_vdb,
            relationships_vdb=self.relationships_vdb,
            llm_response_cache=self.llm_response_cache,
            global_config=asdict(self),
        )
    ]
    
    # Wait for extraction to complete
    await asyncio.gather(*extraction_tasks)
    
    # Get entities and relations from original document
    doc_data_tasks = [
        self._get_document_entities(original_doc_id, original_chunks),
        self._get_document_relations(original_doc_id, original_chunks)
    ]
    
    original_entities, original_relations = await asyncio.gather(*doc_data_tasks)
    
    logger.info(f"Found {len(original_entities)} entities and {len(original_relations)} relations in {source_language} document")
    
    # Update status for original document
    await self.doc_status.upsert({
        original_doc_id: {
            "status": DocStatus.PROCESSED,
            "chunks_count": len(original_chunks),
            "content": data_original,
            "content_summary": self._get_content_summary(data_original),
            "content_length": len(data_original),
            "updated_at": datetime.now().isoformat(),
        }
    })
    
    # Extract corresponding entities and relations in the translated document using parallel processing
    logger.info(f"Extracting matching entities and relations from {target_language} document")
    
    # Extract entities and relations in parallel
    matching_tasks = [
        self._extract_matching_entities(
            original_entities,
            data_translated,
            source_language,
            target_language,
            translated_chunks
        ),
        # We'll get relations after entities are extracted
    ]
    
    translated_entities = await matching_tasks[0]
    
    # Now we can get the relations using the translated entities
    translated_relations = await self._extract_matching_relations(
        original_relations,
        translated_entities,
        data_translated,
        source_language,
        target_language,
        translated_chunks
    )
    
    logger.info(f"Extracted {len(translated_entities)} entities and {len(translated_relations)} relations in {target_language} document")
    
    # Verify counts match and fix if needed - can run in parallel
    verification_tasks = []
    
    if len(original_entities) != len(translated_entities):
        logger.warning(f"Entity count mismatch: {len(original_entities)} {source_language} vs {len(translated_entities)} {target_language}")
        # Force entity count to match by requesting a fix
        verification_tasks.append(
            self._fix_entity_count_mismatch(
                original_entities,
                translated_entities,
                data_translated,
                source_language,
                target_language,
                translated_chunks
            )
        )
    
    if len(original_relations) != len(translated_relations):
        logger.warning(f"Relation count mismatch: {len(original_relations)} {source_language} vs {len(translated_relations)} {target_language}")
        # Force relation count to match by requesting a fix
        verification_tasks.append(
            self._fix_relation_count_mismatch(
                original_relations,
                translated_relations, 
                translated_entities,
                data_translated,
                source_language, 
                target_language,
                translated_chunks
            )
        )
    
    # Execute verification tasks if needed
    if verification_tasks:
        verification_results = await asyncio.gather(*verification_tasks)
        
        # Update entities and relations with fixed versions
        if len(original_entities) != len(translated_entities):
            translated_entities = verification_results[0]
            verification_results = verification_results[1:]
        
        if len(original_relations) != len(translated_relations):
            translated_relations = verification_results[0]
    
    # Prepare data for batch operations - this is for translated entities/relations
    nodes_data_map = {}
    for entity in translated_entities:
        entity_name = f'"{entity["name"].upper()}"'
        
        # Get first chunk ID for this document
        chunk_id = next(iter(translated_chunks.keys()))
        
        # Prepare data
        if entity_name not in nodes_data_map:
            nodes_data_map[entity_name] = []
        
        nodes_data_map[entity_name].append({
            "entity_type": f'"{entity["type"].upper()}"',
            "description": entity["description"],
            "source_id": chunk_id,
            "language": target_language,
        })

    # Prepare data for edges
    edges_data_map = {}
    for relation in translated_relations:
        src_entity = f'"{relation["source"].upper()}"'
        tgt_entity = f'"{relation["target"].upper()}"'
        
        # Get first chunk ID for this document
        chunk_id = next(iter(translated_chunks.keys()))
        
        edge_key = (src_entity, tgt_entity)
        if edge_key not in edges_data_map:
            edges_data_map[edge_key] = []
        
        edges_data_map[edge_key].append({
            "description": relation["description"],
            "keywords": relation["keywords"],
            "weight": 1.0,
            "source_id": chunk_id,
            "language": target_language,
        })
    
    # Create tasks for entity and relation merging
    merge_tasks = []
    
    # Add entity merge tasks
    for entity_name, nodes_data in nodes_data_map.items():
        merge_tasks.append(
            _merge_nodes_then_upsert(
                entity_name, 
                nodes_data,
                self.chunk_entity_relation_graph, 
                asdict(self)
            )
        )
    
    # Add relation merge tasks
    for (src_id, tgt_id), edges_data in edges_data_map.items():
        merge_tasks.append(
            _merge_edges_then_upsert(
                src_id, 
                tgt_id, 
                edges_data,
                self.chunk_entity_relation_graph, 
                asdict(self)
            )
        )
    
    # Execute all merge operations in parallel
    all_merge_results = await asyncio.gather(*merge_tasks)
    
    # Split results into entities and relationships
    # First n results are entities, where n is the number of entities
    entity_count = len(nodes_data_map)
    all_entities_data = all_merge_results[:entity_count]
    all_relationships_data = all_merge_results[entity_count:]
    
    # Prepare data for vector databases - do this in separate loop to not block merge operations
    entities_vdb_data = {}
    for entity_data in all_entities_data:
        if not entity_data:
            continue
        
        entity_id = compute_mdhash_id(entity_data["entity_name"], prefix="ent-")
        entities_vdb_data[entity_id] = {
            "content": f"{entity_data['entity_name']} {entity_data['description']}",
            "entity_name": entity_data["entity_name"],
            "language": entity_data.get("language", target_language),
        }

    relationships_vdb_data = {}
    for rel_data in all_relationships_data:
        if not rel_data:
            continue
            
        relation_id = compute_mdhash_id(rel_data["src_id"] + rel_data["tgt_id"], prefix="rel-")
        relationships_vdb_data[relation_id] = {
            "content": f"{rel_data['keywords']} {rel_data['src_id']} {rel_data['tgt_id']} {rel_data['description']}",
            "src_id": rel_data["src_id"],
            "tgt_id": rel_data["tgt_id"],
            "language": rel_data.get("language", target_language),
        }
    
    # Final tasks to run in parallel
    final_tasks = [
        # Update vector databases
        self.entities_vdb.upsert(entities_vdb_data),
        self.relationships_vdb.upsert(relationships_vdb_data),
        
        # Update document status
        self.doc_status.upsert({
            translated_doc_id: {
                "status": DocStatus.PROCESSED,
                "chunks_count": len(translated_chunks),
                "content": data_translated,
                "content_summary": self._get_content_summary(data_translated),
                "content_length": len(data_translated),
                "updated_at": datetime.now().isoformat(),
            }
        }),
        
        # Create cross-lingual edges
        self._create_cross_lingual_edges(
            original_entities,
            translated_entities,
            source_language,
            target_language
        )
    ]
    
    # Add translation storage if requested
    if store_translations:
        final_tasks.append(
            self._store_translation_pairs(
                original_entities,
                translated_entities,
                original_relations,
                translated_relations,
                source_language,
                target_language,
                translation_db_path
            )
        )
    
    # Execute all final tasks in parallel
    await asyncio.gather(*final_tasks)
    
    # Save changes to all storages
    await self._insert_done()
    
    logger.info(f"Duo insertion completed successfully")
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