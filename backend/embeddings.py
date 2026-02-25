"""
Embeddings Module
Handles fetching summaries from blob storage, generating embeddings, 
and storing them in Azure AI Search
"""

import os
import json
import uuid
from datetime import datetime
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

# Azure AI Search Configuration
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "multimodal-rag-index_2")

# Azure Storage Configuration
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
SUMMARIES_CONTAINER_NAME = os.getenv("AZURE_SUMMARIES_CONTAINER_NAME", "image-summaries")


def get_openai_client():
    """Initialize Azure OpenAI client"""
    return AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )


def get_blob_service_client():
    """Initialize Azure Blob Service Client"""
    return BlobServiceClient.from_connection_string(
        AZURE_STORAGE_CONNECTION_STRING
    )


def get_search_client():
    """Initialize Azure Search Client"""
    return SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY)
    )


def get_search_index_client():
    """Initialize Azure Search Index Client"""
    return SearchIndexClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY)
    )


def create_search_index():
    """
    Create Azure AI Search index with proper schema for embeddings
    
    Returns:
        bool: True if index created or already exists
    """
    try:
        index_client = get_search_index_client()
        
        # Check if index already exists
        try:
            index_client.get_index(AZURE_SEARCH_INDEX_NAME)
            print(f"Index '{AZURE_SEARCH_INDEX_NAME}' already exists")
            return True
        except:
            # Index doesn't exist, create it
            print(f"Creating index '{AZURE_SEARCH_INDEX_NAME}'...")
        
        # Define index fields
        fields = [
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,
            ),
            SearchableField(
                name="image_name",
                type=SearchFieldDataType.String,
                filterable=True,
                sortable=True,
            ),
            SearchableField(
                name="image_url",
                type=SearchFieldDataType.String,
                filterable=True,
            ),
            SearchableField(
                name="summary",
                type=SearchFieldDataType.String,
                analyzer_name="en.microsoft",
            ),
            SearchableField(
                name="summary_full",
                type=SearchFieldDataType.String,
                analyzer_name="en.microsoft",
            ),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,
                vector_search_profile_name="myHnswProfile",
            ),
            SimpleField(
                name="analyzed_at",
                type=SearchFieldDataType.String,
                filterable=True,
                sortable=True,
            ),
            SimpleField(
                name="indexed_at",
                type=SearchFieldDataType.String,
                filterable=True,
                sortable=True,
            ),
            SimpleField(
                name="status",
                type=SearchFieldDataType.String,
                filterable=True,
            ),
        ]
        
        # Configure vector search
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(name="myHnsw"),
            ],
            profiles=[
                VectorSearchProfile(
                    name="myHnswProfile",
                    algorithm_configuration_name="myHnsw",
                ),
            ],
        )
        
        # Create index
        index = SearchIndex(
            name=AZURE_SEARCH_INDEX_NAME,
            fields=fields,
            vector_search=vector_search,
        )
        
        result = index_client.create_index(index)
        print(f"Index '{AZURE_SEARCH_INDEX_NAME}' created successfully")
        return True
        
    except Exception as e:
        if "already exists" not in str(e):
            print(f"Warning: Failed to create index: {str(e)}")
        return False


def generate_embeddings(text: str) -> list:
    """
    Generate embeddings for text using Azure OpenAI's text-embedding-ada-002 model
    
    Args:
        text: Text to generate embeddings for
        
    Returns:
        list: Embedding vector (list of floats)
    """
    try:
        client = get_openai_client()
        
        response = client.embeddings.create(
            input=text,
            model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        )
        
        # Extract embedding from response
        embedding = response.data[0].embedding
        return embedding
        
    except Exception as e:
        raise Exception(f"Failed to generate embeddings: {str(e)}")


def fetch_summaries_from_blob(summary_file_name: str = None) -> list:
    """
    Fetch image summaries from blob storage
    
    Args:
        summary_file_name: Specific summary file to fetch (optional)
                          If not provided, fetches the latest one
        
    Returns:
        list: List of image summaries with image_name, image_url, summary
    """
    try:
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(
            container=SUMMARIES_CONTAINER_NAME
        )
        
        # If no specific file provided, get the latest one
        if not summary_file_name:
            blobs = list(container_client.list_blobs())
            if not blobs:
                raise Exception(f"No summary files found in {SUMMARIES_CONTAINER_NAME} container")
            
            # Sort by creation time and get latest
            latest_blob = sorted(blobs, key=lambda b: b.creation_time, reverse=True)[0]
            summary_file_name = latest_blob.name
        
        # Download the summary file
        blob_client = container_client.get_blob_client(summary_file_name)
        blob_data = blob_client.download_blob().readall()
        
        # Parse JSON
        summaries = json.loads(blob_data.decode('utf-8'))
        
        return summaries
        
    except Exception as e:
        raise Exception(f"Failed to fetch summaries from blob storage: {str(e)}")


def prepare_search_document(summary_item: dict, embedding: list) -> dict:
    """
    Prepare a document for Azure AI Search
    
    Args:
        summary_item: Summary item from blob storage
        embedding: Embedding vector
        
    Returns:
        dict: Search document ready for upload
    """
    # Use image_name as document ID to prevent duplicates on re-indexing
    doc_id = summary_item.get("image_name", "").replace("/", "-").replace(" ", "_")
    
    # Extract summary text for display
    summary_data = summary_item.get("summary", {})
    if isinstance(summary_data, dict):
        summary_text = summary_data.get("description", json.dumps(summary_data))
    else:
        summary_text = str(summary_data)
    
    document = {
        "id": doc_id,
        "image_name": summary_item.get("image_name", ""),
        "image_url": summary_item.get("image_url", ""),
        "summary": summary_text,
        "summary_full": json.dumps(summary_item.get("summary", {})),
        "embedding": embedding,
        "analyzed_at": summary_item.get("analyzed_at", datetime.now().isoformat()),
        "indexed_at": datetime.now().isoformat(),
        "status": summary_item.get("status", "unknown")
    }
    
    return document


def upload_to_search_index(documents: list) -> dict:
    """
    Upload documents with embeddings to Azure AI Search
    
    Args:
        documents: List of documents to upload
        
    Returns:
        dict: Upload result with success and failed count
    """
    try:
        search_client = get_search_client()
        
        # Upload documents (this will merge with existing docs based on ID)
        result = search_client.upload_documents(documents)
        
        # Count successes and failures
        successful = sum(1 for r in result if r.succeeded)
        failed = len(result) - successful
        
        return {
            "total_documents": len(documents),
            "successful_uploads": successful,
            "failed_uploads": failed,
            "upload_results": [
                {
                    "doc_id": r.key,
                    "succeeded": r.succeeded,
                    "error_message": getattr(r, "error_message", None)
                }
                for r in result
            ]
        }
        
    except Exception as e:
        raise Exception(f"Failed to upload documents to search index: {str(e)}")


def remove_duplicate_documents_from_index() -> dict:
    """
    Remove duplicate documents from Azure AI Search index.
    Keeps only the latest document for each image_name.
    
    Returns:
        dict: Result with number of duplicates removed
    """
    try:
        search_client = get_search_client()
        
        # Search for all documents
        results = search_client.search(
            search_text="*",
            select=["id", "image_name", "indexed_at"],
            top=10000
        )
        
        # Group documents by image_name
        docs_by_image = {}
        all_docs = list(results)
        
        for doc in all_docs:
            image_name = doc.get("image_name", "")
            if image_name:
                if image_name not in docs_by_image:
                    docs_by_image[image_name] = []
                docs_by_image[image_name].append({
                    "id": doc.get("id"),
                    "indexed_at": doc.get("indexed_at", "")
                })
        
        # Find duplicates (images with more than one document)
        docs_to_delete = []
        duplicates_found = 0
        
        for image_name, docs in docs_by_image.items():
            if len(docs) > 1:
                duplicates_found += len(docs) - 1
                # Sort by indexed_at and keep the latest, delete the rest
                sorted_docs = sorted(docs, key=lambda x: x.get("indexed_at", ""), reverse=True)
                for doc in sorted_docs[1:]:  # Delete all but the latest
                    docs_to_delete.append({"id": doc["id"]})
        
        # Delete duplicate documents
        deleted_count = 0
        if docs_to_delete:
            print(f"Deleting {len(docs_to_delete)} duplicate documents...")
            try:
                # Use delete_documents method
                result = search_client.delete_documents(docs_to_delete)
                deleted_count = sum(1 for r in result if r.succeeded)
                print(f"Successfully deleted {deleted_count} duplicate documents")
            except AttributeError:
                # Fallback: use upload_documents with "@search.action": "delete"
                print("Using fallback delete method...")
                delete_docs = [{"id": doc["id"], "@search.action": "delete"} for doc in docs_to_delete]
                result = search_client.upload_documents(delete_docs)
                deleted_count = sum(1 for r in result if r.succeeded)
                print(f"Successfully deleted {deleted_count} duplicate documents (using fallback)")
        
        return {
            "total_documents_checked": len(all_docs),
            "duplicate_images_found": len([image_name for image_name, docs in docs_by_image.items() if len(docs) > 1]),
            "total_duplicates_removed": duplicates_found,
            "successfully_deleted": deleted_count,
            "status": "completed"
        }
        
    except Exception as e:
        raise Exception(f"Failed to remove duplicates: {str(e)}")


def embed_and_index_summaries(summary_file_name: str = None) -> dict:
    """
    Main function: Fetch summaries, generate embeddings, and index them
    
    Args:
        summary_file_name: Specific summary file to process (optional)
        
    Returns:
        dict: Processing result with embeddings created and indexed count
    """
    try:
        # Create index if it doesn't exist
        print(f"Ensuring index '{AZURE_SEARCH_INDEX_NAME}' exists...")
        create_search_index()
        
        print(f"Fetching summaries from blob storage ({SUMMARIES_CONTAINER_NAME})...")
        summaries = fetch_summaries_from_blob(summary_file_name)
        
        if not summaries:
            return {
                "status": "completed",
                "total_summaries": 0,
                "embeddings_created": 0,
                "indexed_documents": 0,
                "message": "No summaries found to process"
            }
        
        print(f"Found {len(summaries)} summaries. Generating embeddings...")
        
        documents_to_index = []
        embedding_errors = []
        
        # Process each summary
        for idx, summary_item in enumerate(summaries):
            try:
                # Skip failed analyses
                if summary_item.get("status") == "failed":
                    print(f"  Skipping failed analysis: {summary_item.get('image_name')}")
                    continue
                
                # Extract text to embed
                summary_data = summary_item.get("summary", {})
                if isinstance(summary_data, dict):
                    # Combine all text fields into a single string for embedding
                    text_parts = [
                        str(summary_data.get("description", "")),
                        str(summary_data.get("main_subjects", "")),
                        str(summary_data.get("colors", "")),
                        str(summary_data.get("context", ""))
                    ]
                    text_to_embed = " ".join([t for t in text_parts if t])
                else:
                    text_to_embed = str(summary_data)
                
                # Generate embedding
                print(f"  Generating embedding for: {summary_item.get('image_name')}")
                embedding = generate_embeddings(text_to_embed)
                
                # Prepare document for search index
                document = prepare_search_document(summary_item, embedding)
                documents_to_index.append(document)
                
            except Exception as e:
                error_msg = f"Error processing {summary_item.get('image_name')}: {str(e)}"
                print(f"  ❌ {error_msg}")
                embedding_errors.append(error_msg)
        
        if not documents_to_index:
            return {
                "status": "failed",
                "total_summaries": len(summaries),
                "embeddings_created": 0,
                "indexed_documents": 0,
                "errors": embedding_errors,
                "message": "No valid summaries to embed"
            }
        
        # Deduplicate documents by image_name (keep only the latest)
        print(f"\nDeduplicating documents...")
        unique_docs = {}
        for doc in documents_to_index:
            image_name = doc.get("image_name", "")
            # Keep the latest document for each image (by indexed_at)
            if image_name not in unique_docs or doc.get("indexed_at", "") > unique_docs[image_name].get("indexed_at", ""):
                unique_docs[image_name] = doc
        
        documents_to_index = list(unique_docs.values())
        print(f"After deduplication: {len(documents_to_index)} unique documents to upload")
        
        print(f"\nUploading {len(documents_to_index)} documents to Azure AI Search...")
        upload_result = upload_to_search_index(documents_to_index)
        
        return {
            "status": "completed",
            "total_summaries": len(summaries),
            "embeddings_created": len(documents_to_index),
            "indexed_documents": upload_result["successful_uploads"],
            "failed_uploads": upload_result["failed_uploads"],
            "upload_results": upload_result["upload_results"],
            "errors": embedding_errors,
            "processed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise Exception(f"Failed to embed and index summaries: {str(e)}")


def validate_configuration() -> bool:
    """
    Validate that all required Azure configurations are present
    
    Returns:
        bool: True if configuration is valid
        
    Raises:
        ValueError: If configuration is missing
    """
    required_vars = [
        ("AZURE_OPENAI_API_KEY", AZURE_OPENAI_API_KEY),
        ("AZURE_OPENAI_ENDPOINT", AZURE_OPENAI_ENDPOINT),
        ("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", AZURE_OPENAI_EMBEDDING_DEPLOYMENT),
        ("AZURE_SEARCH_ENDPOINT", AZURE_SEARCH_ENDPOINT),
        ("AZURE_SEARCH_KEY", AZURE_SEARCH_KEY),
        ("AZURE_SEARCH_INDEX_NAME", AZURE_SEARCH_INDEX_NAME),
        ("AZURE_STORAGE_CONNECTION_STRING", AZURE_STORAGE_CONNECTION_STRING),
    ]
    
    missing = [var for var, value in required_vars if not value]
    
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}"
        )
    
    return True


def search_similar_images(query_embedding: list, top_k: int = 5) -> dict:
    """
    Search for similar images in Azure AI Search using vector similarity
    
    Args:
        query_embedding: Embedding vector from user image
        top_k: Number of similar images to return (default: 5)
        
    Returns:
        dict: Similar images with scores and details
    """
    try:
        search_client = get_search_client()
        
        # Perform vector search with extra results to account for deduplication
        results = search_client.search(
            search_text=None,
            vector_queries=[
                {
                    "kind": "vector",
                    "k": top_k * 2,  # Request more to account for deduplication
                    "fields": "embedding",
                    "vector": query_embedding,
                }
            ],
            select=["id", "image_name", "image_url", "summary", "analyzed_at", "status"],
            top=top_k * 2,
        )
        
        # Format results and deduplicate by image_name
        similar_images = []
        seen_images = set()
        
        for result in results:
            image_name = result.get("image_name", "")
            
            # Skip if we've already seen this image
            if image_name in seen_images:
                continue
            
            seen_images.add(image_name)
            similar_images.append({
                "image_name": image_name,
                "image_url": result.get("image_url", ""),
                "summary": result.get("summary", ""),
                "analyzed_at": result.get("analyzed_at", ""),
                "status": result.get("status", ""),
                "similarity_score": result.get("@search.score", 0),
                "document_id": result.get("id", ""),
            })
            
            # Stop once we have enough unique images
            if len(similar_images) >= top_k:
                break
        
        return {
            "total_results": len(similar_images),
            "similar_images": similar_images,
        }
        
    except Exception as e:
        raise Exception(f"Failed to search similar images: {str(e)}")
