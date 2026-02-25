import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
from datetime import datetime


# Import Azure configuration
from azure_config import (
    get_blob_service_client,
    get_container_client,
    ensure_container_exists,
    validate_configuration,
    get_blob_sas_url,
    AZURE_CONTAINER_NAME,
    ALLOWED_IMAGE_EXTENSIONS,
    MAX_FILE_SIZE
)

# Import image analyzer
from image_analyzer import (
    analyze_all_images, 
    analyze_uploaded_image_for_search,
    validate_image_similarity_with_llm,
    filter_and_rank_images
)

# Import embeddings module
from embeddings import embed_and_index_summaries, validate_configuration as validate_embeddings_config, generate_embeddings, search_similar_images, remove_duplicate_documents_from_index

app = FastAPI(title="Image Upload to Azure Blob Storage")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Validate Azure configuration on startup
try:
    validate_configuration()
    ensure_container_exists()
except Exception as e:
    print(f"Warning: Azure configuration validation failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "message": "API is running",
            "container": AZURE_CONTAINER_NAME
        }
    )


async def upload_image_to_blob(file: UploadFile, blob_name: str) -> str:
    """
    Upload an image file to Azure Blob Storage
    
    Args:
        file: The uploaded file object
        blob_name: The name to store the file as in blob storage
        
    Returns:
        str: The URL of the uploaded blob
    """
    try:
        # Read file content
        file_content = await file.read()
        
        # Get container client from configuration
        container_client = get_container_client()
        
        # Upload to Azure Blob Storage
        blob_client = container_client.upload_blob(
            name=blob_name,
            data=file_content,
            overwrite=True
        )
        
        # Return the blob URL
        blob_url = blob_client.url
        return blob_url
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload image to Azure Blob Storage: {str(e)}"
        )


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """
    Endpoint to upload an image to Azure Blob Storage
    
    Args:
        file: Image file to upload
        
    Returns:
        JSON response with upload status and blob URL
    """
    try:
        # Validate file type
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in ALLOWED_IMAGE_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File type '{file_extension}' is not allowed. Allowed types: {ALLOWED_IMAGE_EXTENSIONS}"
            )
        
        # Validate file size
        max_size = MAX_FILE_SIZE
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)
        
        if file_size > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File size ({file_size / (1024*1024):.2f} MB) exceeds maximum allowed size of 50 MB"
            )
        
        # Generate unique blob name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        blob_name = f"{timestamp}_{unique_id}_{file.filename}"
        
        # Upload to Azure Blob Storage
        blob_url = await upload_image_to_blob(file, blob_name)
        
        print(f"✓ Image uploaded successfully: {blob_name}")
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Image uploaded successfully",
                "filename": file.filename,
                "blob_name": blob_name,
                "blob_url": blob_url,
                "file_size": file_size,
                "upload_time": datetime.now().isoformat()
            }
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"✗ Upload error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload image: {str(e)}"
        )


@app.post("/upload-multiple-images")
async def upload_multiple_images(files: list[UploadFile] = File(...)):
    """
    Endpoint to upload multiple images to Azure Blob Storage
    
    Args:
        files: List of image files to upload
        
    Returns:
        JSON response with list of uploaded files
    """
    uploaded_files = []
    failed_uploads = []
    
    for file in files:
        try:
            # Validate file type
            file_extension = os.path.splitext(file.filename)[1].lower()
            
            if file_extension not in ALLOWED_IMAGE_EXTENSIONS:
                failed_uploads.append({
                    "filename": file.filename,
                    "error": f"File type '{file_extension}' is not allowed"
                })
                continue
            
            # Generate unique blob name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            blob_name = f"{timestamp}_{unique_id}_{file.filename}"
            
            # Upload to Azure Blob Storage
            blob_url = await upload_image_to_blob(file, blob_name)
            
            uploaded_files.append({
                "filename": file.filename,
                "blob_name": blob_name,
                "blob_url": get_blob_sas_url(blob_name)
            })
        except Exception as e:
            failed_uploads.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return JSONResponse(
        status_code=200,
        content={
            "total_files": len(files),
            "successfully_uploaded": len(uploaded_files),
            "failed_uploads": len(failed_uploads),
            "uploaded_files": uploaded_files,
            "failed_files": failed_uploads
        }
    )


@app.delete("/delete-image/{blob_name}")
async def delete_image(blob_name: str):
    """
    Endpoint to delete a single image from Azure Blob Storage
    
    Args:
        blob_name: Name of the blob to delete
        
    Returns:
        JSON response with deletion status
    """
    try:
        # Get container client from configuration
        container_client = get_container_client()
        container_client.delete_blob(blob_name)
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Image deleted successfully",
                "blob_name": blob_name
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete image: {str(e)}"
        )


@app.delete("/delete-all-images")
async def delete_all_images():
    """
    Endpoint to delete ALL images from Azure Blob Storage container at once
    
    Returns:
        JSON response with deletion status and count of deleted images
    """
    try:
        # Get container client from configuration
        container_client = get_container_client()
        
        # List all blobs in the container
        blobs = container_client.list_blobs()
        blob_names = [blob.name for blob in blobs]
        
        if not blob_names:
            return JSONResponse(
                status_code=200,
                content={
                    "message": "No images to delete",
                    "total_deleted": 0,
                    "deleted_blobs": []
                }
            )
        
        # Delete all blobs in batch
        deleted_count = 0
        failed_deletions = []
        
        for blob_name in blob_names:
            try:
                container_client.delete_blob(blob_name)
                deleted_count += 1
            except Exception as e:
                failed_deletions.append({
                    "blob_name": blob_name,
                    "error": str(e)
                })
        
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Successfully deleted {deleted_count} image(s) from container",
                "total_deleted": deleted_count,
                "total_blobs_attempted": len(blob_names),
                "failed_deletions": len(failed_deletions),
                "deleted_blobs": blob_names[:deleted_count],  # List of successfully deleted blob names
                "failed_blobs": failed_deletions if failed_deletions else None
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete all images: {str(e)}"
        )

@app.get("/analyze-images")
async def analyze_images_endpoint():
    """
    Analyze all images in blob storage and generate summaries
    
    Returns:
        JSON response with analysis results for each image and summary file URL
    """
    try:
        result = analyze_all_images(AZURE_CONTAINER_NAME)
        
        return JSONResponse(
            status_code=200,
            content=result
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze images: {str(e)}"
        )


@app.get("/image-summaries/{container_name}")
async def get_image_summaries(container_name: str = AZURE_CONTAINER_NAME):
    """
    Analyze all images in a specific container and generate summaries
    
    Args:
        container_name: Name of the blob container to analyze
        
    Returns:
        JSON response with detailed analysis for each image
    """
    try:
        result = analyze_all_images(container_name)
        
        return JSONResponse(
            status_code=200,
            content={
                "container": container_name,
                "total_images": result["total_images"],
                "analyzed_images": result["analyzed_images"],
                "failed_images": result.get("failed_images", 0),
                "summaries": result["summaries"],
                "summary_file_url": result["summary_file_url"],
                "generated_at": result["generated_at"],
                "message": f"Successfully analyzed {result['analyzed_images']} out of {result['total_images']} images"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve image summaries: {str(e)}"
        )


@app.post("/embed-and-index")
async def embed_and_index():
    """
    Fetch image summaries from blob storage, generate embeddings, 
    and index them in Azure AI Search
    
    Returns:
        JSON response with embedding and indexing results
    """
    try:
        # Validate embeddings configuration
        validate_embeddings_config()
        
        result = embed_and_index_summaries()
        
        return JSONResponse(
            status_code=200,
            content=result
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to embed and index summaries: {str(e)}"
        )


@app.post("/embed-and-index/{summary_file}")
async def embed_and_index_specific(summary_file: str):
    """
    Fetch a specific summary file, generate embeddings, 
    and index them in Azure AI Search
    
    Args:
        summary_file: Name of the summary JSON file in blob storage
        
    Returns:
        JSON response with embedding and indexing results
    """
    try:
        # Validate embeddings configuration
        validate_embeddings_config()
        
        result = embed_and_index_summaries(summary_file_name=summary_file)
        
        return JSONResponse(
            status_code=200,
            content=result
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to embed and index summaries: {str(e)}"
        )


@app.post("/cleanup-duplicate-documents")
async def cleanup_duplicate_documents():
    """
    Remove duplicate documents from Azure AI Search index.
    Keeps only the latest document for each image.
    
    Returns:
        JSON response with cleanup results
    """
    try:
        print("Starting duplicate document cleanup...")
        result = remove_duplicate_documents_from_index()
        
        return JSONResponse(
            status_code=200,
            content={
                **result,
                "message": f"Cleanup completed: removed {result['total_duplicates_removed']} duplicate documents"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cleanup duplicates: {str(e)}"
        )


@app.post("/search-similar-images")
async def search_similar_images_endpoint(file: UploadFile = File(...), top_k: int = 5):
    """
    Upload an image, analyze it, and find similar images in Azure AI Search
    
    Args:
        file: Image file to upload
        top_k: Number of similar images to return (default: 5)
        
    Returns:
        JSON response with similar images and their details
    """
    try:
        # Validate file type
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in ALLOWED_IMAGE_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File type '{file_extension}' is not allowed. Allowed types: {ALLOWED_IMAGE_EXTENSIONS}"
            )
        
        # Validate file size
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File size ({file_size / (1024*1024):.2f} MB) exceeds maximum allowed size of 50 MB"
            )
        
        # Upload image to temporary location in blob storage
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        blob_name = f"search_{timestamp}_{unique_id}_{file.filename}"
        
        # Upload to blob storage
        blob_content = await file.read()
        container_client = get_container_client()
        blob_client = container_client.upload_blob(
            name=blob_name,
            data=blob_content,
            overwrite=True
        )
        image_url = blob_client.url
        
        # Analyze the uploaded image
        print(f"Analyzing uploaded image: {file.filename}")
        analysis_result = analyze_uploaded_image_for_search(image_url)
        
        if analysis_result["status"] == "failed":
            raise HTTPException(
                status_code=500,
                detail=f"Failed to analyze image: {analysis_result['summary'].get('error', 'Unknown error')}"
            )
        
        # Extract summary text for embedding
        summary_data = analysis_result["summary"]
        if isinstance(summary_data, dict):
            text_parts = [
                str(summary_data.get("description", "")),
                str(summary_data.get("main_subjects", "")),
                str(summary_data.get("colors", "")),
                str(summary_data.get("context", ""))
            ]
            text_to_embed = " ".join([t for t in text_parts if t])
        else:
            text_to_embed = str(summary_data)
        
        # Generate embedding for the uploaded image
        print(f"Generating embedding for uploaded image")
        query_embedding = generate_embeddings(text_to_embed)
        
        # Search for similar images
        print(f"Searching for similar images in Azure AI Search")
        search_results = search_similar_images(query_embedding, top_k=top_k)
        
        # Validate similarity using LLM (check if results are truly relevant)
        print(f"Validating similar images with LLM...")
        validation_result = validate_image_similarity_with_llm(
            analysis_result,
            search_results["similar_images"]
        )
        
        # Filter and rank images by LLM relevance
        print(f"Filtering and ranking by LLM relevance...")
        validated_images = filter_and_rank_images(
            search_results["similar_images"],
            validation_result.get("validations", []),
            min_relevance=0.5  # Stricter threshold: 0.5 means must be at least 50% relevant
        )
        
        # Count truly relevant images
        truly_relevant = sum(1 for img in validated_images if img.get("is_truly_similar", False))
        
        return JSONResponse(
            status_code=200,
            content={
                "uploaded_image": {
                    "filename": file.filename,
                    "blob_name": blob_name,
                    "blob_url": get_blob_sas_url(blob_name),
                    "file_size": file_size,
                    "analysis": analysis_result["summary"],
                    "analyzed_at": analysis_result["analyzed_at"]
                },
                "search_results": {
                    "total_candidates_from_vector_search": search_results["total_results"],
                    "total_truly_similar_after_validation": truly_relevant,
                    "similar_images": [
                        {
                            **img,
                            "blob_url": get_blob_sas_url(img.get("image_name", img.get("blob_name", ""))) if img.get("image_name") or img.get("blob_name") else img.get("image_url", ""),
                            "blob_name": img.get("image_name", img.get("blob_name", ""))
                        }
                        for img in validated_images
                    ],
                    "llm_validation_summary": validation_result.get("summary", ""),
                    "validation_status": validation_result.get("validation_status", ""),
                    "search_timestamp": datetime.now().isoformat()
                },
                "message": f"Found {truly_relevant} truly similar images after LLM validation (from {search_results['total_results']} vector search candidates)"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to search similar images: {str(e)}"
        )


@app.get("/search-by-text")
async def search_by_text_endpoint(query: str, top_k: int = 5):
    """
    Search for similar images using text query
    
    Args:
        query: Text query to search for
        top_k: Number of similar images to return (default: 5)
        
    Returns:
        JSON response with similar images
    """
    try:
        if not query or not query.strip():
            raise HTTPException(
                status_code=400,
                detail="Query parameter is required and cannot be empty"
            )
        
        # Generate embedding for the text query
        print(f"Generating embedding for text query: {query}")
        query_embedding = generate_embeddings(query.strip())
        
        # Search for similar images
        print(f"Searching for similar images using text query")
        search_results = search_similar_images(query_embedding, top_k=top_k)
        
        # Validate similarity using LLM (check if results are truly relevant)
        print(f"Validating similar images with LLM...")
        validation_result = validate_image_similarity_with_llm(
            {"summary": {"description": query, "main_subjects": query}},  # Mock analysis for text
            search_results["similar_images"]
        )
        
        # Filter and rank images by LLM relevance
        print(f"Filtering and ranking by LLM relevance...")
        validated_images = filter_and_rank_images(
            search_results["similar_images"],
            validation_result.get("validations", []),
            min_relevance=0.3  # Lower threshold for text search
        )
        
        # Count truly relevant images
        truly_relevant = sum(1 for img in validated_images if img.get("is_truly_similar", False))
        
        return JSONResponse(
            status_code=200,
            content={
                "search_results": [
                    {
                        **img,
                        "blob_url": get_blob_sas_url(img.get("image_name", img.get("blob_name", ""))) if img.get("image_name") or img.get("blob_name") else img.get("image_url", ""),
                        "blob_name": img.get("image_name", img.get("blob_name", ""))
                    }
                    for img in validated_images
                ],
                "total_results": len(validated_images),
                "query": query,
                "message": f"Found {truly_relevant} relevant products for text search"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to search by text: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
