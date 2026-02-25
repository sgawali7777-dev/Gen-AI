"""
Image Analyzer Module
Handles image analysis and summary generation using Azure OpenAI
"""

import os
import base64
import json
from datetime import datetime
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

# Azure Storage Configuration
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME", "images")
SUMMARIES_CONTAINER_NAME = os.getenv("AZURE_SUMMARIES_CONTAINER_NAME", "image-summaries")


def get_openai_client():
    """Initialize Azure OpenAI client"""
    from openai import AzureOpenAI
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


def get_image_summary(image_url: str, image_name: str) -> dict:
    """
    Generate summary for an image using Azure OpenAI Vision
    
    Args:
        image_url: URL of the image in Azure Blob Storage
        image_name: Name/path of the image blob
        
    Returns:
        dict: Contains image name, url, summary, and analysis details
    """
    try:
        client = get_openai_client()
        
        # Use Azure OpenAI vision capabilities to analyze the image
        message = client.chat.completions.create(
            model=AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please analyze this image and provide a detailed summary including: 1) Main objects/subjects visible, 2) Key colors and composition, 3) Overall description, 4) Any text visible in the image, 5) Context or use case. Format the response in JSON with these fields: main_subjects, colors, description, visible_text, context."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        
        # Parse the response - correct way to access Azure OpenAI response
        response_text = message.choices[0].message.content
        
        # Try to parse as JSON, if fails, wrap it
        try:
            summary_data = json.loads(response_text)
        except json.JSONDecodeError:
            summary_data = {
                "description": response_text,
                "main_subjects": "See description",
                "colors": "N/A",
                "visible_text": "N/A",
                "context": "N/A"
            }
        
        return {
            "image_name": image_name,
            "image_url": image_url,
            "summary": summary_data,
            "analyzed_at": datetime.now().isoformat(),
            "status": "success"
        }
        
    except Exception as e:
        return {
            "image_name": image_name,
            "image_url": image_url,
            "summary": {"error": str(e)},
            "analyzed_at": datetime.now().isoformat(),
            "status": "failed"
        }


def list_images_in_blob(container_name: str = AZURE_CONTAINER_NAME) -> list:
    """
    List all image blobs in the container
    
    Args:
        container_name: Name of the blob container
        
    Returns:
        list: List of blob names
    """
    try:
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(
            container=container_name
        )
        
        blobs = container_client.list_blobs()
        image_blobs = [blob.name for blob in blobs if blob.name.lower().endswith(
            ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')
        )]
        
        return image_blobs
    except Exception as e:
        raise Exception(f"Failed to list images in blob storage: {str(e)}")


def get_blob_url(blob_name: str, container_name: str = AZURE_CONTAINER_NAME) -> str:
    """
    Get the URL of a blob
    
    Args:
        blob_name: Name of the blob
        container_name: Name of the blob container
        
    Returns:
        str: URL of the blob
    """
    blob_service_client = get_blob_service_client()
    blob_client = blob_service_client.get_blob_client(
        container=container_name,
        blob=blob_name
    )
    return blob_client.url


def store_summary_in_blob(summaries: list, summary_file_name: str = None) -> str:
    """
    Store image summaries in blob storage as JSON
    
    Args:
        summaries: List of image summaries
        summary_file_name: Custom file name for summaries (optional)
        
    Returns:
        str: URL of the stored summary file
    """
    try:
        if not summary_file_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file_name = f"summaries_{timestamp}.json"
        
        # Ensure summaries container exists (create if it doesn't)
        blob_service_client = get_blob_service_client()
        
        try:
            blob_service_client.create_container(name=SUMMARIES_CONTAINER_NAME)
        except Exception as e:
            # Container already exists, which is fine - just continue
            if "ContainerAlreadyExists" not in str(e):
                raise
        
        # Store summaries as JSON
        container_client = blob_service_client.get_container_client(
            container=SUMMARIES_CONTAINER_NAME
        )
        
        summaries_json = json.dumps(summaries, indent=2, default=str)
        
        blob_client = container_client.upload_blob(
            name=summary_file_name,
            data=summaries_json,
            overwrite=True
        )
        
        return blob_client.url
        
    except Exception as e:
        raise Exception(f"Failed to store summaries in blob storage: {str(e)}")


def analyze_all_images(container_name: str = AZURE_CONTAINER_NAME) -> dict:
    """
    Analyze all images in a container and generate summaries
    
    Args:
        container_name: Name of the blob container
        
    Returns:
        dict: Contains all summaries and summary file URL
    """
    try:
        # List all images
        image_blobs = list_images_in_blob(container_name)
        
        if not image_blobs:
            return {
                "total_images": 0,
                "analyzed_images": 0,
                "summaries": [],
                "summary_file_url": None,
                "message": "No images found in blob storage"
            }
        
        # Analyze each image
        summaries = []
        for blob_name in image_blobs:
            try:
                image_url = get_blob_url(blob_name, container_name)
                summary = get_image_summary(image_url, blob_name)
                summaries.append(summary)
            except Exception as e:
                summaries.append({
                    "image_name": blob_name,
                    "status": "failed",
                    "error": str(e)
                })
        
        # Store summaries in blob storage
        summary_file_url = store_summary_in_blob(summaries)
        
        # Count successful analyses
        successful = sum(1 for s in summaries if s.get("status") == "success")
        
        return {
            "total_images": len(image_blobs),
            "analyzed_images": successful,
            "failed_images": len(image_blobs) - successful,
            "summaries": summaries,
            "summary_file_url": summary_file_url,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise Exception(f"Failed to analyze images: {str(e)}")


def analyze_uploaded_image_for_search(image_url: str) -> dict:
    """
    Analyze a user-uploaded image and return summary for search purposes
    
    Args:
        image_url: URL of the uploaded image (from blob storage)
        
    Returns:
        dict: Summary data for the image
    """
    try:
        client = get_openai_client()
        
        # Analyze the image using Azure OpenAI Vision
        message = client.chat.completions.create(
            model=AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please analyze this image and provide a detailed summary including: 1) Main objects/subjects visible, 2) Key colors and composition, 3) Overall description, 4) Any text visible in the image, 5) Context or use case. Format the response in JSON with these fields: main_subjects, colors, description, visible_text, context."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        
        # Parse response
        response_text = message.choices[0].message.content
        
        try:
            summary_data = json.loads(response_text)
        except json.JSONDecodeError:
            summary_data = {
                "description": response_text,
                "main_subjects": "See description",
                "colors": "N/A",
                "visible_text": "N/A",
                "context": "N/A"
            }
        
        return {
            "summary": summary_data,
            "status": "success",
            "analyzed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "summary": {"error": str(e)},
            "status": "failed",
            "analyzed_at": datetime.now().isoformat()
        }


def validate_image_similarity_with_llm(uploaded_image_analysis: dict, similar_images: list) -> dict:
    """
    Use LLM to validate and rank similar images for relevance
    Filters out irrelevant images and ranks truly similar ones
    
    Args:
        uploaded_image_analysis: Analysis of the uploaded/query image
        similar_images: List of candidate similar images with summaries
        
    Returns:
        dict: Validated and ranked similar images with relevance scores
    """
    try:
        client = get_openai_client()
        
        # Build comparison prompt
        uploaded_summary = uploaded_image_analysis.get("summary", {})
        if isinstance(uploaded_summary, dict):
            uploaded_desc = f"""
Main Subjects: {uploaded_summary.get('main_subjects', 'N/A')}
Colors: {uploaded_summary.get('colors', 'N/A')}
Description: {uploaded_summary.get('description', 'N/A')}
Context: {uploaded_summary.get('context', 'N/A')}
"""
        else:
            uploaded_desc = str(uploaded_summary)
        
        # Format similar images for comparison
        similar_images_text = ""
        for idx, img in enumerate(similar_images, 1):
            summary = img.get("summary", "")
            img_name = img.get("image_name", f"Image {idx}")
            similar_images_text += f"""
Image {idx} ({img_name}):
Summary: {summary}
Vector Similarity Score: {img.get('similarity_score', 0):.2f}
---
"""
        
        # Create validation prompt with explicit scoring criteria
        # PRIORITY: Object Type > Color/Size/Material > Other variations
        validation_prompt = f"""You are an image similarity expert. Your PRIMARY task is to identify if images belong to the SAME OBJECT TYPE, regardless of color or size variations.

UPLOADED IMAGE ANALYSIS:
{uploaded_desc}

CANDIDATE SIMILAR IMAGES FROM DATABASE:
{similar_images_text}

CRITICAL INSTRUCTIONS FOR EVALUATION:
PRIMARY ASSESSMENT (Most Important):
- Is the candidate image the SAME OBJECT TYPE as the uploaded image?
  Example: If uploaded is a bottle, all bottles (red, blue, glass, plastic) are matches.
- Same object type = Accept (score 0.75+), color/size/material differences are SECONDARY
- Different object type = Reject (score 0.3-0.4)

SECONDARY ASSESSMENT (Less Important):
- Only after confirming same object type, evaluate visual variations
- Color differences = NOT a reason to reject (both still the same object)
- Size differences = NOT a reason to reject (both still the same object)
- Material differences = NOT a reason to reject (both still the same object)

TASK:
For EACH image, assess:
1. Does it have the SAME OBJECT TYPE as the uploaded image? (PRIMARY - Most critical)
2. How closely does it match in terms of variations? (SECONDARY - Color, size, material)
3. Would these images logically belong together in a collection of the SAME OBJECT TYPE?

Scoring Guide (COLOR/SIZE/MATERIAL ARE SECONDARY):
- 0.9-1.0: Exact match (same object type, same color, size, and material)
- 0.80-0.89: Same object type, DIFFERENT color (e.g., red bottle vs blue bottle) - ACCEPT despite color difference
- 0.70-0.79: Same object type, DIFFERENT size (e.g., large vs small of same object) - ACCEPT despite size difference
- 0.60-0.69: Same object type, DIFFERENT material (e.g., glass vs plastic bottle) - ACCEPT despite material difference
- 0.50-0.59: Same object type, MULTIPLE variations - ACCEPT (still same object)
- 0.3-0.4: Related but DIFFERENT object types (e.g., bottle vs cup - similar but distinct) - Consider carefully
- 0.0-0.29: COMPLETELY DIFFERENT objects (e.g., bottle vs car) - REJECT

CRITICAL REMINDERS:
- If it's the same OBJECT TYPE (e.g., all bottles, all chairs, all dogs), score it HIGH (0.7+)
- Do NOT penalize for color being different - color is just a characteristic, not the defining factor
- Do NOT penalize for size being different - the object type is still the same
- Only REJECT if it's a fundamentally different object type

Return ONLY valid JSON with this EXACT structure:
{{
  "validations": [
    {{
      "image_name": "filename.jpg",
      "relevance_score": 0.85,
      "is_relevant": true,
      "reasoning": "Specific reason: Same object type as uploaded image (e.g., both are bottles, color difference is secondary)"
    }},
    {{
      "image_name": "another_file.jpg", 
      "relevance_score": 0.25,
      "is_relevant": false,
      "reasoning": "Specific reason: Different object type (bottle vs cup - fundamentally different objects)"
    }}
  ],
  "summary": "Overall assessment summary"
}}

IMPORTANT: is_relevant must be FALSE if relevance_score < 0.3 (only if COMPLETELY different object types)
Do NOT include any text before or after the JSON."""
        
        # Call LLM for validation
        response = client.chat.completions.create(
            model=AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages=[
                {
                    "role": "user",
                    "content": validation_prompt
                }
            ],
            max_tokens=2000,
            temperature=0.3  # Lower temperature for consistent evaluation
        )
        
        # Parse response
        response_text = response.choices[0].message.content
        
        try:
            validation_data = json.loads(response_text)
        except json.JSONDecodeError:
            # If parsing fails, return original images with warning
            return {
                "validations": [
                    {
                        "image_name": img.get("image_name", ""),
                        "relevance_score": min(img.get("similarity_score", 0), 0.5),  # Cap at 0.5 if uncertain
                        "is_relevant": img.get("similarity_score", 0) > 0.75,
                        "reasoning": "LLM validation parsing failed, using conservative vector score"
                    }
                    for img in similar_images
                ],
                "summary": "Using vector similarity scores only",
                "validation_status": "fallback"
            }
        
        return {
            "validations": validation_data.get("validations", []),
            "summary": validation_data.get("summary", ""),
            "validation_status": "success"
        }
        
    except Exception as e:
        # Return conservative scores if validation fails
        return {
            "validations": [
                {
                    "image_name": img.get("image_name", ""),
                    "relevance_score": min(img.get("similarity_score", 0), 0.5),
                    "is_relevant": False,
                    "reasoning": f"Validation error - marking as potentially irrelevant: {str(e)}"
                }
                for img in similar_images
            ],
            "summary": f"LLM validation failed: {str(e)}",
            "validation_status": "failed"
        }


def filter_and_rank_images(similar_images: list, validations: list, min_relevance: float = 0.5) -> list:
    """
    Filter images by relevance and rank them
    Only returns truly relevant images (those passing LLM validation)
    
    Args:
        similar_images: Original similar images from search
        validations: LLM validations
        min_relevance: Minimum relevance score to include (0-1, default: 0.5)
        
    Returns:
        list: Filtered and ranked images, highest relevance first
    """
    # Create mapping of image names to validations
    validation_map = {}
    for val in validations:
        validation_map[val.get("image_name", "")] = val
    
    # Merge and filter - ONLY include truly relevant images
    ranked_images = []
    for img in similar_images:
        img_name = img.get("image_name", "")
        validation = validation_map.get(img_name, {})
        
        relevance_score = validation.get("relevance_score", 0)
        is_relevant = validation.get("is_relevant", False)  # Default to False if not specified
        reasoning = validation.get("reasoning", "No validation reasoning provided")
        
        # STRICT FILTERING: Only include if:
        # 1. LLM explicitly marked as relevant, AND
        # 2. Relevance score meets minimum threshold
        if is_relevant and relevance_score >= min_relevance:
            ranked_images.append({
                **img,
                "llm_relevance_score": relevance_score,
                "llm_reasoning": reasoning,
                "is_truly_similar": True
            })
        else:
            # For debugging: include filtered out images with reason
            ranked_images.append({
                **img,
                "llm_relevance_score": relevance_score,
                "llm_reasoning": reasoning,
                "is_truly_similar": False,
                "filter_reason": f"Filtered out: is_relevant={is_relevant}, score={relevance_score:.2f}, min_threshold={min_relevance}"
            })
    
    # Separate truly similar from filtered
    truly_similar = [img for img in ranked_images if img.get("is_truly_similar", False)]
    
    # Sort truly similar by LLM relevance score (highest first)
    truly_similar.sort(key=lambda x: x.get("llm_relevance_score", 0), reverse=True)
    
    return truly_similar
