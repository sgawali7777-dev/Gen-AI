import os
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Azure Blob Storage Configuration
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_ACCOUNT_URL = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME", "images")

# Validation constants
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


def get_blob_service_client() -> BlobServiceClient:
    """
    Initialize and return Azure Blob Service Client
    
    Supports two authentication methods:
    1. Connection String (AZURE_STORAGE_CONNECTION_STRING)
    2. DefaultAzureCredential with Account URL (for managed identity)
    
    Returns:
        BlobServiceClient: Configured Azure Blob Service Client
        
    Raises:
        ValueError: If neither connection string nor account URL is configured
    """
    if AZURE_STORAGE_CONNECTION_STRING:
        return BlobServiceClient.from_connection_string(
            AZURE_STORAGE_CONNECTION_STRING
        )
    elif AZURE_STORAGE_ACCOUNT_URL:
        return BlobServiceClient(
            account_url=AZURE_STORAGE_ACCOUNT_URL,
            credential=DefaultAzureCredential()
        )
    else:
        raise ValueError(
            "Azure configuration is missing. Please set either "
            "AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_URL in .env file"
        )


def get_container_client():
    """
    Get Azure Blob Container Client
    
    Returns:
        ContainerClient: Configured container client for the specified container
        
    Raises:
        Exception: If container client initialization fails
    """
    blob_service_client = get_blob_service_client()
    return blob_service_client.get_container_client(container=AZURE_CONTAINER_NAME)


def ensure_container_exists():
    """
    Ensure the Azure Blob Storage container exists, create if it doesn't
    
    Returns:
        bool: True if container exists or was created successfully
        
    Raises:
        Exception: If container creation fails
    """
    try:
        container_client = get_container_client()
        props = container_client.get_container_properties()
        # If container exists but is not publicly accessible, set blob-level public access
        try:
            if not getattr(props, 'public_access', None):
                print(f"Container '{AZURE_CONTAINER_NAME}' exists but is not public. Setting public access to 'blob'.")
                container_client.set_container_access_policy(public_access='blob')
        except Exception:
            # Non-fatal: proceed even if we cannot change access policy
            print(f"Warning: Unable to modify access policy for container '{AZURE_CONTAINER_NAME}'")

        print(f"Container '{AZURE_CONTAINER_NAME}' already exists")
        return True
    except Exception as e:
        if "ContainerNotFound" in str(e):
            try:
                blob_service_client = get_blob_service_client()
                # Create container with blob-level public access so uploaded blob URLs are accessible
                blob_service_client.create_container(name=AZURE_CONTAINER_NAME, public_access='blob')
                print(f"Container '{AZURE_CONTAINER_NAME}' created successfully with public access")
                return True
            except Exception as create_error:
                print(f"Failed to create container: {str(create_error)}")
                raise
        else:
            raise


def get_blob_sas_url(blob_name: str, expiry_hours: int = 24, container_name: str = AZURE_CONTAINER_NAME) -> str:
    """
    Generate a SAS URL for accessing a blob with time-limited read permissions
    
    Args:
        blob_name: Name of the blob to generate SAS URL for
        expiry_hours: Number of hours the SAS URL should be valid (default: 24)
        container_name: Name of the container (default: AZURE_CONTAINER_NAME)
        
    Returns:
        str: SAS URL with read permissions that can be used to access the blob from frontend
        
    Raises:
        ValueError: If using DefaultAzureCredential (SAS requires connection string)
    """
    if not blob_name:
        raise ValueError("blob_name cannot be empty")
    
    if not AZURE_STORAGE_CONNECTION_STRING:
        raise ValueError(
            "SAS URL generation requires AZURE_STORAGE_CONNECTION_STRING. "
            "DefaultAzureCredential does not support SAS token generation."
        )
    
    # Parse account name from connection string
    conn_str = AZURE_STORAGE_CONNECTION_STRING
    account_name = None
    account_key = None
    
    for part in conn_str.split(';'):
        if part.startswith('AccountName='):
            account_name = part.split('=', 1)[1]
        elif part.startswith('AccountKey='):
            account_key = part.split('=', 1)[1]
    
    if not account_name or not account_key:
        raise ValueError("Could not parse account name or key from connection string")
    
    try:
        # Generate SAS token
        sas_token = generate_blob_sas(
            account_name=account_name,
            container_name=container_name,
            blob_name=blob_name,
            account_key=account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=expiry_hours)
        )
        
        # Construct SAS URL
        sas_url = f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}"
        return sas_url
    except Exception as e:
        print(f"Warning: Failed to generate SAS URL for {blob_name}: {str(e)}")
        # Fallback to plain URL if SAS generation fails
        return f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}"


def validate_configuration() -> bool:
    """
    Validate Azure configuration
    
    Returns:
        bool: True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    if not AZURE_STORAGE_CONNECTION_STRING and not AZURE_STORAGE_ACCOUNT_URL:
        raise ValueError(
            "Azure Blob Storage is not configured. "
            "Please set AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_URL in .env file"
        )
    
    if not AZURE_CONTAINER_NAME:
        raise ValueError("AZURE_CONTAINER_NAME is not set in .env file")
    
    return True
