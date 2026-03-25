import os
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

S3_BUCKET = os.getenv("AWS_S3_BUCKET")
S3_PREFIX = os.getenv("AWS_S3_PREFIX", "rag-qa")


def s3_enabled() -> bool:
    """Returns True if S3 is configured (bucket name set)."""
    return bool(S3_BUCKET)


def get_s3_client():
    return boto3.client("s3")


def upload_file_to_s3(local_path: str, s3_key: str) -> None:
    """Upload a local file to S3. Skips silently if S3 is not configured."""
    if not s3_enabled():
        return
    try:
        client = get_s3_client()
        client.upload_file(local_path, S3_BUCKET, s3_key)
    except NoCredentialsError:
        pass  # S3 credentials not available — skip upload


def download_file_from_s3(s3_key: str, local_path: str) -> bool:
    """Download a file from S3 to local_path. Returns False if not available."""
    if not s3_enabled():
        return False
    try:
        client = get_s3_client()
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        client.download_file(S3_BUCKET, s3_key, local_path)
        return True
    except NoCredentialsError:
        return False
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


def s3_key_for(filename: str) -> str:
    return f"{S3_PREFIX}/{filename}"
