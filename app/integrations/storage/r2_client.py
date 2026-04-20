import boto3
from botocore.exceptions import ClientError
from app.core.config import settings
from app.core.logging import logger

class R2Client:
    def __init__(self):
        self.s3 = None
        self.bucket = settings.R2_BUCKET
        
        # Check if R2 is configured
        if "your_id" in settings.R2_ENDPOINT_URL or not settings.R2_ENDPOINT_URL.startswith("http"):
            logger.warning("R2 storage not configured properly. Using placeholder client.")
            return

        try:
            self.s3 = boto3.client(
                "s3",
                endpoint_url=settings.R2_ENDPOINT_URL,
                aws_access_key_id=settings.R2_ACCESS_KEY_ID,
                aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
                region_name="auto",  # R2 default
            )
        except Exception as e:
            logger.error(f"Failed to initialize R2 client: {e}")
            self.s3 = None

    def upload_file(self, file_path: str, object_name: str) -> bool:
        if not self.s3:
            logger.warning("R2 storage not configured. Skipping upload.")
            return False
        try:
            self.s3.upload_file(file_path, self.bucket, object_name)
            logger.info(f"Uploaded {file_path} to R2 as {object_name}")
            return True
        except ClientError as e:
            logger.error(f"Failed to upload to R2: {e}")
            return False

    def get_download_url(self, object_name: str, expires_in: int = 3600) -> str:
        if not self.s3:
            return f"file://{object_name}" # Local fallback or dummy URL
        try:
            url = self.s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": object_name},
                ExpiresIn=expires_in,
            )
            return url
        except ClientError as e:
            logger.error(f"Failed to generate R2 URL: {e}")
            return ""

storage_client = R2Client()
