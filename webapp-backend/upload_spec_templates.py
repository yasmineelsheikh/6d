#!/usr/bin/env python3
"""
Upload spec templates to S3 for use by the RunPod GPU server.
"""
import boto3
import os
from pathlib import Path

S3_BUCKET = "6d-temp-storage"
S3_PREFIX = "spec_templates"

def upload_spec_templates():
    """Upload all spec template files to S3."""
    s3 = boto3.client("s3")
    
    # Get the spec_templates directory
    spec_dir = Path(__file__).parent / "spec_templates"
    
    if not spec_dir.exists():
        print(f"Error: spec_templates directory not found at {spec_dir}")
        return
    
    # Upload each JSON file
    for json_file in spec_dir.glob("*.json"):
        s3_key = f"{S3_PREFIX}/{json_file.name}"
        
        print(f"Uploading {json_file.name} to s3://{S3_BUCKET}/{s3_key}")
        
        with open(json_file, 'rb') as f:
            s3.put_object(
                Bucket=S3_BUCKET,
                Key=s3_key,
                Body=f.read()
            )
        
        print(f"✓ Uploaded {json_file.name}")
    
    print(f"\nAll spec templates uploaded to s3://{S3_BUCKET}/{S3_PREFIX}/")

if __name__ == "__main__":
    upload_spec_templates()
