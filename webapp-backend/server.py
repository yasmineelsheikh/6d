#server.py file used by gpu

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import tempfile
import os
import subprocess
import json
import boto3
from pathlib import Path
from datetime import datetime
from botocore.exceptions import BotoCoreError, ClientError
import sys

# Add parent directory to path to import generate_prompt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_prompt import generate_prompt_variations

S3_BUCKET = "6d-temp-storage"
S3_REGION = "us-west-2"
s3 = boto3.client("s3", region_name=S3_REGION)

app = FastAPI()

class CosmosRequest(BaseModel):
    dataset_name: str

def download_folder_from_s3(s3_prefix: str, local_path: str) -> None:
    os.makedirs(local_path, exist_ok=True)

    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=s3_prefix)
    for page in pages:
        if 'Contents' not in page:
            continue
        for obj in page['Contents']:
            s3_key = obj['Key']

            if s3_key == s3_prefix or s3_key.endswith('/'):
                continue
            relative_path = s3_key[len(s3_prefix):].lstrip('/')
            local_file_path = os.path.join(local_path, relative_path)

            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            try:
                s3.download_file(S3_BUCKET, s3_key, local_file_path)
                print(f"Downloaded s3://{S3_BUCKET}/{s3_key} to {local_file_path}")
            except (BotoCoreError, ClientError) as e:
                print(f"Error downloading {s3_key}: {e}")
                raise

## @app.post("/run-json")
## def run_cosmos(req: CosmosRequest):
##     dataset_name = req.dataset_name
##     s3_prefix = f"input_data/{dataset_name}"
##     # Create temporary directory for downloaded files
##     with tempfile.TemporaryDirectory() as temp_dir:
##         local_data_path = os.path.join(temp_dir, "data")
##         try:
##             # Download dataset folder from S3
##             print(f"Downloading dataset from s3://{S3_BUCKET}/{s3_prefix}")
##             download_folder_from_s3(s3_prefix, local_data_path)
##             # Find prompt.txt file
##             original_prompt_path = os.path.join(local_data_path, "prompt.txt")
##             video_path = os.path.join(local_data_path, "videos", "episode_videos", "episode_01.mp4")
##             if not os.path.exists(original_prompt_path):
##                 raise FileNotFoundError(f"Prompt file not found at {original_prompt_path}")
##             
##             # Read the original prompt
##             with open(original_prompt_path, 'r') as f:
##                 original_prompt = f.read()
##             
##             n = 5
##
##             for i in range(len(n)):
##                 # Generate prompt variation using generate_prompt module
##                 varied_prompt = generate_prompt_variation(original_prompt)
##                 
##                 # Save the varied prompt to a new file
##                 varied_prompt_path = os.path.join(temp_dir, "prompt_variation.txt")
##                 with open(varied_prompt_path, 'w') as f:
##                     f.write(varied_prompt)
##                 
##                 # Find all video files
##                 video_files = []
##                 for root, dirs, files in os.walk(local_data_path):
##                     for file in files:
##                         if file.endswith('.mp4'):
##                             video_files.append(os.path.join(root, file))
##                 if not video_files:
##                     raise ValueError(f"No video files found in downloaded dataset")
##                 # Process each video
##                 output_urls = []
## #            for i, video_path in enumerate(video_files):
## #               episode_index = Path(video_path).stem
## #              # Prepare JSON input for Cosmos
##                 cosmos_input_spec = os.path.join(temp_dir, "input_spec.json")
##                 cosmos_input = {
##                             "name": "stack_cups",
##                             "prompt_path": f"{varied_prompt_path}",
##                             "video_path": f"{video_path}",
##                             "guidance": 3,
##                             "depth": {"control_weight": 1.0},
##                             "seg": {"control_weight": 1.0},
##                             "edge": {"control_weight": 0.5},
##                             "vis": {"control_weight": 0.2}
##                             }
##                 with open(cosmos_input_spec, "w") as f:
##                     json.dump(cosmos_input, f, indent=2)
##                     # Create output directory
##         #          output_dir = os.path.join(temp_dir, "outputs")
##         #           os.makedirs(output_dir, exist_ok=True)
##
##         #           cosmos_script = "/workspace/cosmos-transfer2.5/examples/inference.py"
##         #          if not os.path.exists(cosmos_script):
##         #             cosmos_script = "examples/inference.py"
##
##             #        cosmos_dir = "/workspace/cosmos-transfer2.5"
##             #       if not os.path.exists(cosmos_dir):
##             #          cosmos_dir = os.getcwd()
##                     #command = [
##                     # "torchrun" 
##                     # "--nproc_per_node=8"
##                     # "--master_port=12341"
##                     # "examples/inference.py"
##                     # "-i"
##                     # "assets/robot_example/depth/robot_depth_spec.json"
##                     # "-o"
##                     # "outputs/depth"]
##
##                 command = [
##                             "python",
##                             "examples/inference.py",
##                             "-i", f"{cosmos_input_spec}",
##                             "-o", "outputs/test"]
##
##                 try:
##                     result = subprocess.run(
##                                 command
##                                 )
##
##
##                 except subprocess.CalledProcessError as e:
##                     raise RuntimeError(f"Cosmos inference failed: {e.stderr}\nCommand: {' '.join(command)}")
##
##             #output_files = list(Path(output_dir).glob("*.mp4"))
##            # if output_files:
##             #        output_video = output_files[0]
##              #       # Upload output to S3
##               #      timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
##                #     s3_key = f"outputs/{dataset_name}/{episode_index}_{timestamp}.mp4"
##                 #    try:
##                  #       s3.upload_file(str(output_video), S3_BUCKET, s3_key)
##                   #      s3_url = f"s3://{S3_BUCKET}/{s3_key}"
##                    #     output_urls.append(s3_url)
##                   #  except (BotoCoreError, ClientError) as e:
##                    #     print(f"Error uploading output to S3: {e}")
##
##            # if not output_urls:
##            #     raise RuntimeError("No output videos were generated")
##            # return {"s3_url": output_urls[0] if len(output_urls) == 1 else output_urls}
##            
##         except Exception as e:
##             raise HTTPException(status_code=500, detail=f"Error processing Cosmos inference: {str(e)}")

@app.post("/run-json-from-prompts")
def run_cosmos_from_prompts(req: CosmosRequest):
    """
    Similar to run_cosmos but iterates through already generated prompts in the prompts folder
    instead of generating new prompts in each iteration.
    """
    dataset_name = req.dataset_name
    s3_prefix = f"input_data/{dataset_name}"
    # Create temporary directory for downloaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        local_data_path = os.path.join(temp_dir, "data")
        try:
            # Download dataset folder from S3
            print(f"Downloading dataset from s3://{S3_BUCKET}/{s3_prefix}")
            download_folder_from_s3(s3_prefix, local_data_path)
            
            # Find the prompts folder (should be at input_data/{dataset_name}/prompts/{timestamp}_{uuid}/)
            prompts_base_path = os.path.join(local_data_path, "prompts")
            if not os.path.exists(prompts_base_path):
                raise FileNotFoundError(f"Prompts folder not found at {prompts_base_path}")
            
            # Find metadata.json file in the prompts folder
            metadata_path = None
            for root, dirs, files in os.walk(prompts_base_path):
                if "metadata.json" in files:
                    metadata_path = os.path.join(root, "metadata.json")
                    break
            
            if not metadata_path or not os.path.exists(metadata_path):
                raise FileNotFoundError(f"metadata.json not found in {prompts_base_path}")
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            if not metadata:
                raise ValueError(f"metadata.json is empty or invalid")
            
            print(f"Found {len(metadata)} prompt variations in metadata")
            
            # Process each entry in metadata
            output_urls = []
            for i, entry in enumerate(metadata):
                video_relative_path = entry.get("video_path", "")
                txt_relative_path = entry.get("txt_file_path", "")
                axis = entry.get("axis", "Unknown")
                
                # Resolve absolute paths from metadata relative paths
                video_path = os.path.join(local_data_path, video_relative_path)
                prompt_file_path = os.path.join(local_data_path, txt_relative_path)
                
                if not os.path.exists(video_path):
                    print(f"Warning: Video not found at {video_path}, skipping entry {i+1}")
                    continue
                
                if not os.path.exists(prompt_file_path):
                    print(f"Warning: Prompt file not found at {prompt_file_path}, skipping entry {i+1}")
                    continue
                
                print(f"Processing variation {i+1}/{len(metadata)}: {prompt_file_path} (axis: {axis}, video: {video_relative_path})")
                
                # Map axis names to spec template files
                axis_to_spec = {
                    "Lighting": "indoor_lighting_spec.json",
                    "Objects": "objects_spec.json",
                    "Color/Material": "color_texture_spec.json",
                    "Materials": "color_texture_spec.json",  # Alias for Color/Material
                    "Weather": "weather_spec.json",
                    "Road Surface": "road_surface_spec.json",
                    "Outdoor Lighting": "outdoor_lighting_spec.json",
                }
                
                # Get spec template file name for this axis
                spec_filename = axis_to_spec.get(axis)
                if not spec_filename:
                    print(f"Warning: Unknown axis '{axis}', using default spec. Available axes: {list(axis_to_spec.keys())}")
                    spec_filename = "objects_spec.json"  # Default fallback
                
                # Load spec template from spec_templates directory
                spec_templates_dir = Path(__file__).parent / "spec_templates"
                spec_file_path = spec_templates_dir / spec_filename
                
                if not spec_file_path.exists():
                    raise FileNotFoundError(f"Spec template not found: {spec_file_path}")
                
                # Load the spec template
                with open(spec_file_path, 'r') as f:
                    cosmos_input = json.load(f)
                
                # Override with prompt_path and video_path from metadata
                cosmos_input["name"] = dataset_name
                cosmos_input["prompt_path"] = f"{prompt_file_path}"
                cosmos_input["video_path"] = f"{video_path}"
                
                # Prepare JSON input for Cosmos
                cosmos_input_spec = os.path.join(temp_dir, f"input_spec_{i}.json")
                with open(cosmos_input_spec, "w") as f:
                    json.dump(cosmos_input, f, indent=2)
                
                #command = [
                #    "python",
                #    "examples/inference.py",
                #    "-i", f"{cosmos_input_spec}",
                #    "-o", f"outputs/test_{i}"
                #]

                command = [
                    "torchrun",
                    "--nproc_per_node=8",
                    "--master_port=12341",
                    "examples/inference.py",
                    "-i",
                    f"{cosmos_input_spec}",
                    "-o",
                    f"outputs/test_{i}",
                ]
                # Example reference:
                # torchrun --nproc_per_node=8 --master_port=12341 examples/inference.py \
                #   -i assets/robot_example/depth/robot_depth_spec.json -o outputs/depth
                
                try:
                    result = subprocess.run(command, check=True)
                    print(f"Completed processing variation {i+1}/{len(metadata)} (axis: {axis})")
                except subprocess.CalledProcessError as e:
                    print(f"Error processing variation {i+1}: {e}")
                    continue  # Continue with next variation if this one fails
                
                # After successful Cosmos run, upload any generated videos in the output folder to S3
                try:
                    output_dir = Path(f"outputs/test_{i}")
                    if not output_dir.exists():
                        print(f"Warning: Output directory not found at {output_dir}, skipping upload for variation {i+1}")
                        continue
                    
                    output_files = list(output_dir.glob("*.mp4"))
                    if not output_files:
                        print(f"Warning: No .mp4 files found in {output_dir}, skipping upload for variation {i+1}")
                        continue
                    
                    for output_video in output_files:
                        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                        s3_key = f"outputs/{dataset_name}/{output_video.stem}_{timestamp}.mp4"
                        try:
                            s3.upload_file(str(output_video), S3_BUCKET, s3_key)
                            s3_url = f"s3://{S3_BUCKET}/{s3_key}"
                            output_urls.append(s3_url)
                            print(f"Uploaded Cosmos output to {s3_url}")
                        except (BotoCoreError, ClientError) as upload_err:
                            print(f"Error uploading output to S3: {upload_err}")
                except Exception as upload_wrapper_err:
                    print(f"Unexpected error while handling Cosmos outputs for variation {i+1}: {upload_wrapper_err}")
                    continue
            
            if not output_urls:
                # Return success even if no outputs were uploaded (since upload code is commented out)
                return {"success": True, "processed_variations": len(metadata), "message": "All variations processed"}
            
            return {"s3_url": output_urls[0] if len(output_urls) == 1 else output_urls}
           
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing Cosmos inference: {str(e)}")