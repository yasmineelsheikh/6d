"""
RunPod serverless handler for Cosmos-Transfer2.5 inference.

This handler processes jobs from RunPod endpoints by:
1. Reading input files (video, prompt) from local paths
2. Running Cosmos inference
3. Returning output file paths

Expected job input format:
{
    "input": {
        "video_path": "/path/to/video.mp4",
        "prompt_path": "/path/to/prompt.txt",
        "output_path": "/path/to/output/",  # Optional, defaults to /tmp/cosmos_output
        "params": {
            "name": "job_name",
            "guidance": 3,
            "edge": {"control_weight": 0.5},
            "vis": {"control_weight": 0.2}
        }  # Optional, will use defaults if not provided
    }
}
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict

import runpod


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler function for Cosmos inference.
    
    Args:
        job: RunPod job dictionary with 'input' key containing:
            - video_path: Path to input video (local path)
            - prompt_path: Path to prompt file (local path)
            - output_path: Path for output directory (local path, optional)
            - params: Optional inference parameters dict
            
    Returns:
        Dictionary with 'output' key containing results or 'error' key if failed
    """
    try:
        job_input = job.get("input", {})
        
        # Extract input paths
        video_path = job_input.get("video_path")
        prompt_path = job_input.get("prompt_path")
        output_path = job_input.get("output_path", "/tmp/cosmos_output")
        params = job_input.get("params", {})
        
        if not video_path or not prompt_path:
            return {
                "error": "Missing required input: 'video_path' and 'prompt_path' are required"
            }
        
        # Validate input files exist
        if not os.path.exists(video_path):
            return {
                "error": f"Video file not found: {video_path}"
            }
        
        if not os.path.exists(prompt_path):
            return {
                "error": f"Prompt file not found: {prompt_path}"
            }
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Copy input files to temp directory
            local_video = temp_path / "input_video.mp4"
            local_prompt = temp_path / "prompt.txt"
            local_params = temp_path / "params.json"
            local_output_dir = temp_path / "output"
            local_output_dir.mkdir(exist_ok=True)
            
            # Copy video and prompt files
            shutil.copy(video_path, str(local_video))
            shutil.copy(prompt_path, str(local_prompt))
            
            # Create inference parameters JSON
            inference_params = {
                "name": params.get("name", "cosmos_inference"),
                "prompt_path": str(local_prompt),
                "video_path": str(local_video),
                "guidance": params.get("guidance", 3),
                "edge": params.get("edge", {"control_weight": 0.5}),
                "vis": params.get("vis", {"control_weight": 0.2}),
            }
            
            # Add other optional params if provided
            if "depth" in params:
                inference_params["depth"] = params["depth"]
            if "seg" in params:
                inference_params["seg"] = params["seg"]
            
            # Write params JSON
            with open(local_params, "w") as f:
                json.dump(inference_params, f, indent=2)
            
            # Run Cosmos inference
            # Note: This assumes cosmos-transfer2.5 is installed and available
            cosmos_script = "/workspace/cosmos-transfer2.5/examples/inference.py"
            if not os.path.exists(cosmos_script):
                # Try relative path
                cosmos_script = "cosmos-transfer2.5/examples/inference.py"
            
            command = [
                "python",
                cosmos_script,
                "-i", str(local_params),
                "--setup.output_dir", str(local_output_dir),
            ]
            
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd="/workspace" if os.path.exists("/workspace") else os.getcwd()
            )
            
            if result.returncode != 0:
                return {
                    "error": f"Cosmos inference failed: {result.stderr}",
                    "stdout": result.stdout,
                }
            
            # Find output video file
            output_files = list(local_output_dir.glob("*.mp4"))
            if not output_files:
                return {
                    "error": "No output video file generated",
                    "stdout": result.stdout,
                }
            
            output_video = output_files[0]
            
            # Copy output to final output directory
            final_output = Path(output_path) / output_video.name
            shutil.copy(str(output_video), str(final_output))
            final_output_path = str(final_output)
            
            return {
                "output": {
                    "output_path": final_output_path,
                    "output_file": output_video.name,
                    "status": "success",
                    "stdout": result.stdout,
                }
            }
    
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


# Start the RunPod serverless worker
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
