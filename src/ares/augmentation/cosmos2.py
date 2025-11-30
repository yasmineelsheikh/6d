import os
import time
import uuid
import json
import logging
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
from openai import OpenAI

from ares.models.grounding import GroundingAnnotator
from ares.utils.image_utils import split_video_to_frames, choose_and_preprocess_frames
from ares.utils.runpod_executor import RunPodExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CosmosAugmentor:
    """
    Implements the Cosmos Data Augmentation Strategy (6.2.2).

    """
    
    def __init__(
        self,
        use_gpu: bool = True,
        execution_mode: str = "auto",
    ):
        """
        Initialize CosmosAugmentor.
        
        Args:
            use_gpu: Whether to use GPU for local execution
            execution_mode: Execution mode - 'local', 'runpod', or 'auto'
                           'auto' will use RunPod if RUNPOD_API_KEY is set, otherwise local
        """
        self.initial_caption = ""
        self.guidance_scale = 3
        
        # Determine execution mode
        if execution_mode == "auto":
            # Auto-detect based on RUNPOD_API_KEY
            self.execution_mode = "runpod" if os.environ.get("RUNPOD_API_KEY") else "local"
        else:
            self.execution_mode = execution_mode
        
        logger.info(f"Cosmos execution mode: {self.execution_mode}")
        
        # Initialize RunPod executor if needed
        self.runpod_executor = None
        if self.execution_mode == "runpod":
            try:
                storage_backend = "r2" if os.environ.get("R2_BUCKET_NAME") else "s3"
                self.runpod_executor = RunPodExecutor(storage_backend=storage_backend)
                logger.info("RunPod executor initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize RunPod executor: {e}. Falling back to local execution.")
                self.execution_mode = "local"
        
        # Initialize GroundingAnnotator for robot segmentation (local only)
        device = "cuda" if use_gpu else "cpu"
        try:
            self.grounding_annotator = GroundingAnnotator(
                detector_id="IDEA-Research/grounding-dino-tiny",
                segmenter_id="facebook/sam-vit-base",
                device=device,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize GroundingAnnotator: {e}. Segmentation will be skipped.")
            self.grounding_annotator = None
        
    def augment_episode(self, original_video_path: Path, output_dir: Path) -> List[Dict[str, Any]]:
        """
        Augments a single episode by generating 5 synthetic variants.
        
        Args:
            original_video_path: The original episode path.
            output_dir: Directory to save generated videos.
            
        Returns:
            List of dictionaries representing the new augmented episodes.
        """
        
        if not original_video_path or not os.path.exists(original_video_path):
            logger.warning(f"Video not found for episode {original_video_path}. Skipping augmentation.")
            return []

        # 2. Caption inputted by user
        initial_caption = self.initial_caption
        
        # 4. Generate Variations (LLM)
        variations = self.generate_prompt_variations(initial_caption, num_variations=5, openai_api_key=self.openai_api_key)
        
        # 5. Segment Robot (Grounding DINO + SAMv2) - generates binary mask video
        robot_mask_video_path = self._segment_robot(original_video_path, output_dir)
        
        augmented_episodes = []
        
        for i, variation_prompt in enumerate(variations):
            # 6. Generate Video (Cosmos)
            new_video_filename = f"aug_{original_video_path.stem}_{i}.mp4"
            new_video_path = output_dir / new_video_filename
            with open("prompt_path.txt", "w") as prompt_path:
                prompt_path.write(variation_prompt)
            
            self._generate_video(
                prompt=prompt_path,
                original_video_path=original_video_path,
                mask_video_path=robot_mask_video_path,
                output_path=new_video_path
            )
            
            
        return new_video_path

    def _get_video_path(self, episode_data: pd.Series) -> Optional[str]:
        """
        Retrieves the video path for the episode.
        """
        # Check for direct video_path field
        if 'video_path' in episode_data and pd.notna(episode_data['video_path']):
            video_path = episode_data['video_path']
            if os.path.exists(video_path):
                return video_path
        
        # Try to construct from dataset_filename and filename
        if 'dataset_filename' in episode_data and 'filename' in episode_data:
            from ares.utils.image_utils import get_video_mp4
            try:
                dataset_path = episode_data.get('path') if 'path' in episode_data else None
                video_path = get_video_mp4(
                    episode_data['dataset_filename'],
                    episode_data['filename'],
                    dataset_path=dataset_path
                )
                return video_path
            except Exception as e:
                logger.warning(f"Could not construct video path: {e}")
        
        return None
    
    def _extract_video_frames(self, video_path: str, n_frames: int = 8) -> List[np.ndarray]:
        """Extract frames from video for processing."""
        try:
            frames = split_video_to_frames(video_path, filesize_limit_mb=100)
            # Select representative frames
            selected_frames = choose_and_preprocess_frames(frames, n_frames=n_frames)
            return selected_frames
        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {e}")
            # Fallback: extract using cv2 directly
            cap = cv2.VideoCapture(video_path)
            frames = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            cap.release()
            return frames


    def generate_prompt_variations(self, input_prompt: str, num_variations: int, openai_api_key: str):
        """
        Calls the OpenAI API to generate variations of a prompt.
        
        Args:
            input_prompt (str): The base prompt containing variables in [brackets].
            num_variations (int): How many variations to produce.
            openai_api_key (str): Your OpenAI API key.

        Returns:
            list[str]: A list of generated prompt variations.
        """
        
        client = OpenAI(api_key=openai_api_key)

        # Prepare the instruction for GPT
        system_message = (
            "You create variations of prompts by modifying ONLY the variables inside square brackets. "
            "Do not change any text outside bracketed variables."
        )

        user_message = (
            f"Create {num_variations} variations of this prompt. Only modify bracketed variables:\n\n"
            f"{input_prompt}"
        )

        response = client.chat.completions.create(
            model="gpt-5.1",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7
        )

        # Extract the assistant's output
        raw_output = response.choices[0].message.content
        
        # Basic parsing: split on blank lines or numbered headings
        # Users can customize this parsing depending on preferred formatting
        variations = [
            v.strip()
            for v in raw_output.split("\n\n")
            if v.strip()
        ]

        return variations


    def _segment_robot(self, video_path: str, output_dir: Path) -> Optional[str]:
        """
        Grounding DINO + SAMv2: Isolates the robot from the video.
        
        Returns a binary spatiotemporal mask video in MP4 format where:
        - White pixels (255) = robot regions (control will be applied)
        - Black pixels (0) = background (control will NOT be applied)
        
        Args:
            video_path: Path to the input video
            output_dir: Directory to save the mask video
            
        Returns:
            Path to the generated mask video (MP4), or None if segmentation fails
        """
        if self.grounding_annotator is None:
            logger.warning("GroundingAnnotator not available, skipping robot segmentation")
            return None
        
        logger.info("Segmentation: Generating binary robot mask video...")
        try:
            # Read the entire video
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Prepare output path for mask video
            mask_video_path = output_dir / f"robot_mask_{Path(video_path).stem}.mp4"
            
            # Initialize video writer for mask video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            mask_writer = cv2.VideoWriter(
                str(mask_video_path),
                fourcc,
                fps,
                (width, height),
                isColor=False  # Grayscale for binary mask
            )
            
            # Process video in batches for efficiency
            batch_size = 8
            frame_buffer = []
            mask_buffer = []
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_buffer.append(frame)
                
                # Process batch when full or at end of video
                if len(frame_buffer) >= batch_size or frame_idx == total_frames - 1:
                    # Convert frames to PIL Images
                    pil_images = [
                        Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                        for f in frame_buffer
                    ]
                    
                    # Run detection and segmentation on batch
                    labels_str = "robot"
                    batch_annotations = self.grounding_annotator.process_batch(pil_images, labels_str)
                    
                    # Generate binary masks for each frame
                    for frame, annotations in zip(frame_buffer, batch_annotations):
                        # Create blank mask (all black = no control)
                        binary_mask = np.zeros((height, width), dtype=np.uint8)
                        
                        # If robot detected, set those pixels to white (255 = apply control)
                        if annotations and len(annotations) > 0:
                            for annotation in annotations:
                                if "segmentation" in annotation:
                                    seg_mask = annotation["segmentation"]
                                    # Ensure mask is binary (0 or 255)
                                    if isinstance(seg_mask, np.ndarray):
                                        # Convert boolean or float mask to binary
                                        binary_mask = np.where(seg_mask > 0, 255, binary_mask).astype(np.uint8)
                        
                        # Write binary mask frame to video
                        mask_writer.write(binary_mask)
                    
                    # Clear buffers
                    frame_buffer = []
                
                frame_idx += 1
            
            # Release resources
            cap.release()
            mask_writer.release()
            
            logger.info(f"Successfully generated robot mask video: {mask_video_path}")
            return str(mask_video_path)
            
        except Exception as e:
            logger.error(f"Error segmenting robot: {e}")
            return None

    def _generate_video(
        self, prompt_path: str, original_video_path: str, mask_video_path: Optional[str], output_path: Path
    ) -> None:
        """
        Cosmos-Transfer2.5-2B: Generates the augmented video.
        
        Supports both local and RunPod execution modes.
        """
        # Read prompt from file
        with open(prompt_path, 'r') as f:
            prompt = f.read().strip()
        
        # Prepare parameters
        parameters = {
            "guidance": 3,
            "edge": {"control_weight": 0.4},
            "vis": {"control_weight": 0.2},
        }
        
        # Add mask to vis control if available
        if mask_video_path and os.path.exists(mask_video_path):
            parameters["vis"]["mask_path"] = mask_video_path
            logger.info(f"Using robot mask video: {mask_video_path}")
        
        # Execute based on mode
        if self.execution_mode == "runpod" and self.runpod_executor:
            logger.info("Using RunPod GPU for Cosmos inference...")
            try:
                # Run inference on RunPod
                temp_output = self.runpod_executor.run_inference(
                    video_path=original_video_path,
                    prompt=prompt,
                    mask_path=mask_video_path,
                    parameters=parameters,
                    timeout=600,
                )
                
                # Move output to final location
                import shutil
                shutil.move(temp_output, str(output_path))
                logger.info(f"Successfully generated video via RunPod: {output_path}")
                return
                
            except Exception as e:
                logger.error(f"RunPod inference failed: {e}")
                logger.warning("Falling back to local execution...")
                # Fall through to local execution
        
        # Local execution (subprocess)
        logger.info("Using local GPU for Cosmos inference...")
        import subprocess
        
        # Build full parameters for local execution
        full_parameters = {
            "prompt_path": prompt_path,
            "output_dir": str(output_path.parent),
            "video_path": original_video_path,
            **parameters
        }
        
        # Write parameters to JSON file
        params_json_path = output_path.parent / f"params_{output_path.stem}.json"
        with open(params_json_path, "w") as f:
            json.dump(full_parameters, f, indent=4)
        
        logger.info(f"Running Cosmos inference with parameters: {params_json_path}")
        
        # Run Cosmos inference
        try:
            result = subprocess.run(
                ['python', 'examples/inference.py', '-i', str(params_json_path), '-o', str(output_path)],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully generated video: {output_path}")
            else:
                logger.error(f"Cosmos inference failed: {result.stderr}")
                raise RuntimeError(f"Cosmos inference failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            logger.error("Cosmos inference timed out after 10 minutes")
            raise
        except Exception as e:
            logger.error(f"Error running Cosmos inference: {e}")
            raise
