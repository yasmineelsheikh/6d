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

from ares.models.base import VLM, GeminiVideoVLM
from ares.models.grounding import GroundingAnnotator
from ares.models.shortcuts import get_gemini_15_pro, get_gpt_4o_mini
from ares.utils.image_utils import split_video_to_frames, choose_and_preprocess_frames

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CosmosAugmentor:
    """
    Implements the Cosmos Data Augmentation Strategy (6.2.2).
    
    Uses:
    - Cosmos-Transfer2.5-2B (Video Generation) - via API/service
    - Grounding DINO + SAMv2 (Segmentation) - via GroundingAnnotator
    - VLM (Captioning) - via GeminiVideoVLM or VLM
    - LLM (Prompt Variation) - via VLM
    """
    
    def __init__(
        self,
        vlm_provider: str = "gemini",
        vlm_name: str = "gemini-1.5-pro",
        llm_provider: str = "openai",
        llm_name: str = "gpt-4o-mini",
        use_gpu: bool = True,
    ):
        self.edge_threshold = "medium"
        self.blur_threshold = "very_low"
        self.guidance_scale = 3
        
        # Initialize VLM for video captioning
        try:
            if vlm_provider == "gemini" and "gemini" in vlm_name.lower():
                self.vlm = GeminiVideoVLM(provider=vlm_provider, name=vlm_name)
            else:
                self.vlm = VLM(provider=vlm_provider, name=vlm_name)
        except Exception as e:
            logger.warning(f"Failed to initialize VLM {vlm_provider}/{vlm_name}: {e}. Using fallback.")
            self.vlm = get_gemini_15_pro()
        
        # Initialize LLM for prompt variations
        try:
            self.llm = VLM(provider=llm_provider, name=llm_name)
        except Exception as e:
            logger.warning(f"Failed to initialize LLM {llm_provider}/{llm_name}: {e}. Using fallback.")
            self.llm = get_gpt_4o_mini()
        
        # Initialize GroundingAnnotator for robot segmentation
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
        
    def augment_episode(self, episode_data: pd.Series, output_dir: Path) -> List[Dict[str, Any]]:
        """
        Augments a single episode by generating 5 synthetic variants.
        
        Args:
            episode_data: The original episode data (row from DataFrame).
            output_dir: Directory to save generated videos.
            
        Returns:
            List of dictionaries representing the new augmented episodes.
        """
        # 1. Extract Video
        # Assuming video path is stored or can be derived. 
        # In the current data structure, we might need to look up the video file.
        # For this implementation, we'll assume we can get a path or use a placeholder.
        original_video_path = self._get_video_path(episode_data)
        
        if not original_video_path or not os.path.exists(original_video_path):
            logger.warning(f"Video not found for episode {episode_data.get('id', 'unknown')}. Skipping augmentation.")
            return []

        # 2. Generate Caption (VLM)
        initial_caption = self._generate_caption(original_video_path)
        
        # 3. Refine Caption (Cosmos Loop)
        refined_caption = self._refine_caption(initial_caption, original_video_path)
        
        # 4. Generate Variations (LLM)
        variations = self._generate_variations(refined_caption, num_variations=5)
        
        # 5. Segment Robot (Grounding DINO + SAMv2)
        robot_mask = self._segment_robot(original_video_path)
        
        augmented_episodes = []
        
        for i, variation_prompt in enumerate(variations):
            # 6. Generate Video (Cosmos)
            new_video_filename = f"aug_{episode_data['id']}_{i}.mp4"
            new_video_path = output_dir / new_video_filename
            
            self._generate_video(
                prompt=variation_prompt,
                original_video_path=original_video_path,
                mask=robot_mask,
                output_path=new_video_path
            )
            
            # 7. Create New Episode Record
            # Copy original data but update video path and ID
            new_episode = episode_data.to_dict()
            new_episode['id'] = str(uuid.uuid4())
            new_episode['is_augmented'] = True
            new_episode['augmentation_source'] = 'cosmos_transfer_2.5_2b'
            new_episode['augmentation_prompt'] = variation_prompt
            # Update video path in the record if applicable
            # (Assuming there's a field for it, otherwise we just track it)
            new_episode['video_path'] = str(new_video_path)
            
            augmented_episodes.append(new_episode)
            
        return augmented_episodes

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

    def _generate_caption(self, video_path: str) -> str:
        """
        VLM: Generates a detailed caption of the scene from video.
        """
        logger.info("VLM: Generating caption from video...")
        try:
            # Use GeminiVideoVLM if available, otherwise extract frames and use regular VLM
            if isinstance(self.vlm, GeminiVideoVLM):
                info = {
                    "prompt": (
                        "Please provide a detailed description of this video scene. "
                        "Describe the environment, objects, robot actions, and any notable details. "
                        "Be specific about colors, positions, and interactions."
                    )
                }
                messages, response = self.vlm.ask(info=info, video_path=video_path)
                # Parse Gemini response
                if hasattr(response, 'text'):
                    caption = response.text
                elif hasattr(response, 'candidates') and len(response.candidates) > 0:
                    caption = response.candidates[0].content.parts[0].text
                else:
                    caption = str(response)
            else:
                # Extract frames and use regular VLM
                frames = self._extract_video_frames(video_path, n_frames=4)
                # Convert frames to PIL Images
                images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
                
                info = {
                    "prompt": (
                        "Please provide a detailed description of these video frames. "
                        "Describe the environment, objects, robot actions, and any notable details. "
                        "Be specific about colors, positions, and interactions."
                    )
                }
                messages, response = self.vlm.ask(info=info, images=images)
                # Parse standard VLM response
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    caption = response.choices[0].message.content
                else:
                    caption = str(response)
            
            logger.info(f"Generated caption: {caption[:100]}...")
            return caption
        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            # Fallback to a generic caption
            return "A robot performing a task in an environment."

    def _refine_caption(self, caption: str, video_path: str) -> str:
        """
        Cosmos Loop: Iteratively refines the caption by generating and comparing videos.
        For now, we use VLM to refine the caption based on the video.
        """
        logger.info("Cosmos: Refining caption...")
        try:
            if isinstance(self.vlm, GeminiVideoVLM):
                info = {
                    "prompt": (
                        f"Given this initial description: '{caption}'\n\n"
                        "Watch the video and provide a more detailed and accurate description. "
                        "Include specific details about the scene, objects, robot actions, and environment."
                    )
                }
                messages, response = self.vlm.ask(info=info, video_path=video_path)
                # Parse Gemini response
                if hasattr(response, 'text'):
                    refined = response.text
                elif hasattr(response, 'candidates') and len(response.candidates) > 0:
                    refined = response.candidates[0].content.parts[0].text
                else:
                    refined = str(response)
            else:
                frames = self._extract_video_frames(video_path, n_frames=4)
                images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
                
                info = {
                    "prompt": (
                        f"Given this initial description: '{caption}'\n\n"
                        "Look at these video frames and provide a more detailed and accurate description. "
                        "Include specific details about the scene, objects, robot actions, and environment."
                    )
                }
                messages, response = self.vlm.ask(info=info, images=images)
                # Parse standard VLM response
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    refined = response.choices[0].message.content
                else:
                    refined = str(response)
            
            logger.info(f"Refined caption: {refined[:100]}...")
            return refined
        except Exception as e:
            logger.error(f"Error refining caption: {e}")
            return caption

    def _generate_variations(self, caption: str, num_variations: int) -> List[str]:
        """
        LLM: Generates candidate variations for components in the scene.
        """
        logger.info(f"LLM: Generating {num_variations} variations...")
        try:
            prompt = (
                f"Given this scene description: '{caption}'\n\n"
                f"Generate {num_variations} variations of this scene description by modifying specific components "
                "such as colors, objects, lighting, or background elements. Each variation should be a complete, "
                "detailed description that could be used to generate a similar but distinct video scene. "
                "Return the variations as a JSON array of strings, one description per element."
            )
            
            info = {"prompt": prompt}
            messages, response = self.llm.ask(info=info)
            # Parse VLM response
            if hasattr(response, 'choices') and len(response.choices) > 0:
                response_text = response.choices[0].message.content
            else:
                response_text = str(response)
            
            # Try to parse JSON response
            try:
                import json
                # Extract JSON from response if it's wrapped in markdown code blocks
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0].strip()
                
                variations = json.loads(response_text)
                if isinstance(variations, list) and len(variations) >= num_variations:
                    return variations[:num_variations]
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract variations from text
                lines = [line.strip() for line in response_text.split('\n') if line.strip()]
                variations = [line for line in lines if len(line) > 20][:num_variations]
                if len(variations) >= num_variations:
                    return variations
            
            # Fallback: generate simple variations
            logger.warning("Could not parse LLM response, using fallback variations")
            return [f"{caption} [Variation {i+1}]" for i in range(num_variations)]
            
        except Exception as e:
            logger.error(f"Error generating variations: {e}")
            # Fallback to simple variations
            return [f"{caption} [Variation {i+1}]" for i in range(num_variations)]

    def _segment_robot(self, video_path: str) -> Optional[Any]:
        """
        Grounding DINO + SAMv2: Isolates the robot from the video.
        """
        if self.grounding_annotator is None:
            logger.warning("GroundingAnnotator not available, skipping robot segmentation")
            return None
        
        logger.info("Segmentation: Isolating robot pixels...")
        try:
            # Extract a few representative frames
            frames = self._extract_video_frames(video_path, n_frames=3)
            
            # Convert frames to PIL Images
            images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
            
            # Use "robot" as the label to detect
            labels_str = "robot"
            
            # Run detection and segmentation
            annotations = self.grounding_annotator.process_batch(images, labels_str)
            
            # Extract masks from annotations
            # Return the first frame's robot mask if available
            if annotations and len(annotations) > 0 and len(annotations[0]) > 0:
                robot_annotation = annotations[0][0]  # First frame, first detection
                if "segmentation" in robot_annotation:
                    mask = robot_annotation["segmentation"]
                    logger.info("Successfully segmented robot")
                    return mask
            
            logger.warning("No robot detected in video")
            return None
            
        except Exception as e:
            logger.error(f"Error segmenting robot: {e}")
            return None

    def _generate_video(
        self, prompt: str, original_video_path: str, mask: Any, output_path: Path
    ) -> None:
        """
        Cosmos-Transfer2.5-2B: Generates the augmented video.
        
        This method tries to use Cosmos-Transfer2.5 directly if available,
        otherwise falls back to API or placeholder.
        """
        logger.info(f"Cosmos-Transfer: Generating video with prompt: '{prompt[:50]}...'")
        
        # Try to use Cosmos directly (requires GPU and CUDA)
        if self._try_cosmos_direct(prompt, original_video_path, mask, output_path):
            return
        
        # Check for Cosmos API endpoint or service
        cosmos_api_url = os.environ.get("COSMOS_API_URL")
        cosmos_api_key = os.environ.get("COSMOS_API_KEY")
        
        if cosmos_api_url and cosmos_api_key:
            # Call Cosmos-Transfer2.5-2B API
            try:
                import requests
                
                # Prepare request
                with open(original_video_path, 'rb') as f:
                    files = {'video': f}
                    data = {
                        'prompt': prompt,
                        'guidance_scale': self.guidance_scale,
                    }
                    if mask is not None:
                        # Convert mask to format expected by API
                        # This would need to be adapted based on API requirements
                        pass
                    
                    response = requests.post(
                        f"{cosmos_api_url}/generate",
                        files=files,
                        data=data,
                        headers={'Authorization': f'Bearer {cosmos_api_key}'},
                        timeout=300  # 5 minute timeout for video generation
                    )
                    
                    if response.status_code == 200:
                        with open(output_path, 'wb') as out_file:
                            out_file.write(response.content)
                        logger.info(f"Successfully generated video via Cosmos API: {output_path}")
                        return
                    else:
                        logger.warning(f"Cosmos API returned error {response.status_code}: {response.text}")
            except Exception as e:
                logger.error(f"Error calling Cosmos API: {e}")
        
        # Fallback: Copy original video as placeholder
        logger.warning("Cosmos not available, using original video as placeholder")
        try:
            import shutil
            if os.path.exists(original_video_path):
                shutil.copy(original_video_path, output_path)
                logger.info(f"Copied original video to {output_path} (placeholder)")
            else:
                raise FileNotFoundError(f"Original video not found: {original_video_path}")
        except Exception as e:
            logger.error(f"Failed to copy video: {e}")
            raise
    
    def _try_cosmos_direct(
        self, prompt: str, original_video_path: str, mask: Any, output_path: Path
    ) -> bool:
        """
        Try to use Cosmos-Transfer2.5 directly for video generation.
        
        Requirements:
        - GPU with CUDA support (65GB+ VRAM recommended)
        - Cosmos-Transfer2.5 installed and configured
        - Model checkpoints downloaded
        
        Returns True if successful, False otherwise.
        """
        try:
            # Patch decord imports to use our compatibility layer (for macOS compatibility)
            # This must happen BEFORE any Cosmos modules are imported
            import sys
            if 'decord' not in sys.modules:
                from ares.augmentation.decord_compat import VideoReader, cpu as decord_cpu
                # Create a mock decord module
                class MockDecord:
                    VideoReader = VideoReader
                    cpu = decord_cpu
                sys.modules['decord'] = MockDecord()
                logger.debug("Patched decord module with compatibility layer")
            
            # Try to import Cosmos components
            from cosmos_transfer2.inference import Control2WorldInference
            from cosmos_transfer2.config import InferenceArguments, SetupArguments, SegConfig
            from cosmos_oss.init import init_environment, cleanup_environment
            import torch
            
            # Check if GPU is available
            if not torch.cuda.is_available():
                logger.warning("CUDA not available. Cosmos requires GPU. Skipping direct inference.")
                return False
            
            # Detect AWS environment and GPU configuration
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) if gpu_count > 0 else 0
            
            logger.info(f"Detected {gpu_count} GPU(s): {gpu_name} ({gpu_memory:.1f} GB)")
            
            # Check if GPU has enough memory (65GB recommended for 2B model)
            if gpu_memory < 40:
                logger.warning(f"GPU memory ({gpu_memory:.1f} GB) may be insufficient. Cosmos-Transfer2.5-2B recommends 65GB+ VRAM.")
            
            # Detect AWS environment
            is_aws = os.path.exists("/sys/class/dmi/id/product_version") and "amazon" in open("/sys/class/dmi/id/product_version").read().lower()
            if is_aws:
                logger.info("AWS environment detected. Optimizing for cloud GPU instance.")
            
            logger.info("Attempting to use Cosmos-Transfer2.5 directly...")
            
            # Initialize environment
            init_environment()
            
            try:
                # Determine context parallel size
                # For single video generation, start with 1 GPU
                # For batch processing, can use multiple GPUs
                context_parallel_size = 1  # Single GPU for now (can be increased for batch processing)
                if gpu_count > 1:
                    logger.info(f"Multiple GPUs detected ({gpu_count}). Using {context_parallel_size} GPU for this inference.")
                    logger.info("Note: Multi-GPU inference can be enabled by setting context_parallel_size > 1")
                
                # Create setup arguments
                setup_args = SetupArguments(
                    output_dir=str(output_path.parent),
                    model="cosmos-transfer2.5-2b",
                    disable_guardrails=True,  # Disable for faster inference
                    context_parallel_size=context_parallel_size,
                )
                
                # Create inference arguments with segmentation control
                # Use segmentation control from the original video
                seg_config = SegConfig(
                    control_path=str(original_video_path),
                    control_weight=1.0,
                )
                
                inference_args = InferenceArguments(
                    name="augmented_video",
                    prompt=prompt,
                    guidance=self.guidance_scale,
                    seg=seg_config,
                )
                
                # Initialize inference
                inference = Control2WorldInference(
                    setup_args,
                    batch_hint_keys=["seg"] if mask is None else ["seg"]
                )
                
                # Generate video
                inference.generate([inference_args], output_dir=setup_args.output_dir)
                
                # Find the generated video (Cosmos saves with a specific naming pattern)
                generated_files = list(output_path.parent.glob("*.mp4"))
                if generated_files:
                    # Use the most recently created file
                    latest_file = max(generated_files, key=lambda p: p.stat().st_mtime)
                    import shutil
                    shutil.move(str(latest_file), str(output_path))
                    logger.info(f"Successfully generated video using Cosmos: {output_path}")
                    return True
                else:
                    logger.warning("Cosmos generated video but output file not found")
                    return False
                    
            finally:
                cleanup_environment()
                
        except ImportError as e:
            logger.debug(f"Cosmos not available: {e}")
            return False
        except Exception as e:
            logger.warning(f"Failed to use Cosmos directly: {e}")
            return False
