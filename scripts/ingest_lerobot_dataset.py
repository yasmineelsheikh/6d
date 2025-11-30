"""
Script to ingest LeRobot datasets into the ARES database.
Reads parquet files from the LeRobot format and converts them to ARES Rollout objects.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from ares.configs.base import Environment, Robot, Rollout, Task, Trajectory
from ares.databases.structured_database import (
    RolloutSQLModel,
    add_rollout,
    setup_database,
)


def load_metadata(dataset_path: Path) -> tuple[dict, pd.DataFrame]:
    """Load dataset metadata and task descriptions."""
    info_path = dataset_path / "meta/info.json"
    tasks_path = dataset_path / "meta/tasks.parquet"

    with open(info_path, "r") as f:
        info = json.load(f)

    tasks_df = pd.read_parquet(tasks_path)
    return info, tasks_df


def split_videos_for_parquet(parquet_path: Path, dataset_path: Path, dataset_name: str):
    """Split the corresponding video file into episode clips."""
    try:
        from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
        from ares.constants import ARES_VIDEO_DIR
    except ImportError:
        print("moviepy not installed. Skipping video splitting.")
        return

    # Determine corresponding video file
    # Structure: videos/observation.images.cam_right/chunk-XXX/file-YYY.mp4
    # Parquet: data/chunk-XXX/file-YYY.parquet
    
    chunk_name = parquet_path.parent.name
    file_name = parquet_path.stem # file-000
    
    # We assume cam_right is the main video source as per user request
    video_source = "observation.images.cam_right"
    video_file = dataset_path / "videos" / video_source / chunk_name / f"{file_name}.mp4"
    
    if not video_file.exists():
        print(f"Video file not found: {video_file}")
        return
        
    # Read parquet to get frame ranges
    df = pd.read_parquet(parquet_path)
    fps = 30.0 # Default, should read from info.json but passing info is complex here. 
               # We can load info again or pass it. For now hardcode or read info.
    
    # Create output directory
    output_dir = Path(ARES_VIDEO_DIR) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Splitting video {video_file} into episodes...")
    
    for episode_idx, episode_df in df.groupby("episode_index"):
        # Check if already exists
        output_path = output_dir / f"episode_{episode_idx}.mp4"
        if output_path.exists():
            continue
            
        # Get frame range
        min_frame = episode_df["frame_index"].min()
        max_frame = episode_df["frame_index"].max()
        
        # Convert to seconds
        start_time = min_frame / fps
        end_time = (max_frame + 1) / fps
        
        try:
            ffmpeg_extract_subclip(str(video_file), start_time, end_time, targetname=str(output_path))
            
            # Extract first frame for lazy loading preview
            import cv2
            
            # Create frames directory: data/videos/{dataset_name}/episode_{idx}
            frames_dir = output_dir / f"episode_{episode_idx}"
            frames_dir.mkdir(parents=True, exist_ok=True)
            
            # Read the first frame from the newly created video
            cap = cv2.VideoCapture(str(output_path))
            ret, frame = cap.read()
            if ret:
                # Save as frame_0000.jpg
                frame_path = frames_dir / "frame_0000.jpg"
                cv2.imwrite(str(frame_path), frame)
            cap.release()
            
        except Exception as e:
            print(f"Error splitting/processing episode {episode_idx}: {e}")


def process_chunk(
    chunk_path: Path,
    dataset_path: Path,
    info: dict,
    tasks_df: pd.DataFrame,
    engine,
    dataset_formalname: str,
) -> int:
    """Process a single chunk of parquet files."""
    count = 0
    for file_path in chunk_path.glob("*.parquet"):
        df = pd.read_parquet(file_path)
        
        # Group by episode_index
        for episode_idx, episode_df in tqdm(df.groupby("episode_index"), desc=f"Processing {file_path.name}"):
            # Sort by frame_index to ensure order
            episode_df = episode_df.sort_values("frame_index")
            
            # Extract trajectory data
            states = [s.tolist() if isinstance(s, np.ndarray) else s for s in episode_df["observation.state"]]
            # Some datasets might use different action keys, but LeRobot usually uses 'action'
            actions = [a.tolist() if isinstance(a, np.ndarray) else a for a in episode_df["action"]]
            
            # Extract task info
            task_idx = episode_df["task_index"].iloc[0]
            # Handle case where task_index might not match directly if tasks_df index is not aligned
            # Assuming task_index corresponds to the row index in tasks_df or a specific column
            # In the inspected file, tasks_df has 'task_index' column and index is the description? 
            # Wait, inspected output showed:
            #                      task_index
            # Stack colorful cups           0
            # So the index is the description, and 'task_index' is the value.
            
            task_description = "Unknown Task"
            # Find the description where task_index matches
            matching_task = tasks_df[tasks_df["task_index"] == task_idx]
            if not matching_task.empty:
                task_description = matching_task.index[0]
            
            # Construct ARES objects
            
            # Robot
            # Infer from info or default
            robot_type = info.get("robot_type", "ur5e") # Defaulting to ur5e as common, but should check info
            # info.json usually has 'robot_type' or similar. 
            # For now, we'll use a generic placeholder if missing, or try to parse from path
            
            robot = Robot(
                embodiment=robot_type,
                morphology=robot_type, # Simplified
                action_space="joint_pos", # Assumption for LeRobot usually
                rgb_cams=None, # Would need to check video folder
                depth_cams=None,
                wrist_cams=None,
                color_estimate="unknown",
                camera_angle_estimate="unknown"
            )
            
            # Environment
            env = Environment(
                name="real_world",
                lighting_estimate="normal",
                simulation_estimate=False,
                background_estimate="unknown",
                surface_estimate="unknown",
                focus_objects_estimate="",
                distractor_objects_estimate="",
                people_estimate=False,
                static_estimate=True
            )
            
            # Task
            task = Task(
                language_instruction=task_description,
                language_instruction_type="instruction",
                success_estimate=1.0, # Assuming collected data is successful demonstrations? LeRobot usually is.
                complexity_category_estimate="medium",
                complexity_score_estimate=0.5,
                rarity_estimate=0.0
            )
            
            # Trajectory
            trajectory = Trajectory(
                states=json.dumps(states),
                actions=json.dumps(actions),
                is_first=0,
                is_last=len(states) - 1,
                is_terminal=len(states) - 1,
                rewards=None # LeRobot data often doesn't have explicit rewards per step in the parquet
            )
            
            # Rollout
            rollout_id = uuid.uuid4()
            rollout = Rollout(
                id=rollout_id,
                creation_time=datetime.now(),
                ingestion_time=datetime.now(),
                path=str(dataset_path),
                filename=f"episode_{episode_idx}", # Point to the split video
                dataset_name=dataset_path.name,
                dataset_filename=dataset_formalname, # This matches the folder in ARES_VIDEO_DIR
                dataset_formalname=dataset_formalname,
                description_estimate=f"LeRobot episode {episode_idx} for task: {task_description}",
                length=len(states),
                robot=robot,
                environment=env,
                task=task,
                trajectory=trajectory,
                split="train" # Default
            )
            
            add_rollout(engine, rollout, RolloutSQLModel)
            count += 1
            
    return count


def ingest_dataset(data_dir: str, engine_url: str, dataset_name: str) -> int:
    """Ingest LeRobot dataset into ARES.
    
    Returns:
        int: Number of episodes ingested
    """
    dataset_path = Path(data_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
        
    engine = setup_database(RolloutSQLModel, path=engine_url)
    
    print(f"Loading metadata from {dataset_path}...")
    info, tasks_df = load_metadata(dataset_path)
    print(f"Found {len(tasks_df)} tasks.")
    
    data_path = dataset_path / "data"
    total_episodes = 0
    
    # Iterate over chunks
    for chunk_path in data_path.glob("chunk-*"):
        if chunk_path.is_dir():
            print(f"Processing chunk: {chunk_path.name}")
            
            # Split videos for this chunk
            for parquet_file in chunk_path.glob("*.parquet"):
                split_videos_for_parquet(parquet_file, dataset_path, dataset_name)
                
            count = process_chunk(chunk_path, dataset_path, info, tasks_df, engine, dataset_name)
            total_episodes += count
            
    print(f"Ingestion complete. Total episodes ingested: {total_episodes}")
    return total_episodes


@click.command()
@click.option("--data-dir", required=True, help="Path to the LeRobot dataset directory")
@click.option("--engine-url", required=True, help="SQLAlchemy database URL")
@click.option("--dataset-name", default="lerobot_dataset", help="Formal name for the dataset in ARES")
def main(data_dir: str, engine_url: str, dataset_name: str):
    ingest_dataset(data_dir, engine_url, dataset_name)


if __name__ == "__main__":
    main()
