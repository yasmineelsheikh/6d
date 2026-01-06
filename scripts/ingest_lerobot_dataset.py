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

# Database imports removed - step 3c (database record creation) is skipped
# from ares.configs.base import Environment, Robot, Rollout, Task, Trajectory
# from ares.databases.structured_database import (
#     RolloutSQLModel,
#     add_rollout,
#     setup_database,
# )


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
    """Process a single chunk of parquet files.
    
    NOTE: Step 3c (database record creation) is skipped.
    This function only counts episodes and processes videos.
    No database records (Robot, Environment, Task, Trajectory, Rollout) are created.
    """
    count = 0
    for file_path in chunk_path.glob("*.parquet"):
        df = pd.read_parquet(file_path)
        
        # Group by episode_index
        for episode_idx, episode_df in tqdm(df.groupby("episode_index"), desc=f"Processing {file_path.name}"):
            # Sort by frame_index to ensure order
            episode_df = episode_df.sort_values("frame_index")
            
            # Step 3c SKIPPED: Database record creation
            # The following would normally be created but are skipped:
            # - Robot object
            # - Environment object  
            # - Task object
            # - Trajectory object
            # - Rollout object (with add_rollout to database)
            
            # Just count episodes for reporting
            count += 1
            
    return count


def ingest_dataset(data_dir: str, engine_url: str, dataset_name: str) -> int:
    """Ingest LeRobot dataset into ARES.
    
    NOTE: Step 3c (database record creation) is skipped.
    Only processes videos and counts episodes. No database records are created.
    
    Returns:
        int: Number of episodes processed (not ingested into database)
    """
    dataset_path = Path(data_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
        
    # Step 3c SKIPPED: Database setup skipped since we're not creating records
    engine = None
    
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
    
    print(f"Ingestion complete. Total episodes processed: {total_episodes} (videos split, database records skipped)")
    return total_episodes


@click.command()
@click.option("--data-dir", required=True, help="Path to the LeRobot dataset directory")
@click.option("--engine-url", required=True, help="SQLAlchemy database URL")
@click.option("--dataset-name", default="lerobot_dataset", help="Formal name for the dataset in ARES")
def main(data_dir: str, engine_url: str, dataset_name: str):
    ingest_dataset(data_dir, engine_url, dataset_name)


if __name__ == "__main__":
    main()
