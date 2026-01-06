# put each episode in dataset into openxembodiment format

# create list of each episode in openxembodiment format (iterator class -> ds)

#set up engine (sqlite database)

#run run_structured_database_ingestion on ds




import asyncio
import os
import typing as t

from tqdm import tqdm

from ares.constants import (
    ARES_DATA_DIR,
    ARES_OXE_DIR,
    DATASET_NAMES,
    get_dataset_info_by_key,
)
#from ares.databases.annotation_database import ANNOTATION_DB_PATH
from ares.databases.embedding_database import EMBEDDING_DB_PATH
from ares.databases.structured_database import (
    ROBOT_DB_PATH,
    RolloutSQLModel,
    setup_database,
)
from ares.extras.pi_demo_utils import PI_DEMO_TASKS
from ares.models.shortcuts import get_nomic_embedder
from ares.utils.image_utils import get_video_frames, split_video_to_frames, save_video
from main import run_ingestion_pipeline

#dataset_filename = "pi_demos"
split = "test"
#dataset_info = get_dataset_info_by_key("dataset_filename", dataset_filename)
#dataset_formalname = dataset_info["dataset_formalname"]

full_dataset_info = {
    "Dataset": 'Stack Cups',
    "Dataset Filename": 'stack_cups',
    "Dataset Formalname": 'Stack Cups',
    "Split": split,
    "Robot": None,
    "Robot Morphology": None,
    "Gripper": None,
    "Action Space": None,
    "# RGB Cams": None,
    "# Depth Cams": None,
    "# Wrist Cams": None,
    "Language Annotations": "Natural",
    "Data Collect Method": "Expert Policy",
    "Scene Type": None,
    "Citation": "year={2024}",
}


def prep_for_oxe_episode(task: str, episode_filename: str) -> dict | None:
    """
    Force the videos and task information into the OpenXEmbodimentEpisode format.
    """
    filename = episode_filename
    try:
        frames = split_video_to_frames(filename)
    except Exception as e:
        print(f"Error getting video frames for {filename}: {e}")
        return None
    metadata = {"file_path": filename}
    steps = []
    for i, frame in enumerate(frames):
        observation = {
            "image": frame,
        }
        steps.append(
            {
                "image": frame,
                "action": None,
                "state": None,
                "is_first": i == 0,
                "is_last": i == len(frames) - 1,
                "is_terminal": False,
                "language_embedding": None,
                "language_instruction": task,
                "observation": observation,
            }
        )
    return {"episode_metadata": metadata, "steps": steps}


class DemoIngestion:
    def __init__(self, task: str, no_of_episodes: int):
        self.task = task
        self._episodes = []
        for i in tqdm(range(no_of_episodes)):
            episode = prep_for_oxe_episode(task, f"/Users/mac/demo/ares-platform/data/videos/stack_cups/episode_{i}.mp4")
            if episode is not None:
                self._episodes.append(episode)
        self._index = 0

    def __iter__(self) -> "DemoIngestion":
        self._index = 0
        return self

    def __next__(self) -> dict:
        if self._index >= len(self._episodes):
            raise StopIteration
        episode = self._episodes[self._index]
        self._index += 1
        return episode

    def __len__(self) -> int:
        return len(self._episodes)


if __name__ == "__main__":
    vlm_name = "gpt-4o"
    engine = setup_database(RolloutSQLModel, path=ROBOT_DB_PATH)
    embedder = get_nomic_embedder()
    task = "stack_cups"
    no_of_episodes = 5
    #for i in tqdm(range(no_of_episodes)):
        #print(task)
    ds = DemoIngestion(task, no_of_episodes)
    run_ingestion_pipeline(
        ds,
        full_dataset_info,
        'Stack Cups',
        vlm_name,
        engine,
        'stack_cups',
        embedder,
        split,
    )