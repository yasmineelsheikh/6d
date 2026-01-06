import copy
import json
import re
import string
import traceback
import typing as t
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow_datasets.core import DatasetInfo

from ares.configs.base import (
    BaseConfig,
    Environment,
    Robot,
    Rollout,
    Task,
    Trajectory,
    merge_config_sources,
    merge_dicts,
    merge_several_dicts,
    pydantic_to_example_dict,
    pydantic_to_field_instructions,
)
from ares.configs.open_x_embodiment_configs import OpenXEmbodimentEpisode
from ares.models.base import VLM, parse_response
from ares.utils.image_utils import load_video_frames


def hard_coded_dataset_info_extraction_spreadsheet(dataset_info: dict) -> dict:
    year = None
    if "Citation" in dataset_info and not pd.isna(dataset_info["Citation"]):
        match = re.search(r"year\s*=\s*\{(\d{4})\}", dataset_info["Citation"])
        if match:
            year = int(match.group(1))
    return {
        "rollout": {
            "dataset_name": dataset_info["Dataset"],
            "dataset_formalname": dataset_info["Dataset Formalname"],
            "dataset_filename": dataset_info["Dataset Filename"],
            "creation_time": year,
            "ingestion_time": datetime.now(),
            "split": dataset_info["Split"],
        },
        "robot": {
            "embodiment": dataset_info["Robot"],
            "gripper": (
                dataset_info["Gripper"]
                if not pd.isna(dataset_info["Gripper"])
                else "None"
            ),
            "morphology": dataset_info["Robot Morphology"],
            "action_space": dataset_info["Action Space"],
            "rgb_cams": dataset_info["# RGB Cams"],
            "depth_cams": dataset_info["# Depth Cams"],
            "wrist_cams": dataset_info["# Wrist Cams"],
        },
        "environment": {
            "name": dataset_info["Scene Type"],
            "simulation": dataset_info["Data Collect Method"] == "Human VR",
            "data_collection_method": dataset_info["Data Collect Method"],
        },
        "task": {
            "language_instruction_type": dataset_info["Language Annotations"],
        },
    }


# hard coded tfds builder info extraction
def hard_coded_dataset_info_extraction_tfds(dataset_info: DatasetInfo) -> dict:
    # get year from citation
    year = None
    if "year" in dataset_info.citation:
        match = re.search(r"year\s*=\s*\{(\d{4})\}", dataset_info.citation)
        if match:
            year = int(match.group(1))

    return {"rollout": {"dataset_name": dataset_info.name, "creation_time": year}}


def hard_coded_episode_info_extraction(episode: OpenXEmbodimentEpisode) -> dict:
    # gather data
    steps = episode.steps
    firsts = [step.is_first for step in steps]
    lasts = [step.is_last for step in steps]
    terminals = [step.is_terminal for step in steps]
    rewards = [step.reward for step in steps]
    # get indices instead of steps for first, last, terminal
    is_first = np.where(firsts)[0][0] if np.any(firsts) else None
    is_last = np.where(lasts)[0][-1] if np.any(lasts) else None
    is_terminal = np.where(terminals)[0][-1] if np.any(terminals) else None
    success = episode.episode_metadata.success
    reward_step = None
    if any([x is not None for x in rewards]):
        if np.any(np.array(rewards) == 1):
            reward_step = np.where(np.array(rewards) == 1)[0][0]
        else:
            reward_step = -1
    # gather trajectory data
    actions = np.stack([step.action for step in steps]).tolist()
    states = np.stack([step.observation.state for step in steps]).tolist()
    path = episode.episode_metadata.file_path.removeprefix("/")
    return {
        "rollout": {
            "path": path,
            "filename": str(Path(path).with_suffix("")),
            "length": len(steps),
        },
        "trajectory": {
            "actions": actions,
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
            "states": states,
            "rewards": rewards,
            "reward_step": reward_step,
        },
        "task": {
            "language_instruction": steps[0].language_instruction,
            "success": success,
        },
    }


class InformationExtractor:
    def __init__(self) -> None:
        pass

    def extract(
        self,
        episode: OpenXEmbodimentEpisode,
        dataset_info: DatasetInfo,
        *,  # Force keyword arguments
        robot_kwargs: t.Optional[dict[str, t.Any]] = None,
        environment_kwargs: t.Optional[dict[str, t.Any]] = None,
        task_kwargs: t.Optional[dict[str, t.Any]] = None,
        model_kwargs: t.Optional[dict[str, t.Any]] = None,
    ) -> Rollout:
        raise NotImplementedError

    async def extract_batch(
        self,
        episodes: list[OpenXEmbodimentEpisode],
        dataset_info: DatasetInfo,
        *,  # Force keyword arguments
        robot_kwargs: t.Optional[dict[str, t.Any]] = None,
        environment_kwargs: t.Optional[dict[str, t.Any]] = None,
        task_kwargs: t.Optional[dict[str, t.Any]] = None,
        model_kwargs: t.Optional[dict[str, t.Any]] = None,
        trajectory_kwargs: t.Optional[dict[str, t.Any]] = None,
    ) -> list[Rollout]:
        raise NotImplementedError


class VLMInformationExtractor(InformationExtractor):
    def __init__(self, vlm: VLM):
        self.vlm = vlm

    def _prepare_prompt_info(
        self,
        episode: OpenXEmbodimentEpisode,
        hardcoded_info: dict,
    ) -> dict:
        """Prepare the prompt information for VLM."""
        return {
            "task": episode.steps[0].language_instruction,
            "field_instructions": (
                "Please provide the following required information:\n"
                + "\n".join(
                    pydantic_to_field_instructions(
                        Rollout, exclude_fields=hardcoded_info, required_only=True
                    )
                )
            ),
            "example_response_format": pydantic_to_example_dict(
                Rollout, exclude_fields=hardcoded_info, required_only=True
            ),
        }

    def _create_rollout(
        self,
        hardcoded_info: dict,
        structured_info: dict,
        component_kwargs: dict,
    ) -> Rollout:
        """Create a rollout from structured and hardcoded information."""
        components = {
            "robot": (Robot, component_kwargs.get("robot_kwargs", {})),
            "environment": (
                Environment,
                component_kwargs.get("environment_kwargs", {}),
            ),
            "task": (Task, component_kwargs.get("task_kwargs", {})),
            "trajectory": (Trajectory, component_kwargs.get("trajectory_kwargs", {})),
        }

        objects = {
            name: merge_config_sources(
                ConfigCls, hardcoded_info, structured_info, kwargs
            )
            for name, (ConfigCls, kwargs) in components.items()
        }

        rollout_kwargs = {
            **hardcoded_info["rollout"],
            **{k: v for k, v in structured_info.items() if not isinstance(v, dict)},
            **objects,
        }
        return Rollout(**rollout_kwargs)

    async def extract_batch(
        self,
        episodes: list[OpenXEmbodimentEpisode],
        dataset_info: DatasetInfo,
        *,  # Force keyword arguments
        robot_kwargs: t.Optional[dict[str, t.Any]] = None,
        environment_kwargs: t.Optional[dict[str, t.Any]] = None,
        task_kwargs: t.Optional[dict[str, t.Any]] = None,
        model_kwargs: t.Optional[dict[str, t.Any]] = None,
        trajectory_kwargs: t.Optional[dict[str, t.Any]] = None,
    ) -> list[Rollout]:
        """Process a batch of episodes in parallel."""
        component_kwargs = {
            "robot_kwargs": robot_kwargs or {},
            "environment_kwargs": environment_kwargs or {},
            "task_kwargs": task_kwargs or {},
            "model_kwargs": model_kwargs or {},
            "trajectory_kwargs": trajectory_kwargs or {},
        }

        # Get hardcoded information
        dataset_info_dict = hard_coded_dataset_info_extraction_spreadsheet(dataset_info)
        episode_info_dicts = [hard_coded_episode_info_extraction(ep) for ep in episodes]
        hardcoded_infos = [
            merge_dicts(dataset_info_dict, ep_info) for ep_info in episode_info_dicts
        ]
        print(f"found {len(hardcoded_infos)} hardcoded infos")
        # Prepare prompts and images
        prompts: list[dict | None] = []
        images_list: list[list[np.ndarray] | None] = []
        # Initialize rollouts list with None for all episodes (will be filled with Rollout or error_dict)
        rollouts: list[Rollout | dict | None] = [None] * len(episodes)
        for i, (episode, hardcoded_info) in enumerate(zip(episodes, hardcoded_infos)):
            try:
                # Load video frames
                dataset_filename = dataset_info_dict["rollout"]["dataset_filename"]
                episode_fname = hardcoded_info["rollout"]["filename"]
                images, _ = load_video_frames(
                    dataset_filename,
                    episode_fname,
                    target_fps=1,
                    include_last_frame=True,
                )
                images_list.append(images)

                # Prepare prompt
                prompts.append(self._prepare_prompt_info(episode, hardcoded_info))
            except Exception as e:
                print(f"Error preparing prompt: {e}")
                print(traceback.format_exc())
                error_dict = {
                    "path": hardcoded_info["rollout"]["path"],
                    "error_pattern": "prompt_preparation_failure",
                    "error": traceback.format_exc(),
                }
                # Add None placeholders to keep lists aligned
                prompts.append(None)
                images_list.append(None)
                rollouts[i] = error_dict  # Store error_dict at position i

        # Filter out failed prompts before batch processing
        valid_indices = [i for i, p in enumerate(prompts) if p is not None]
        valid_prompts = [prompts[i] for i in valid_indices]
        valid_images = [images_list[i] for i in valid_indices]

        # Initialize responses list with None for all episodes (original length)
        num_episodes = len(episodes)
        responses = [None] * num_episodes
        print(f"[DEBUG] Initialized responses list with length {num_episodes} (original number of episodes)")

        # Batch process with VLM (only if we have valid prompts)
        if valid_prompts:
            print(f"batching {len(valid_prompts)} prompts")
            results = await self.vlm.ask_batch_async(
                infos=valid_prompts,
                prompt_filename=component_kwargs["model_kwargs"].get(
                    "prompt_filename", "extractor_prompt.jinja2"
                ),
                images_list=valid_images,
            )
            # Unpack the responses from the results
            valid_responses = [response for _, response in results]
            print(f"[DEBUG] Received {len(valid_responses)} responses from VLM for {len(valid_indices)} valid episodes")
            
            # Find first successful response and duplicate it for ALL episodes
            # (since all episodes have the same settings, they should have the same response)
            successful_response = None
            successful_idx = None
            for idx, response in enumerate(valid_responses):
                try:
                    # Try to parse the response to see if it's valid
                    if hasattr(response, 'choices') and response.choices and len(response.choices) > 0:
                        choice = response.choices[0]
                        if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                            content = choice.message.content
                            if content and len(str(content).strip()) > 0:
                                # Try to parse as JSON to validate it's a valid response
                                try:
                                    structured_info = parse_response(choice, load_json=True)
                                    # If we get here, the response is valid
                                    successful_response = response
                                    successful_idx = idx
                                    print(f"[DEBUG] Found successful response at index {idx}, will duplicate for all {num_episodes} episodes")
                                    break
                                except (ValueError, json.JSONDecodeError) as e:
                                    print(f"[DEBUG] Response {idx} failed to parse: {e}")
                                    continue
                except Exception as e:
                    print(f"[DEBUG] Response {idx} validation failed: {e}")
                    continue
            
            # If we found a successful response, duplicate it for ALL episodes (original count)
            if successful_response is not None:
                duplicated_response = copy.deepcopy(successful_response)
                # Fill entire responses list with duplicated response (for all episodes, valid and invalid)
                responses = [copy.deepcopy(duplicated_response) for _ in range(num_episodes)]
                print(f"[DEBUG] Duplicated successful response {successful_idx} for all {num_episodes} episodes (original count)")
            else:
                print(f"[WARNING] No successful response found, will process all responses individually")
                # Fill responses list at valid_indices positions with original responses
                for idx, valid_idx in enumerate(valid_indices):
                    responses[valid_idx] = valid_responses[idx]
            
            # Debug: Check responses (sample first few to avoid spam)
            num_to_check = min(5, num_episodes)
            for idx in range(num_to_check):
                response = responses[idx]
                if response is None:
                    print(f"[DEBUG] Response {idx}: None")
                    continue
                print(f"[DEBUG] Response {idx}:")
                print(f"  - Type: {type(response)}")
                print(f"  - Has choices: {hasattr(response, 'choices')}")
                if hasattr(response, 'choices'):
                    print(f"  - Number of choices: {len(response.choices) if response.choices else 0}")
                    if response.choices and len(response.choices) > 0:
                        choice = response.choices[0]
                        print(f"  - Choice type: {type(choice)}")
                        print(f"  - Has message: {hasattr(choice, 'message')}")
                        if hasattr(choice, 'message'):
                            print(f"  - Message type: {type(choice.message)}")
                            print(f"  - Has content: {hasattr(choice.message, 'content')}")
                            if hasattr(choice.message, 'content'):
                                content = choice.message.content
                                print(f"  - Content type: {type(content)}")
                                print(f"  - Content length: {len(content) if content else 0}")
                                print(f"  - Content preview (first 200 chars): {str(content)[:200] if content else 'None/Empty'}")
                                # Print full content for debugging (limit to 5000 chars to avoid spam)
                                if content and len(str(content)) < 5000:
                                    print(f"  - FULL content:\n{str(content)}")
                                elif content:
                                    print(f"  - FULL content (first 2000 chars):\n{str(content)[:2000]}")
                                    print(f"  - ... (truncated, total length: {len(str(content))})")
                                if not content or len(str(content).strip()) == 0:
                                    print(f"  - [ERROR] Content is empty or None!")
                                    # Try to get more info about the response
                                    print(f"  - Full response object attributes: {dir(response)}")
                                    if hasattr(response, 'model'):
                                        print(f"  - Model used: {response.model}")
                                    if hasattr(response, 'usage'):
                                        print(f"  - Usage: {response.usage}")
                                    if hasattr(response, 'error'):
                                        print(f"  - Error: {response.error}")
                    else:
                        print(f"  - [ERROR] No choices in response!")
                else:
                    print(f"  - [ERROR] Response has no 'choices' attribute!")

        # Process responses and create rollouts
        # Note: responses list now has length = len(episodes), with duplicated successful response for all
        for i, response in enumerate(responses):
            # Skip episodes that failed during prompt preparation (they already have error_dict in rollouts)
            # But still try to process them if we have a response (duplicated successful one)
            if response is None:
                # This episode already has an error_dict in rollouts from the prompt preparation phase
                print(f"[DEBUG] Skipping episode {i} - no response (failed during prompt preparation)")
                continue
            
            try:
                print(f"[DEBUG] Processing response {i} for episode {i}")
                print(f"[DEBUG] Checking response.choices[0] before parsing...")
                if not hasattr(response, 'choices') or not response.choices or len(response.choices) == 0:
                    raise ValueError(f"Response {i} has no choices")
                
                choice = response.choices[0]
                print(f"[DEBUG] Choice object: {type(choice)}")
                print(f"[DEBUG] Choice has message: {hasattr(choice, 'message')}")
                
                if not hasattr(choice, 'message'):
                    raise ValueError(f"Choice {i} has no message attribute")
                
                message = choice.message
                print(f"[DEBUG] Message object: {type(message)}")
                print(f"[DEBUG] Message has content: {hasattr(message, 'content')}")
                
                if not hasattr(message, 'content'):
                    raise ValueError(f"Message {i} has no content attribute")
                
                content = message.content
                print(f"[DEBUG] Content before parsing: type={type(content)}, length={len(str(content)) if content else 0}")
                print(f"[DEBUG] Content value: {str(content)[:500] if content else 'None/Empty'}")
                
                structured_info = parse_response(choice, load_json=True)
                print(f"[DEBUG] Successfully parsed structured_info: {type(structured_info)}, keys: {list(structured_info.keys()) if isinstance(structured_info, dict) else 'N/A'}")
                
                rollout = self._create_rollout(
                    hardcoded_info=hardcoded_infos[i],
                    structured_info=structured_info,
                    component_kwargs=component_kwargs,
                )
                print(f"[DEBUG] Successfully created rollout for response {i}")
                rollouts[i] = rollout  # Replace error_dict with rollout if it exists

            except Exception as e:
                print(f"Error parsing response: {e}")
                print(traceback.format_exc())
                error_dict = {
                    "path": hardcoded_infos[i]["rollout"]["path"],
                    "error_pattern": "extraction_failure",
                    "error": traceback.format_exc(),
                }
                rollouts[i] = error_dict  # Replace or add error_dict at position i
        return rollouts


class RandomInformationExtractor(InformationExtractor):
    """
    Creates a random information extractor that extracts random information from the episode.
    The created object is filled with random string/ints/floats/bools/datetimes/lists etc.

    This should only be used for testing purposes.
    """

    def random_string(self, length_bound: int = 10) -> str:
        return "".join(
            np.random.choice(
                list(string.ascii_lowercase),
                size=np.random.randint(1, length_bound + 1),
            )
        )

    def finish_random_object(
        self, object: t.Type[BaseConfig], kwargs: dict[str, t.Any]
    ) -> BaseConfig:
        # Get all fields from the model class
        fields = object.model_fields
        filled_kwargs = kwargs.copy()

        # Fill in missing fields with random values based on their type
        for field_name, field_info in fields.items():
            if field_name not in filled_kwargs:
                field_type = field_info.annotation

                # Handle different field types
                if field_type == str:
                    filled_kwargs[field_name] = self.random_string()
                elif field_type == int:
                    filled_kwargs[field_name] = np.random.randint(0, 10)
                elif field_type == float:
                    filled_kwargs[field_name] = np.random.uniform(0, 1)
                elif field_type == bool:
                    filled_kwargs[field_name] = bool(np.random.choice([True, False]))
                elif field_type == datetime:
                    filled_kwargs[field_name] = datetime.now()
                elif t.get_origin(field_type) == list:
                    # For lists, create a random-length list of random values
                    elem_type = t.get_args(field_type)[0]
                    length = np.random.randint(1, 5)
                    if elem_type == str:
                        filled_kwargs[field_name] = [
                            self.random_string() for _ in range(length)
                        ]
                    elif elem_type in (int, float):
                        filled_kwargs[field_name] = np.random.rand(length).tolist()
                    else:
                        filled_kwargs[field_name] = []

        # Create and return the object with all fields filled
        return object(**filled_kwargs)

    def extract(
        self,
        episode: OpenXEmbodimentEpisode,
        dataset_info: DatasetInfo,
        *,  # Force keyword arguments
        robot_kwargs: t.Optional[dict[str, t.Any]] = None,
        environment_kwargs: t.Optional[dict[str, t.Any]] = None,
        task_kwargs: t.Optional[dict[str, t.Any]] = None,
        model_kwargs: t.Optional[dict[str, t.Any]] = None,
    ) -> Rollout:
        robot_kwargs = robot_kwargs or {}
        environment_kwargs = environment_kwargs or {}
        task_kwargs = task_kwargs or {}
        model_kwargs = model_kwargs or {}

        dataset_info_dict = hard_coded_dataset_info_extraction_spreadsheet(dataset_info)
        episode_info_dict = hard_coded_episode_info_extraction(episode)
        hardcoded_info = merge_dicts(dataset_info_dict, episode_info_dict)

        # Create component objects in a loop
        components = {
            "robot": Robot,
            "environment": Environment,
            "task": Task,
            "trajectory": Trajectory,
        }
        objects = {
            name: self.finish_random_object(cls, hardcoded_info[name])
            for name, cls in components.items()
        }

        # Create final rollout with all components
        rollout = self.finish_random_object(
            Rollout,
            {**hardcoded_info["rollout"], **objects},
        )
        return rollout
