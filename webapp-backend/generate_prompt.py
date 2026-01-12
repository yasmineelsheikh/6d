"""
Utilities for generating prompt variations using an LLM.

The main entrypoint is `get_prompt`, which takes a base prompt of the form:

    "The scene depicts ... robot faces [TABLE]. On the table rest [COLOR_APPLE] apple
     and [COLOR_BOWL] bowl. [SENTENCE_LIGHT] ... [SENTENCE_BACKGROUND] ..."

and returns a *single variation* where only the aspects inside square brackets
are changed. All other wording and structure should remain as close as possible
to the original.
"""

from __future__ import annotations

from typing import List, Tuple, Optional
from pathlib import Path

from transformers import AutoProcessor, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
import os 
import dashscope
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'
os.environ['DASHSCOPE_API_KEY'] = 'sk-***' # Your DashScope API Key
os.environ['OPENAI_BASE_HTTP_API_URL'] = 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'
DASH_MODEL_ID = '***' # Your model-ID for API
model_path = "***" #  The following output example is from a tiny test model
processor = AutoProcessor.from_pretrained(model_path)

model, output_loading_info = AutoModelForVision2Seq.from_pretrained(model_path, torch_dtype="auto", device_map="auto", output_loading_info=True)
print("output_loading_info", output_loading_info)

try:
    # litellm is already used elsewhere in the Ares codebase
    from litellm import completion  # type: ignore
except ImportError:  # pragma: no cover
    completion = None  # type: ignore

def generate_prompt_variations(
    prompts: List[str],
    axes: List[str],
    model: str = "gpt-4o-mini",
    task_description: Optional[str] = None,
    current_analysis: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """
    Given a list of rich robot video captions and a list of augmentation axes,
    generate one semantics-preserving variation per caption.

    Each variation:
    - Changes exactly ONE axis from `axes` (e.g. 'Weather', 'Lighting', 'Color/Texture').
    - Makes minimal, realistic edits to the scene description only.
    - Keeps the robot's task, goal, and action sequence unchanged.
    - Avoids introducing objects or events that are irrelevant or implausible
      for the given environment (e.g. no elephant in a data center).

    Returns a list of (axis_changed, new_caption) tuples, in the same order
    as the input prompts.
    """

    if completion is None:
        # Fallback: no augmentation if LLM is unavailable.
        return [("None", p) for p in prompts]

    axes_text = ", ".join(axes)
    analysis_text = current_analysis or (
        "No detailed statistics are provided. In general, prefer conditions "
        "that are different from the original caption but still realistic for "
        "the same task and environment type."
    )
    task_text = task_description or (
        "The robot is performing manipulation tasks appropriate to the scene."
    )

    system_message = (
        "You adapt detailed scene descriptions for robot videos to create a more "
        "balanced and diverse dataset.\n\n"
        "You will receive multiple rich, single-paragraph captions. Each caption "
        "describes:\n"
        "- The environment (room or setting), layout, lighting, and background.\n"
        "- The key objects, their appearance, and their positions.\n"
        "- The robot's actions over time as a coherent sequence.\n\n"
        "Your job is to generate ONE new caption for EACH original caption, such that:\n"
        "- You change exactly ONE high-level augmentation axis per caption.\n"
        "- You pick that axis from the provided list of augmentation axes.\n"
        "- Your changes are minimal, realistic, and consistent with the environment.\n"
        "- The robot's task, goal, and action sequence remain unchanged.\n"
        "- You do NOT introduce implausible or irrelevant elements for the setting "
        "- The new caption should remain at a similar level of detail and length "
        "  (4–7 sentences, vivid but grounded prose).\n\n"
        "Your overall objective is that the combined set of original and new captions "
        "covers a broader, more balanced range of conditions (lighting, weather, "
        "appearance, etc.) for the same tasks."
    )

    user_message = (
        "Here is the high-level robot task context:\n"
        f"{task_text}\n\n"
        "Here is an analysis of the current dataset distribution and what is "
        "over- or underrepresented:\n"
        f"{analysis_text}\n\n"
        "You may change exactly ONE axis per caption, chosen from this list:\n"
        f"{axes_text}\n\n"
        "Original captions (one per line, numbered):\n"
    )

    for i, p in enumerate(prompts, start=1):
        user_message += f"{i}. {p}\n"

    user_message += (
        "\nFor EACH original caption, produce exactly ONE adapted caption.\n"
        "For every adapted caption, choose one axis from the list that you changed.\n\n"
        "Output format:\n"
        "- Return a valid Python list of tuples.\n"
        "- Each tuple must be of the form: (\"AxisName\", \"<adapted caption>\").\n"
        "- The adapted caption should be a single paragraph with 4–7 sentences.\n"
        "- Do NOT include the original captions in the output.\n"
        "- Do NOT include any commentary, explanations, or keys.\n\n"
        "Example (structure only):\n"
        "[\n"
        "  (\"Lighting\", \"The scene ...\"),\n"
        "  (\"Weather\", \"The environment ...\"),\n"
        "]\n\n"
        "Now generate the list for all captions given above."
    )

    resp = completion(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
    )

    try:
        content: Optional[str] = resp["choices"][0]["message"]["content"]  # type: ignore[index]
    except Exception:
        # If anything unexpected happens, fall back to identity mapping.
        return [("None", p) for p in prompts]

    raw = (content or "").strip()

    # Parse the Python list of tuples safely.
    import ast
    try:
        parsed = ast.literal_eval(raw)
        # Basic sanity check: list of (axis, caption) pairs
        out: List[Tuple[str, str]] = []
        for item in parsed:
            if (
                isinstance(item, (list, tuple))
                and len(item) == 2
                and isinstance(item[0], str)
                and isinstance(item[1], str)
            ):
                out.append((item[0], item[1].strip()))
        # If parsing fails or output length mismatches, degrade gracefully.
        if len(out) != len(prompts):
            return [("None", p) for p in prompts]
        return out
    except Exception:
        return [("None", p) for p in prompts]


def generate_structured_prompt_from_video(
    video_path: str,
    task_description: str,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Describe this video in detail.

    The returned prompt should follow the structure of:

        The scene depicts a bright, modern kitchen with plenty of ambient light.
        From a first-person perspective, a robot faces [TABLE]. On the table rest
        [COLOR_APPLE] apple and [COLOR_BOWL] bowl. [SENTENCE_LIGHT] In the
        background are a black cooking range featuring a black stovetop, wooden
        countertops, and cabinetry with white doors and drawers, including a built-in
        white dishwasher on the left. [SENTENCE_BACKGROUND] A wide black curtain hangs
        vertically on the right side, covering a large portion of the space. As the
        video progresses, the robot picks up the apple, then the bowl, places the
        apple into the bowl, and sets the bowl down on the table.

    Constraints:
    - The parts that describe the *task* (what the robot does, i.e. the action
      sequence) should be written out in full and MUST NOT be inside square brackets.
      They should be derived from `task_description`.
    - Other aspects of the scene (table, colors, lighting sentence, background
      sentence, etc.) MAY be represented with square-bracket placeholders such as
      [TABLE], [COLOR_APPLE], [COLOR_BOWL], [SENTENCE_LIGHT], [SENTENCE_BACKGROUND].
    - The output must be a single coherent prompt string.

    Notes:
    - This function currently passes `video_path` as text context to the model.
      In a future iteration, you can replace this with actual frame extraction and
      pass images/video to a true VLM endpoint via `litellm`.
    """
    if completion is None:
        # Fallback: construct a minimal structured prompt using the task description
        base = (
            "The scene depicts a workspace suitable for robotic manipulation. "
            "From a first-person perspective, a robot faces [TABLE]. On the surface rest "
            "[OBJECTS]. [SENTENCE_LIGHT] In the background there are shelves and tools. "
            "[SENTENCE_BACKGROUND] As the video progresses, "
        )
        return base + task_description.strip()

    system_message = (
        "You are a vision-language model that writes structured scene prompts for robot videos.\n\n"
        "You must output ONE prompt that closely follows this structure:\n\n"
        "The scene depicts a bright, modern kitchen with plenty of ambient light. "
        "From a first-person perspective, a robot faces [TABLE]. On the table rest "
        "[COLOR_APPLE] apple and [COLOR_BOWL] bowl. [SENTENCE_LIGHT] In the "
        "background are a black cooking range featuring a black stovetop, wooden "
        "countertops, and cabinetry with white doors and drawers, including a built-in "
        "white dishwasher on the left. [SENTENCE_BACKGROUND] A wide black curtain hangs "
        "vertically on the right side, covering a large portion of the space. As the "
        "video progresses, the robot picks up the apple, then the bowl, places the "
        "apple into the bowl, and sets the bowl down on the table.\n\n"
        "Rules:\n"
        "- Use a similar sentence structure and level of detail.\n"
        "- Use square-bracket placeholders like [TABLE], [COLOR_APPLE], [COLOR_BOWL], "
        "  [SENTENCE_LIGHT], [SENTENCE_BACKGROUND] (and similar) for *scene* aspects "
        "  that could be varied later.\n"
        "- The parts that describe the *task* (what the robot does over time) must be "
        "  written out concretely using the provided task description and MUST NOT be "
        "  inside square brackets.\n"
        "- Do NOT include any meta commentary. Return only the final prompt text."
    )

    user_message = (
        "We have a robot video at the following path (for your context):\n"
        f"{video_path}\n\n"
        "The high-level task the robot is performing is:\n"
        f"{task_description}\n\n"
        "Using this information, write ONE prompt following the structure and rules "
        "described above."
    )

    resp = completion(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
    )

    try:
        content: Optional[str] = (
            resp["choices"][0]["message"]["content"]  # type: ignore[index]
        )
    except Exception:
        # Fallback: simple structured prompt with inline task description
        base = (
            "The scene depicts a workspace suitable for robotic manipulation. "
            "From a first-person perspective, a robot faces [TABLE]. On the surface rest "
            "[OBJECTS]. [SENTENCE_LIGHT] In the background there are shelves and tools. "
            "[SENTENCE_BACKGROUND] As the video progresses, "
        )
        return base + task_description.strip()

    return (content or "").strip()

def generate_video_description(directory_path: str, task_description: Optional[str] = None) -> List[str]:
    """
    Use the local Qwen VLM to generate rich, structured descriptions for all videos in a directory.

    Iterates through each video file in the specified directory and generates a description
    for each one using the Qwen VLM model.

    Parameters
    ----------
    directory_path:
        Path to the directory containing video files.
    task_description:
        Optional high-level task description that applies to all videos.

    Returns
    -------
    List[str]
        A list of generated descriptions, one per video file found in the directory.
        Descriptions are returned in alphabetical order by filename.

    Notes
    -----
    Supported video file extensions: .mp4, .avi, .mov, .mkv, .webm, .flv, .m4v
    """
    # Configure sampling: e.g., 0.25 FPS = 1 frame per 4 seconds
    sample_fps = 0.25

    # The maximum number of pixels expected to be used from the video — adjustable based on available GPU memory.
    total_pixels = 24 * 1024 * 32 * 32

    base_prompt = """You are a vision-language model describing a short robot training video. Write one vivid, natural paragraph that gives a complete mental picture of the scene and the robot's actions.
Your description must include:

Scene layout and spatial context: Identify the environment type (e.g., kitchen, lab, desk, workshop) and describe its layout and size impression. Mention key background features such as walls, furniture, appliances, shelves, or windows.

Lighting and atmosphere: Describe lighting direction, brightness, and tone (warm, cool, daylight, artificial) and any reflections or shadows that affect the scene's mood.

Camera or viewpoint: Indicate whether the view is first-person (robot POV), third-person, or fixed overhead, and describe depth or perspective if relevant.

Objects and properties: List the main visible objects — especially those used by the robot — with their approximate color, shape, size, and position (e.g., "a red apple on a white ceramic plate to the robot's left"). Emphasize textures or materials when they contribute to realism.

Action and temporal flow: Describe the sequence of robot actions across time: how it reaches, grasps, moves, or manipulates objects, and what the final outcome is. Use verbs that convey smooth motion and cause–effect transitions.

Output requirements:

Produce one cohesive paragraph of 4–7 sentences.

The first 2–3 sentences should focus on static scene details (environment, lighting, object arrangement).

The remaining sentences should narrate the robot's actions as a clear temporal sequence.

Avoid giving lists, bullet points, numbers, or JSON.

Do not mention "video," "frames," or "camera footage."

Write naturally as if describing a real observation, not as an instruction or program output.

If a high-level task is provided, integrate it seamlessly into the action portion so it is naturally expressed through what the robot does."""

    if task_description:
        task_clause = (
            "\n\nThe high-level task the robot is performing is:\n"
            f"{task_description}\n\n"
            "Make sure the final part of your paragraph clearly reflects this task, while still following the rules above."
        )
    else:
        task_clause = (
            "\n\nIf there is a clear task or goal for the robot, make sure the action part of your paragraph clearly reflects it."
        )

    prompt = base_prompt + task_clause

    # Supported video file extensions
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v'}

    # Get directory path and find all video files
    dir_path = Path(directory_path)
    if not dir_path.exists():
        raise ValueError(f"Directory does not exist: {directory_path}")
    if not dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory_path}")

    # Find all video files in the directory (non-recursive)
    video_files = sorted([f for f in dir_path.iterdir() if f.is_file() and f.suffix.lower() in video_extensions])

    if not video_files:
        print(f"Warning: No video files found in directory: {directory_path}")
        return []

    # Generate description for each video
    descriptions: List[str] = []
    total_videos = len(video_files)
    
    for idx, video_path in enumerate(video_files, 1):
        try:
            print(f"Processing video {idx}/{total_videos}: {video_path.name}")
            # Convert Path to string for inference function
            video_path_str = str(video_path.absolute())
            response = inference(video_path_str, prompt, sample_fps=sample_fps, total_pixels=total_pixels)
            descriptions.append(response.strip())
            print(f"Successfully generated description for {video_path.name}")
        except Exception as e:
            print(f"Error processing video {video_path.name}: {e}")
            import traceback
            traceback.print_exc()
            # Add empty string to maintain list length/indexing
            descriptions.append("")
    
    return descriptions

def inference(video, prompt, max_new_tokens=2048, total_pixels=20480 * 32 * 32, min_pixels=64 * 32 * 32, max_frames= 2048, sample_fps = 2):
    """
    Perform multimodal inference on input video and text prompt to generate model response.

    Args:
        video (str or list/tuple): Video input, supports two formats:
            - str: Path or URL to a video file. The function will automatically read and sample frames.
            - list/tuple: Pre-sampled list of video frames (PIL.Image or url). 
              In this case, `sample_fps` indicates the frame rate at which these frames were sampled from the original video.
        prompt (str): User text prompt to guide the model's generation.
        max_new_tokens (int, optional): Maximum number of tokens to generate. Default is 2048.
        total_pixels (int, optional): Maximum total pixels for video frame resizing (upper bound). Default is 20480*32*32.
        min_pixels (int, optional): Minimum total pixels for video frame resizing (lower bound). Default is 16*32*32.
        sample_fps (int, optional): ONLY effective when `video` is a list/tuple of frames!
            Specifies the original sampling frame rate (FPS) from which the frame list was extracted.
            Used for temporal alignment or normalization in the model. Default is 2.

    Returns:
        str: Generated text response from the model.

    Notes:
        - When `video` is a string (path/URL), `sample_fps` is ignored and will be overridden by the video reader backend.
        - When `video` is a frame list, `sample_fps` informs the model of the original sampling rate to help understand temporal density.
    """

    messages = [
        {"role": "user", "content": [
                {"video": video,
                "total_pixels": total_pixels, 
                "min_pixels": min_pixels, 
                "max_frames": max_frames,
                'sample_fps':sample_fps},
                {"type": "text", "text": prompt},
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True, 
                                                                   image_patch_size= 16,
                                                                   return_video_metadata=True)
    if video_inputs is not None:
        video_inputs, video_metadatas = zip(*video_inputs)
        video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
    else:
        video_metadatas = None
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, video_metadata=video_metadatas, **video_kwargs, do_resize=False, return_tensors="pt")
    inputs = inputs.to('cuda')

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]