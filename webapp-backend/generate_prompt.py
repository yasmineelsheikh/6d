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

from typing import Optional

try:
    # litellm is already used elsewhere in the Ares codebase
    from litellm import completion  # type: ignore
except ImportError:  # pragma: no cover
    completion = None  # type: ignore


def generate_prompt_variation(prompt: str, model: str = "gpt-4o-mini") -> str:
    """
    Generate a single prompt variation using an LLM.

    The variation:
    - ONLY changes the aspects that are described inside square brackets in the
      original prompt (e.g. [TABLE], [COLOR_APPLE], [COLOR_BOWL],
      [SENTENCE_LIGHT], [SENTENCE_BACKGROUND]).
    - Keeps the overall sentence structure, ordering, and all other words
      as close to the original as possible.
    - Replaces each bracketed placeholder with a concrete description.
    - Does NOT include any square brackets in the output.

    Parameters
    ----------
    prompt:
        The base scene description prompt containing bracketed placeholders.
    model:
        The LLM model name to use via `litellm.completion`.

    Returns
    -------
    str
        A single, fully-written prompt variation.
    """
    if completion is None:
        # Fallback: if litellm is not available, just return the original prompt.
        # This avoids hard crashes if the environment is not configured yet.
        return prompt

    system_message = (
        "You rewrite scene description prompts for robot videos. "
        "You will be given a base prompt where certain aspects are marked with "
        "square brackets, like [TABLE], [COLOR_APPLE], [COLOR_BOWL], "
        "[SENTENCE_LIGHT], [SENTENCE_BACKGROUND]. "
        "Your job is to produce ONE new prompt variation by changing only the "
        "concrete values for those bracketed aspects.\n\n"
        "Rules:\n"
        "- Keep the overall structure, wording, and ordering of the rest of the "
        "  text as close as possible to the original.\n"
        "- Replace each bracketed token with a specific description.\n"
        "- Do NOT include any square brackets in your output.\n"
        "- Do NOT add meta commentary; return only the final prompt text."
    )

    user_message = (
        "Base prompt:\n\n"
        f"{prompt}\n\n"
        "Now generate ONE variation following the rules above."
    )

    resp = completion(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
    )

    # litellm.completion returns an OpenAI-like response object
    try:
        content: Optional[str] = (
            resp["choices"][0]["message"]["content"]  # type: ignore[index]
        )
    except Exception:
        # If anything unexpected happens, just fall back to original prompt
        return prompt

    return (content or "").strip()


def generate_structured_prompt_from_video(
    video_path: str,
    task_description: str,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Generate a structured scene prompt for a video + task description using a VLM.

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

    
