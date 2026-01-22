import asyncio
import base64
import json
import os
import typing as t
from asyncio import Semaphore
from contextlib import nullcontext

import numpy as np
import torch
#import vertexai
from jinja2 import Environment, FileSystemLoader
from litellm import acompletion, completion
from litellm.utils import ModelResponse
from PIL import Image
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential
from transformers import AutoModel, AutoProcessor
#from vertexai.generative_models import GenerativeModel, Part

from ares.utils.image_utils import encode_image

# dependent on your key / organization tier
RATE_LIMITS = {
    "openai": 10,  # 5000 RPM
    "anthropic": 2,  # 1000 RPM
    "gemini": 4,  # 1000 RPM
}

MAX_RETRIES = 3
INITIAL_WAIT_SECONDS = 1
MAX_WAIT_SECONDS = 10


def structure_image_messages(
    image: t.Union[str, np.ndarray, Image.Image], provider: str
) -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image)}"},
    }


class VLM:
    def __init__(self, provider: str, name: str):
        self.provider = provider
        self.name = name
        self.full_name = f"{provider}/{name}"
        # Add rate limiting per provider
        self.semaphore: Semaphore | nullcontext = nullcontext()
        self._setup_rate_limits()
        self.check_valid_key()

    def _setup_rate_limits(self, max_concurrent: int | None = None) -> None:
        max_concurrent = max_concurrent or RATE_LIMITS.get(self.provider, 10)
        self.semaphore = Semaphore(max_concurrent)

    def check_valid_key(self) -> bool:
        # note: don't use litellm util here, it uses 10 tokens! roll our own check with 1
        try:
            res = completion(
                model=self.full_name,
                messages=[{"role": "user", "content": "!"}],
                max_tokens=1,
            )
        except Exception as e:
            print(f"Error checking valid key: {e}")
            return False
        return True

    def _get_prompt(self, prompt_filename: str, info: dict) -> str:
        # Set up Jinja environment to load templates from src/ares/models/prompts directory
        jinja_env = Environment(
            loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), "prompts"))
        )
        template = jinja_env.get_template(prompt_filename)
        # Render template with information
        return template.render(**info)

    def _construct_messages(
        self,
        info: dict,
        prompt_filename: str | None = None,
        images: t.Sequence[t.Union[str, np.ndarray, Image.Image]] | None = None,
        double_prompt: bool = False,
    ) -> list[dict[str, t.Any]]:
        content: list[dict[str, t.Any]] = []
        if "prompt" not in info and prompt_filename is not None:
            prompt = self._get_prompt(prompt_filename, info)
        else:
            prompt = info["prompt"]
        content.append({"type": "text", "text": prompt})
        if images is not None and len(images) > 0:
            image_contents = []
            for i in range(len(images)):
                text_content = {"type": "text", "text": f"Image {i}:"}
                image_content = structure_image_messages(
                    images[i], provider=self.provider
                )
                image_contents.extend([text_content, image_content])
            content.extend(image_contents)
        if double_prompt:
            content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=INITIAL_WAIT_SECONDS, max=MAX_WAIT_SECONDS),
        reraise=True,
    )
    async def _make_api_call(
        self, messages: list[dict], model_kwargs: dict
    ) -> ModelResponse:
        """Wrapper for API calls with retry logic"""
        print(f"[DEBUG] Making API call to model: {self.full_name}")
        print(f"[DEBUG] Number of messages: {len(messages)}")
        print(f"[DEBUG] Model kwargs: {model_kwargs}")
        
        try:
            response = await acompletion(
            model=self.full_name, messages=messages, **model_kwargs
        )
            print(f"[DEBUG] API call successful, response type: {type(response)}")
            print(f"[DEBUG] Response has choices: {hasattr(response, 'choices')}")
            if hasattr(response, 'choices') and response.choices:
                print(f"[DEBUG] Number of choices: {len(response.choices)}")
                if len(response.choices) > 0:
                    choice = response.choices[0]
                    if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                        content = choice.message.content
            print(f"[DEBUG] First choice content length: {len(str(content)) if content else 0}")
            print(f"[DEBUG] First choice content preview: {str(content)[:200] if content else 'None/Empty'}")
            # Print full content if it's not too long (for debugging)
            if content and len(str(content)) < 5000:
                print(f"[DEBUG] First choice FULL content:\n{str(content)}")
            else:
                print(f"[WARNING] Response has no choices or choices is empty")
                print(f"[DEBUG] Response attributes: {dir(response)}")
                if hasattr(response, 'error'):
                    print(f"[ERROR] Response has error attribute: {response.error}")
            
            return response
        except Exception as e:
            print(f"[ERROR] API call failed: {type(e).__name__}: {str(e)}")
            import traceback
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            raise

    async def ask_async(
        self,
        info: dict,
        prompt_filename: str | None = None,
        images: t.Sequence[t.Union[str, np.ndarray, Image.Image]] | None = None,
        video_path: str | None = None,
        double_prompt: bool = False,
        model_kwargs: dict | None = None,
    ) -> tuple[list[dict[str, t.Any]], ModelResponse]:
        """Rate-limited async version of ask method"""
        async with self.semaphore:
            print(f"[DEBUG] ask_async called with prompt_filename={prompt_filename}")
            print(f"[DEBUG] Info keys: {list(info.keys()) if isinstance(info, dict) else 'N/A'}")
            print(f"[DEBUG] Number of images: {len(images) if images else 0}")
            
            model_kwargs = model_kwargs or dict()
            if video_path:
                raise NotImplementedError("Video path not implemented for this VLM")
            
            messages = self._construct_messages(
                info, prompt_filename, images, double_prompt
            )
            print(f"[DEBUG] Constructed {len(messages)} messages")
            print(f"[DEBUG] First message role: {messages[0].get('role') if messages else 'N/A'}")
            if messages and 'content' in messages[0]:
                content = messages[0]['content']
                print(f"[DEBUG] First message content type: {type(content)}")
                if isinstance(content, list):
                    print(f"[DEBUG] Content list length: {len(content)}")
                    for idx, item in enumerate(content[:3]):  # Show first 3 items
                        print(f"[DEBUG] Content item {idx} type: {type(item)}, keys: {list(item.keys()) if isinstance(item, dict) else 'N/A'}")
            
            response = await self._make_api_call(messages, model_kwargs)
            return messages, response

    def ask(
        self,
        info: dict,
        prompt_filename: str | None = None,
        images: t.Sequence[t.Union[str, np.ndarray, Image.Image]] | None = None,
        video_path: str | None = None,
        double_prompt: bool = False,
        model_kwargs: dict | None = None,
    ) -> tuple[list[dict[str, t.Any]], ModelResponse]:
        return asyncio.run(
            self.ask_async(
                info,
                prompt_filename,
                images,
                video_path,
                double_prompt,
                model_kwargs,
            )
        )

    async def ask_batch_async(
        self,
        infos: list[dict],
        prompt_filename: str | None = None,
        images_list: (
            list[t.Sequence[t.Union[str, np.ndarray, Image.Image]]] | None
        ) = None,
        double_prompt: bool = False,
        model_kwargs: dict | None = None,
    ) -> list[tuple[list[dict[str, t.Any]], ModelResponse]]:
        """Process multiple requests with rate limiting"""
        if images_list is None:
            images_list = [None] * len(infos)

        tasks = [
            self.ask_async(
                info,
                prompt_filename=prompt_filename,
                images=images,
                double_prompt=double_prompt,
                model_kwargs=model_kwargs,
            )
            for info, images in zip(infos, images_list)
        ]

        return await asyncio.gather(*tasks)


"""class GeminiVideoVLM(VLM):
    def __init__(self, provider: str, name: str):
        super().__init__(provider, name)
        vertexai.init(
            project=os.environ["VERTEX_PROJECT"], location=os.environ["VERTEX_LOCATION"]
        )
        self.model = GenerativeModel(name)
        self.generation_config = {
            "max_output_tokens": 8192,
            "temperature": 1,
            "top_p": 0.95,
        }

    def ask(
        self,
        info: dict,
        prompt_filename: str | None = None,
        images: t.Sequence[t.Union[str, np.ndarray, Image.Image]] | None = None,
        video_path: str | None = None,
        double_prompt: bool = False,
        model_kwargs: dict | None = None,
    ) -> tuple[list[dict[str, t.Any]], ModelResponse]:
        model_kwargs = model_kwargs or dict()
        prompt = self._get_prompt(prompt_filename, info)
        if video_path:
            if "https://" in video_path:
                video = Part.from_uri(
                    mime_type="video/mp4",
                    uri=video_path,
                )
            else:
                encoded_video = base64.b64encode(open(video_path, "rb").read()).decode(
                    "utf-8"
                )
                video = Part.from_data(
                    data=base64.b64decode(encoded_video), mime_type="video/mp4"
                )
        messages = [prompt, video]
        if double_prompt:
            messages.append(prompt)
        responses = self.model.generate_content(
            messages,
            generation_config=self.generation_config,
        )
        return messages, responses """


class Embedder:
    def __init__(self, provider: str, name: str):
        self.provider = provider
        self.name = name
        self._setup_model()

    def _setup_model(self) -> None:
        """Initialize model and processor. Should be implemented by child classes."""
        name = f"{self.provider}/{self.name}"
        self.processor = AutoProcessor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name)

    def _process_image_input(
        self, inp: t.Union[Image.Image, np.ndarray]
    ) -> Image.Image:
        """Convert various image input types to PIL Image."""
        if isinstance(inp, np.ndarray):
            return Image.fromarray(inp)
        elif isinstance(inp, str):
            return Image.open(inp)
        return inp

    def embed(self, inp: t.Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """Embed text or image input."""
        with torch.inference_mode():
            if isinstance(inp, str):
                # Text embedding
                inputs = self.processor(text=[inp], padding=True, return_tensors="pt")
                outputs = self.model.get_text_features(**inputs)
                return outputs.detach().numpy()[0]
            else:
                # Image embedding
                image = self._process_image_input(inp)
                inputs = self.processor(images=image, return_tensors="pt")
                outputs = self.model.get_image_features(**inputs)
                return outputs.detach().numpy()[0]


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, provider: str, name: str):
        super().__init__(provider, name)

    def _setup_model(self) -> None:
        self.model = SentenceTransformer(
            f"{self.provider}/{self.name}", trust_remote_code=True
        )
        self.name = self.model.model_card_data.base_model

    def embed(self, inp: t.Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        if not isinstance(inp, str):
            raise ValueError("Input must be a string for SentenceTransformerEmbedder")
        prefix = "clustering: "
        return np.array(self.model.encode(prefix + inp))


def parse_response(choice: t.Any, load_json: bool = False) -> dict | str:
    print(f"[DEBUG] parse_response called with load_json={load_json}")
    print(f"[DEBUG] Choice type: {type(choice)}")
    print(f"[DEBUG] Choice has message: {hasattr(choice, 'message')}")
    
    if not hasattr(choice, 'message'):
        raise ValueError(f"Choice object has no 'message' attribute. Choice type: {type(choice)}, attributes: {dir(choice)}")
    
    message = choice.message
    print(f"[DEBUG] Message type: {type(message)}")
    print(f"[DEBUG] Message has content: {hasattr(message, 'content')}")
    
    if not hasattr(message, 'content'):
        raise ValueError(f"Message object has no 'content' attribute. Message type: {type(message)}, attributes: {dir(message)}")
    
    content: str = message.content
    print(f"[DEBUG] Raw content type: {type(content)}, value: {str(content)[:200] if content else 'None/Empty'}")
    print(f"[DEBUG] Content is None: {content is None}")
    print(f"[DEBUG] Content is empty string: {content == ''}")
    print(f"[DEBUG] Content length: {len(str(content)) if content else 0}")
    
    if content is None:
        raise ValueError("Content is None - VLM returned no response")
    if content == '':
        raise ValueError("Content is empty string - VLM returned empty response")
    
    if load_json:
        print(f"[DEBUG] Processing JSON content...")
        content_stripped = content.strip()
        print(f"[DEBUG] After strip, length: {len(content_stripped)}")
        
        # Check if VLM is refusing to analyze (check BEFORE extraction)
        refusal_indicators = [
            "i'm sorry",
            "i can't",
            "i cannot",
            "unable to",
            "can't assist",
            "can't directly",
            "can't view",
            "can't analyze",
        ]
        content_lower = content_stripped.lower()
        if any(indicator in content_lower[:300] for indicator in refusal_indicators):
            print(f"[WARNING] VLM response appears to be refusing to analyze images")
            print(f"[WARNING] Full response: {content_stripped[:1000]}")
            raise ValueError(f"VLM refused to analyze content. This may be due to content policy or image encoding issues. Response: {content_stripped[:500]}")
        
        # Try multiple strategies to extract JSON:
        import re
        # 1. Look for ```json ... ``` block (even if there's text before it)
        json_match = re.search(r'```json\s*(.*?)\s*```', content_stripped, re.DOTALL)
        if json_match:
            print(f"[DEBUG] Found JSON in markdown code block with json tag")
            content_stripped = json_match.group(1).strip()
        else:
            # 2. Look for ``` ... ``` block (without json tag) containing JSON
            json_match = re.search(r'```\s*(\{.*?\})\s*```', content_stripped, re.DOTALL)
            if json_match:
                print(f"[DEBUG] Found JSON in code block without json tag")
                content_stripped = json_match.group(1).strip()
            else:
                # 3. Look for JSON object directly (starts with { and ends with })
                # Use a more robust regex that handles nested braces
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content_stripped, re.DOTALL)
                if json_match:
                    print(f"[DEBUG] Found JSON object in text (no code blocks)")
                    content_stripped = json_match.group(0).strip()
                else:
                    # 4. Fallback: try simple prefix/suffix removal
                    content_stripped = content_stripped.removeprefix("```json").removesuffix("```").strip()
        
        print(f"[DEBUG] After extracting JSON, length: {len(content_stripped)}")
        print(f"[DEBUG] Content to parse (first 500 chars): {content_stripped[:500]}")
        
        if not content_stripped:
            raise ValueError("Content is empty after extracting JSON - cannot parse JSON")
        
        try:
            parsed = json.loads(content_stripped) if isinstance(content_stripped, str) else content_stripped
            print(f"[DEBUG] Successfully parsed JSON, type: {type(parsed)}, keys: {list(parsed.keys()) if isinstance(parsed, dict) else 'N/A'}")
            return parsed
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON decode error: {e}")
            print(f"[ERROR] Error position: {e.pos if hasattr(e, 'pos') else 'unknown'}")
            print(f"[ERROR] Failed to parse content (first 1000 chars): {content_stripped[:1000]}")
            
            # Try to find and extract JSON starting from first {
            if '{' in content_stripped:
                json_start = content_stripped.find('{')
                print(f"[DEBUG] Found '{{' at position {json_start}, trying to extract from there...")
                # Try to find matching closing brace by counting braces
                brace_count = 0
                json_end = json_start
                for i in range(json_start, len(content_stripped)):
                    if content_stripped[i] == '{':
                        brace_count += 1
                    elif content_stripped[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                
                if brace_count == 0:
                    json_extract = content_stripped[json_start:json_end]
                    print(f"[DEBUG] Extracted JSON substring (length {len(json_extract)})")
                    try:
                        parsed = json.loads(json_extract)
                        print(f"[DEBUG] Successfully parsed extracted JSON")
                        return parsed
                    except json.JSONDecodeError as e2:
                        print(f"[ERROR] Still failed to parse extracted JSON: {e2}")
            
            raise ValueError(f"Failed to parse JSON from content: {e}. Content preview: {content_stripped[:200]}")
    
    return content


def parse_responses(res: ModelResponse, load_json: bool = False) -> list[dict | str]:
    outputs = []
    for choice in res.choices:
        try:
            outputs.append(parse_response(choice, load_json))
        except Exception as e:
            print(f"Failed to parse JSON from response: {e}")
            print(f"Response: {choice.message.content}")
            outputs.append(choice.message.content)
    return outputs
