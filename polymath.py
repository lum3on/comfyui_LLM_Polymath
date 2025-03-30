import subprocess
import sys
import json
import os, glob
import re
import base64
import numpy as np
from io import BytesIO
from PIL import Image

# For the Websearch
from googlesearch import search
import requests
from bs4 import BeautifulSoup

# For the Scraper
import torch
import folder_paths
import shutil

api_key_oai = os.getenv("OPENAI_API_KEY")
api_key_anthropic = os.getenv("ANTHROPIC_API_KEY")
api_key_xai = os.getenv("XAI_API_KEY")
api_key_deepseek = os.getenv("DEEPSEEK_API_KEY")
api_key_gemini = os.getenv("GEMINI_API_KEY")

# Get the node list
script_dir = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(script_dir, "../ComfyUI-Manager/custom-node-list.json")
json_file_path = os.path.normpath(json_file_path)

# Get the directory of the current script (Polymath.py)
script_directory = os.path.dirname(os.path.abspath(__file__))
custom_instructions_directory = os.path.join(script_directory, 'custom_instructions')

# Construct the full path to config.json
config_path = os.path.join(script_directory, 'config.json')

def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        print(f"Package {package} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    finally:
        globals()[package] = __import__(package)

install_and_import("googlesearch")
install_and_import("requests")
install_and_import("bs4")

script_directory = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_directory, 'config.json')
ollama_url = "http://127.0.0.1:11434/api/tags"

def load_models():
    config_models = {}
    base_urls = {}
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            # Iterate over all configuration entries
            for conf in config_data:
                models = conf.get('models', {})
                config_models.update(models)
                # Store each baseurl keyed by something, e.g., the first model name
                for name in models.keys():
                    base_urls[name] = conf.get('baseurl')
    except FileNotFoundError:
        print(f"Error: config.json not found at {config_path}")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {config_path}")

    # Fetch Ollama models (as before)
    ollama_models = {}
    try:
        response = requests.get(ollama_url)
        response.raise_for_status()
        models = response.json().get('models', [])
        ollama_models = {
            "Ollama: " + " ".join(model['name'].split(':')): model['name']
        for model in models
        }
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Ollama models: {e}")

    merged_models = {**config_models, **ollama_models}
    models_list = list(merged_models.keys())
    default_model = models_list[0] if models_list else None

    print("\033[92mPolymath loaded these models: \033[0m", f"\033[38;5;99m{models_list}\033[0m")
    return base_urls, merged_models, models_list, default_model

# Load models and set default
base_url, models_dict, models_list, default_model = load_models()

def load_custom_instructions():
    instructions = ["None"]
    if not os.path.exists(custom_instructions_directory):
        os.makedirs(custom_instructions_directory)
    for filename in os.listdir(custom_instructions_directory):
        if filename.endswith(".txt"):
            instructions.append(filename[:-4])
    return instructions

custom_instructions_list = load_custom_instructions()

def fetch_and_extract_content(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text_content = "\n".join([p.text for p in paragraphs])
        return text_content.strip()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return None
    except Exception as e:
        print(f"Error processing content from {url}: {e}")
        return None

class OllamaAPI:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        self.chat_history = []

    def generate_completion(self, model, prompt, stream=False, ollama_chat_mode=False, images=None, seed=42, options=None):
        url = f"{self.base_url}/api/chat" if ollama_chat_mode else f"{self.base_url}/api/generate"

        payload_options = options if isinstance(options, dict) else {} # Ensure options is a dict
        payload_options['seed'] = seed

        if ollama_chat_mode:
            # Append the new user message to the chat history
            self.chat_history.append({"role": "user", "content": prompt})
            payload = {
                "model": model,
                "messages": self.chat_history,  # Include the entire chat history
                "stream": stream,
                "options": payload_options
            }
        else:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": stream,
                "options": payload_options
            }

        if images:
            payload["images"] = images
        if options:
            payload["temperature"] = options.get("temperature", 0.8)
            payload["top_p"] = options.get("top_p", 0.95)
            payload["top_k"] = options.get("top_k", 40)
            payload["num_predict"] = options.get("max_tokens", -1)
            payload["seed"] = options.get("seed", 42)
            payload["repeat_penalty"] = options.get("repeat_penalty", 1.1)
            payload["repeat_last_n"] = options.get("repeat_last_n", 64)
            payload["keep_alive"] = options.get("ollama_keep_alive", 5)

        response = requests.post(url, json=payload)
        response.raise_for_status()
        response_json = response.json()

        if ollama_chat_mode:
            # Append the model's response to the chat history
            self.chat_history.append({"role": "assistant", "content": response_json.get('message', {}).get('content', '')})

        return response_json

class PolymathSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 100}),
                "max_tokens": ("INT", {"default": 1024, "min": -1, "max": 8192}), # -1 might mean 'unlimited' or 'default'
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 2.0, "step": 0.05}),
                "repeat_last_n": ("INT", {"default": 64, "min": 0, "max": 2048}),
                "response_format_json": ("BOOLEAN", {"default": False}),
                "ollama_keep_alive": ("INT", {"default": 5, "min": 1, "max": 10}),
                "request_timeout": ("INT", {"default": 120, "min": 0, "max": 600}),
                 "dalle_quality": (["standard", "hd"], {"default":"standard"}),
                 "dalle_style": (["vivid", "natural"], {"default":"vivid"}),
                 "dalle_size": (["1024x1024", "1792x1024", "1024x1792"], {"default":"1024x1024"}),
                 "dalle_n": ("INT", {"default": 1, "min": 1, "max": 4})
            }
        }
    # Output a dictionary or a specific tuple/pipe format
    RETURN_TYPES = ("LLM_SETTINGS",) # Or use "BASIC_PIPE" if it fits
    FUNCTION = "get_settings"
    CATEGORY = "Polymath/Settings"

    def get_settings(self, **kwargs):
        # Package settings into a dictionary for easy use
        return (kwargs,) # Return as a tuple containing the dict

class Polymath:
    chat_history = []
    default_model = None
    ollama_api = OllamaAPI()  # Class-level instance of OllamaAPI

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "default": "Enter your prompt here. Use {additional_text} as a placeholder if needed.",
                        "multiline": True
                    }
                ),
                "additional_text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True
                    }
                ),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffff, "step": 1}),
            },
            "optional": {
                "llm_settings": ("LLM_SETTINGS",),
                "model": (models_list, {"default": default_model}),
                "custom_instruction": (custom_instructions_list, {"default": "None"}),
                "enable_web_search": ("BOOLEAN", {"default": False}),
                "list_sources": ("BOOLEAN", {"default": False}),
                "num_search_results": ("INT", {"default": 5, "min": 1, "max": 10}),
                "keep_context": ("BOOLEAN", {"default": True}),
                "ollama_chat_mode": ("BOOLEAN", {"default": False}),
                "compress": ("BOOLEAN", {"default": False}),
                "compression_level": (["soft", "medium", "hard"],),
                "console_log": ("BOOLEAN", {"default": True}),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING","IMAGE")
    FUNCTION = "execute"
    CATEGORY = "Polymath"

    def tensor_to_pil(self, img_tensors):
        def to_pil(img_tensor):
            arr = img_tensor.cpu().numpy() if hasattr(img_tensor, "cpu") else np.array(img_tensor)
            arr = np.squeeze(arr)
            if arr.ndim == 3 and arr.shape[0] == 3:
                arr = np.transpose(arr, (1, 2, 0))
            arr = np.clip(255.0 * arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr)
        
        if isinstance(img_tensors, (list, tuple)):
            return [to_pil(t) for t in img_tensors]
        elif hasattr(img_tensors, 'ndim') and img_tensors.ndim == 4:  # Batch tensor
            return [to_pil(t) for t in img_tensors]
        else:
            return to_pil(img_tensors)

    def encode_image(self, image):
        def encode_single(img):
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        if isinstance(image, list):
            return [encode_single(img) for img in image]
        else:
            return encode_single(image)

    def execute(
        self, prompt, additional_text, seed, llm_settings=None, model=None, custom_instruction="None", enable_web_search=False, list_sources=False, num_search_results=3, keep_context=True, ollama_chat_mode=False, compress=False, compression_level="soft", console_log=True, image=None
        ):
        selected_model_value = models_dict.get(model)
        settings = llm_settings if isinstance(llm_settings, dict) else {}

        if not selected_model_value:
            print(f"Error: Model '{model}' not found in config.json. Using default.")
            selected_model_value = list(models_dict.values())[0] if models_dict else None
            if not selected_model_value:
                return ("Error: No valid model found.",)

        selected_base_url = base_url.get(model)

        if console_log:
            print(
                f"\033[92mPolymath Chat Request: \033[0mModel='{model}', Prompt='{prompt}', Additional Text='{additional_text}'", 
                f"Seed='{seed}', Web Search Enabled={enable_web_search}, Keep Context={keep_context},"
                f"Ollama Chat Mode={ollama_chat_mode}, Custom Instruction='{custom_instruction}'"
            )

        # Process additional_text using the given prompt:
        # If the prompt contains the placeholder "{additional_text}", then replace it with the additional_text input.
        # Otherwise, if additional_text is provided, append it at the end.
        if "{additional_text}" in prompt:
            augmented_prompt = prompt.replace("{additional_text}", additional_text)
        elif additional_text.strip():
            augmented_prompt = prompt + "\n\n" + additional_text
        else:
            augmented_prompt = prompt

        if enable_web_search:
            urls_in_prompt = re.findall(r'(https?://\S+)', augmented_prompt)
            try:
                search_results_content = []
                searched_urls = set()
                for url in urls_in_prompt:
                    if url not in searched_urls:
                        print(f"Fetching content from URL in prompt: {url}")
                        content = fetch_and_extract_content(url)
                        if content:
                            search_results_content.append(
                                f"Source (from prompt): {url}\nContent:\n{content}\n---\n"
                            )
                        else:
                            search_results_content.append(
                                f"Source (from prompt): {url}\nCould not retrieve content.\n---\n"
                            )
                        searched_urls.add(url)
                for i, url in enumerate(search(augmented_prompt, num_results=num_search_results)):
                    if url not in searched_urls:
                        print(f"Fetching content from search result {i+1}: {url}")
                        content = fetch_and_extract_content(url)
                        if content:
                            search_results_content.append(
                                f"Search Result {i+1}:\nSource: {url}\nContent:\n{content}\n---\n"
                            )
                        else:
                            search_results_content.append(
                                f"Search Result {i+1}:\nSource: {url}\nCould not retrieve content.\n---\n"
                            )
                        searched_urls.add(url)
                if search_results_content:
                    augmented_prompt = (
                        f"Original query: {augmented_prompt}\n\nWeb search results and linked content:\n\n"
                        f"{''.join(search_results_content)}\n\nBased on this information, please provide a response."
                    )
                else:
                    augmented_prompt = (
                        f"Original query: {augmented_prompt}\n\nNo relevant web search results or linked content found. "
                        "Please proceed with the original query."
                    )
                    if console_log:
                        print("No relevant web search results or linked content found.")
            except Exception as e:
                print(f"Web Search Error: {e}")
                augmented_prompt = (
                    f"Original query: {augmented_prompt}\n\nAn error occurred during web search or fetching linked content. "
                    "Please proceed with the original query."
                )

        if compress:
            compression_chars = {
                "soft": 250,
                "medium": 150,
                "hard": 75,
            }
            char_limit = compression_chars[compression_level]
            augmented_prompt += (
                f" Compress the output to be concise while retaining key visual details. MAX OUTPUT SIZE no more than "
                f"{char_limit} characters."
            )
            if console_log:
                print(f"\033[92mFull Polymath Prompt\033[0m", augmented_prompt)

        if list_sources:
            augmented_prompt += " Always list all fetched sources."

        if custom_instruction == "node finder":
            with open(json_file_path, "r", encoding="utf-8") as file:
                json_data = json.load(file)
                filtered_data = [
                    f"Title: {node['title']}\nReference: {node['reference']}\nDescription: {node['description']}\n"
                    for node in json_data.get("custom_nodes", [])
                ]
                data = "\n".join(filtered_data)
                print("Filtered JSON Data:")
                print(data)
                augmented_prompt += data

        b64 = None
        if image is not None:
            pil_images = self.tensor_to_pil(image)
            b64 = self.encode_image(pil_images)
            if isinstance(b64, str):
                b64 = [b64]

        response = self.polymath_interaction(
            selected_base_url,
            api_key_oai,
            api_key_anthropic,
            api_key_xai,
            api_key_deepseek,
            api_key_gemini,
            selected_model_value,
            augmented_prompt,
            console_log,
            keep_context,
            ollama_chat_mode,
            custom_instruction,
            seed,
            llm_settings=settings,
            b64=b64
        )
        return response

    def polymath_interaction(self, selected_base_url, api_key_oai, api_key_anthropic, api_key_xai, api_key_deepseek, api_key_gemini, model_value, prompt, console_log, keep_context, ollama_chat_mode, custom_instruction, seed=42, llm_settings={}, b64=None):
        settings = llm_settings
        # OpenAI-based models (e.g., GPT, o1, o3)
        if model_value.startswith(('gpt', 'o1', 'o3')):
            from openai import OpenAI
            client = OpenAI(api_key=api_key_oai, base_url=selected_base_url)
            messages = []
            if custom_instruction != "None":
                try:
                    with open(os.path.join(custom_instructions_directory, f"{custom_instruction}.txt"), 'r') as f:
                        system_instruction = f.read()
                        messages.append({"role": "system", "content": system_instruction})
                except Exception as e:
                    print("Error reading custom instruction file:", e)
            if keep_context and self.chat_history:
                messages.extend(self.chat_history)
            if b64:
                 image_data = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                 user_content = [
                     {"type": "text", "text": prompt},
                     image_data
                 ]
                 messages.append({"role": "user", "content": user_content})
            messages.append({"role": "user", "content": prompt})
            api_params = {
                "model": model_value,
                "messages": messages,
                "stream": False,
                "temperature": settings.get("temperature", 0.8),
                "top_p": settings.get("top_p", 0.95),
                "max_tokens": settings.get("max_tokens", 1024),
                "seed": settings.get("seed", 42),
            }
            stop_seq = settings.get("stop_sequences", "")
            if stop_seq:
                api_params["stop"] = [s.strip() for s in stop_seq.split(',')]
            request_timeout = settings.get("request_timeout", 120) or None
            completion = client.chat.completions.create(
                **api_params,
                timeout=request_timeout
            )
            output_text = completion.choices[0].message.content
            self.chat_history.extend([{"role": "user", "content": prompt},
                                    {"role": "assistant", "content": output_text}])
            return (output_text,)

        elif model_value.startswith('dall'):
            from openai import OpenAI
            from PIL import Image
            from io import BytesIO
            client = OpenAI(api_key=api_key_oai, base_url=selected_base_url)
            dalle_params = {
                "model": model_value,
                "prompt": prompt,
                "n": settings.get("dalle_n", 1),
                "size": settings.get("dalle_size", "1024x1024"),
                "quality": settings.get("dalle_quality", "standard"),
                "style": settings.get("dalle_style", "vivid"),
                "response_format": "b64_json"
                }
            if console_log: print(f"\033[92mDALL-E Params:\033[0m", dalle_params)
            request_timeout = settings.get("request_timeout", 120) or None
            response = client.images.generate(**dalle_params, timeout=request_timeout)
            output_text = response.data[0].revised_prompt
            image_tensors = []
            for i, img_data in enumerate(response.data):
                try:
                    decoded_data = base64.b64decode(img_data.b64_json)
                    image = Image.open(BytesIO(decoded_data)).convert("RGB") # Ensure RGB
                    # Convert PIL to Tensor
                    img_array = np.array(image).astype(np.float32) / 255.0
                    img_tensor = torch.from_numpy(img_array).unsqueeze(0) # Add batch dim
                    image_tensors.append(img_tensor)
                except Exception as e:
                        print(f"Error processing DALL-E image data {i+1}: {e}")
            final_image_output = torch.cat(image_tensors, dim=0) if len(image_tensors) > 1 else (image_tensors[0] if image_tensors else None)
            output_text = output_text if output_text else "(No revised prompt provided)"
            return (output_text, final_image_output)
        
        # Anthropic (e.g., Claude models) branch
        elif model_value.startswith('claude'):
            from anthropic import Anthropic
            client = Anthropic(api_key=api_key_anthropic)
            messages = []
            if custom_instruction != "None":
                try:
                    with open(os.path.join(custom_instructions_directory, f"{custom_instruction}.txt"), 'r') as f:
                        system_instruction = f.read()
                        messages.append({"role": "system", "content": system_instruction})
                except Exception as e:
                    print("Error reading custom instruction file:", e)
            if keep_context and self.chat_history:
                messages.extend(self.chat_history)
            messages.append({"role": "user", "content": prompt})
            completion = client.messages.create(
                model=model_value,
                max_tokens=1024,
                messages=messages
                #seed=seed not yet supported
            )
            output_text = completion.content
            self.chat_history.extend([{"role": "user", "content": prompt},
                                    {"role": "assistant", "content": output_text}])
            return (output_text,)

        # x.ai / Grok branch
        elif model_value.startswith('grok'):
            from openai import OpenAI
            client = OpenAI(api_key=api_key_xai, base_url=selected_base_url)
            messages = []
            if custom_instruction != "None":
                try:
                    with open(os.path.join(custom_instructions_directory, f"{custom_instruction}.txt"), 'r') as f:
                        system_instruction = f.read()
                        messages.append({"role": "system", "content": system_instruction})
                except Exception as e:
                    print("Error reading custom instruction file:", e)
            if keep_context and self.chat_history:
                messages.extend(self.chat_history)
            messages.append({"role": "user", "content": prompt})
            completion = client.chat.completions.create(
                model=model_value,
                messages=messages,
                seed=seed
            )
            output_text = completion.choices[0].message.content
            self.chat_history.extend([{"role": "user", "content": prompt},
                                    {"role": "assistant", "content": output_text}])
            return (output_text,)

        # DeepSeek branch (compatible with OpenAI SDK)
        elif model_value.startswith('deepseek'):
            from openai import OpenAI
            client = OpenAI(api_key=api_key_deepseek, base_url=selected_base_url)
            messages = []
            if custom_instruction != "None":
                try:
                    with open(os.path.join(custom_instructions_directory, f"{custom_instruction}.txt"), 'r') as f:
                        system_instruction = f.read()
                        messages.append({"role": "system", "content": system_instruction})
                except Exception as e:
                    print("Error reading custom instruction file:", e)
            if keep_context and self.chat_history:
                messages.extend(self.chat_history)
            messages.append({"role": "user", "content": prompt})
            completion = client.chat.completions.create(
                model=model_value,
                messages=messages,
                stream=False,
                seed=seed
            )
            output_text = completion.choices[0].message.content
            self.chat_history.extend([{"role": "user", "content": prompt},
                                    {"role": "assistant", "content": output_text}])
            return (output_text,)

        # Gemini branch
        elif model_value.startswith(('gemini-','gemma-')):
            from google import genai
            from google.genai import types
            from PIL import Image
            from io import BytesIO
            import os

            # Initialize the Gemini client (using your Gemini API key)
            client = genai.Client(api_key=api_key_gemini)

            if custom_instruction != "None":
                try:
                    with open(os.path.join(custom_instructions_directory, f"{custom_instruction}.txt"), 'r') as f:
                        custom_instr = f.read()
                except Exception as e:
                    print("Error reading custom instruction file:", e)
                    custom_instr = ""
            else:
                custom_instr = ""
                
            # Combine custom instructions (if any) with the prompt.
            prompt_text = (custom_instr + "\n" + prompt).strip() if custom_instr else prompt

            # Build the contents for the generation call.
            # If a base64 image is provided, decode it into a PIL Image and pass a list [prompt_text, pil_image]
            if b64:
                images = []
                for img_str in b64:
                    image_bytes = base64.b64decode(img_str)
                    pil_image = Image.open(BytesIO(image_bytes))
                    images.append(pil_image)
                contents = [prompt_text] + images
            else:
                contents = prompt_text

            # Call the model with a configuration that asks for both text and image outputs.
            try:
                response = client.models.generate_content(
                    model=model_value,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_modalities=["Text", "Image"],
                        seed=seed,
                    ),
                )
            except Exception as e:
                if "only supports text output" in str(e):
                    if console_log:
                        print("Gemini model does not support image output. Falling back to text-only.")
                    response = client.models.generate_content(
                        model=model_value,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            response_modalities=["Text"],
                            seed=seed,
                        ),
                    )
                else:
                    raise e

            # Parse the response: concatenate all text parts and load any image part.
            output_text = ""
            output_image = None
            img_tensor = None

            for part in response.candidates[0].content.parts:
                if part.text is not None:
                    output_text += part.text
                elif part.inline_data is not None:
                    output_image = Image.open(BytesIO(part.inline_data.data))

            if output_image is not None:
                img_array = np.array(output_image).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array)
                if img_tensor.dim() == 3:
                    img_tensor = img_tensor.unsqueeze(0)

            return (output_text, img_tensor if img_tensor is not None else "")

        # Fallback to Ollama API
        else:
            images = b64 if b64 else None
            ollama_options = {
                "temperature": settings.get("temperature", 0.8),
                "top_p": settings.get("top_p", 0.95),
                "top_k": settings.get("top_k", 40),
                "num_predict": settings.get("max_tokens", -1), # Map max_tokens
                "seed": settings.get("seed", 42),
                "repeat_penalty": settings.get("repeat_penalty", 1.1),
                "repeat_last_n": settings.get("repeat_last_n", 64),
                "keep_alive": settings.get("ollama_keep_alive", 5)
                }
            stop_seq = settings.get("stop_sequences", "")
            if stop_seq:
                ollama_options["stop"] = [s.strip() for s in stop_seq.split(',')]
            response = self.ollama_api.generate_completion(
                ollama_chat_mode=ollama_chat_mode,
                model=model_value,
                prompt=prompt,
                stream=False,
                images=images,
                seed=seed,
                options=ollama_options
            )
            if ollama_chat_mode:
                output_text = response.get('message', {}).get('content', '')
            else:
                output_text = response.get('response', '')
            if console_log:
                print("Ollama API Response:", output_text)
            return (output_text,)


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


class MediaScraper:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "urls": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
                "output_file_path": ("STRING", {"multiline": False, "default": ""}),
                "file_name": ("STRING", {"multiline": False, "default": ""}),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                #"preview_images": ("BOOLEAN", {"default": False}),
                "keep_temp_path": ("BOOLEAN", {"default": False, "tooltip": "Images are saved temporarily, then renamed. Set true to keep the gallery_dl structure."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "filepath_texts")
    FUNCTION = "execute"
    CATEGORY = "Polymath"
    OUTPUT_IS_LIST = (True, True)

    def save_images_custom(self, images, dest_folder, file_prefix, compress_level=4):
        # If no dest_folder is given, use ComfyUI's default
        if not dest_folder:
            dest_folder = folder_paths.get_output_directory()
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        # Attempt to figure out the (height, width) from the first image
        try:
            tensor = images[0]
            if len(tensor.shape) == 4:
                height, width = tensor.shape[1], tensor.shape[2]
            elif len(tensor.shape) == 3:
                height, width = tensor.shape[0], tensor.shape[1]
            else:
                height, width = 100, 100
        except:
            height, width = 100, 100

        # Use folder_paths logic for naming
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            file_prefix or "ComfyUI", dest_folder, width, height
        )
        valid_exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')
        existing_images = []
        for ext in valid_exts:
            pattern = os.path.join(dest_folder, f"{filename}_*{ext}")
            existing_images.extend(glob.glob(pattern))

        counter += len(existing_images)

        saved_paths = []
        for idx, image_tensor in enumerate(images, start=1):
            try:
                if isinstance(image_tensor, torch.Tensor):
                    arr = image_tensor.squeeze(0).cpu().numpy() if image_tensor.ndim == 4 else image_tensor.cpu().numpy()
                    arr = (255. * arr).clip(0, 255).astype("uint8")
                    pil_img = Image.fromarray(arr)
                else:
                    pil_img = image_tensor
            except Exception:
                continue

            file_name = f"{filename}_{counter:05}.png"
            out_path = os.path.join(dest_folder, file_name)
            # If file already exists, append idx to avoid overwriting
            if os.path.exists(out_path):
                file_name = f"{filename}_{counter:05}_{idx:02}.png"
                out_path = os.path.join(dest_folder, file_name)
            pil_img.save(out_path, compress_level=compress_level)
            saved_paths.append(out_path)
            counter += 1

        return saved_paths

    def execute(self, urls, output_file_path, file_name, image_load_cap=0, preview_images=False, keep_temp_path=False):
       
        # Fallback: if output_file_path is empty, use output/scraped_by_polymath
        if not output_file_path:
            output_file_path = os.path.join(folder_paths.get_output_directory(), "scraped_by_polymath")
        
        # 1) Create a temp folder for downloads if both output path and file name are given.    
        if output_file_path and file_name:
            temp_folder = os.path.join(os.path.abspath(output_file_path), "__temp__")
            os.makedirs(temp_folder, exist_ok=True)
            download_dest = temp_folder
        else:
            # If user didn't specify both, download directly to output_file_path (or current dir if empty)
            download_dest = os.path.abspath(output_file_path) if output_file_path else None

        # 2) Download images using gallery-dl
        url_list = [u.strip() for u in urls.split("\n") if u.strip()]
        downloaded_files = []
        for url in url_list:
            if image_load_cap > 0 and len(downloaded_files) >= image_load_cap:
                break
            cmd = ["gallery-dl", url]
            if download_dest:
                cmd.extend(["-d", download_dest])
            if image_load_cap > 0:
                cap = image_load_cap - len(downloaded_files)
                cmd.extend(["--range", f"1-{cap}"])
            subprocess.run(cmd)

        # 3) Gather downloaded files
        valid_exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')
        if download_dest and os.path.isdir(download_dest):
            for root, dirs, files in os.walk(download_dest):
                for f in sorted(files):
                    if f.lower().endswith(valid_exts):
                        downloaded_files.append(os.path.join(root, f))
                        if image_load_cap > 0 and len(downloaded_files) >= image_load_cap:
                            break
                if image_load_cap > 0 and len(downloaded_files) >= image_load_cap:
                    break

        # 4) Convert downloaded images to tensors
        images, filepath_texts = [], []
        for fp in downloaded_files:
            try:
                img = Image.open(fp)
                images.append(pil2tensor(img))
                filepath_texts.append(fp)
            except Exception:
                pass

        # 5) If user specified both output path + file name
        if output_file_path and file_name:
            final_folder = os.path.abspath(output_file_path)  # We'll store final images here
            prefix = os.path.splitext(file_name)[0]

            if keep_temp_path:
                # Simply rename in place, keep the temp folder
                base, ext = os.path.splitext(file_name)
                ext = ext if ext else ".jpg"
                for idx, old_file in enumerate(downloaded_files, start=1):
                    new_file_name = f"{base}_{idx}{ext}"
                    new_file_path = os.path.join(os.path.dirname(old_file), new_file_name)
                    os.rename(old_file, new_file_path)
                    downloaded_files[idx - 1] = new_file_path
                filepath_texts = downloaded_files

            else:
                # Move (save) images into final_folder with file_name as prefix
                # Then remove the temp folder entirely
                filepath_texts = self.save_images_custom(images, final_folder, prefix)
                for old_file in downloaded_files:
                    try:
                        os.remove(old_file)
                    except:
                        pass
                # Remove temp folder if empty (or remove it entirely)
                try:
                    shutil.rmtree(temp_folder)
                except:
                    pass

        else:
            # If not both set, we just keep the downloaded files in place
            filepath_texts = downloaded_files

        return (images, filepath_texts)


NODE_CLASS_MAPPINGS = {
    "polymath_settings": PolymathSettings,
    "polymath_chat": Polymath,
    "polymath_scraper": MediaScraper
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "polymath_settings": "LLM Polymath Chat and API Settings",
    "polymath_chat": "LLM Polymath Chat with Advanced Web and Link Search",
    "polymath_scraper": "LLM Polymath Scraper for various sites"
}
