import subprocess
import sys
import json
import os
import re
import random
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from jax import config
import openai
from googlesearch import search
import requests
from bs4 import BeautifulSoup

api_key = os.getenv("OPENAI_API_KEY")

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
    # Initialize dictionaries
    config_models = {}
    ollama_models = {}

    # Load models from config.json
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            config_models = config_data.get('models', {})
            base_url = config_data.get('baseurl')
    except FileNotFoundError:
        print(f"Error: config.json not found at {config_path}")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {config_path}")

    # Fetch models from Ollama API
    try:
        response = requests.get(ollama_url)
        response.raise_for_status()
        models = response.json().get('models', [])
        ollama_models = {model['name']: model['name'] for model in models}
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Ollama models: {e}")

    # Merge dictionaries
    merged_models = {**config_models, **ollama_models}
     
    # Create a list of model names
    models_list = list(merged_models.keys())

    # Determine the default model
    default_model = models_list[0] if models_list else None
    print("\033[92mAvailable models for Polymath: \033[0m", f"\033[94m{models_list}\033[0m")

    return base_url, merged_models, models_list, default_model

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

    def generate_completion(self, model, prompt, stream=False, keep_alive=5, images=None):
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "keep_alive": keep_alive,
        }
        if images:
            payload["images"] = images

        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

class Polymath:
    chat_history = []
    default_model = None

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
                "seed": ("INT", {"default": 0, "min": 0, "step": 1}),
            },
            "optional": {
                "model": (models_list, {"default": default_model}),
                "custom_instruction": (custom_instructions_list, {"default": "None"}),
                "enable_web_search": ("BOOLEAN", {"default": False}),
                "list_sources": ("BOOLEAN", {"default": True}),
                "num_search_results": ("INT", {"default": 5, "min": 1, "max": 10}),
                "keep_context": ("BOOLEAN", {"default": True}),
                "compress": ("BOOLEAN", {"default": False}),
                "compression_level": (["soft", "medium", "hard"],),
                "Console_log": ("BOOLEAN", {"default": True}),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "Polymath"

    def tensor_to_pil(self, img_tensor):
        arr = img_tensor.cpu().numpy() if hasattr(img_tensor, "cpu") else np.array(img_tensor)
        arr = np.squeeze(arr)  # remove singleton dimensions
        if arr.ndim == 3 and arr.shape[0] == 3:  # if in (C, H, W), transpose to (H, W, C)
            arr = np.transpose(arr, (1, 2, 0))
        arr = np.clip(255.0 * arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def encode_image(self, image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def execute(
        self, prompt, additional_text, seed, model=None, custom_instruction="None",
        enable_web_search=False, list_sources=True, num_search_results=3, keep_context=True,
        compress=False, compression_level="soft", Console_log=True, image=None
    ):
        selected_model_value = models_dict.get(model)
        if not selected_model_value:
            print(f"Error: Model '{model}' not found in config.json. Using default.")
            selected_model_value = list(models_dict.values())[0] if models_dict else None
            if not selected_model_value:
                return ("Error: No valid model found.",)

        if Console_log:
            print(
                f"Polymath Chat Request: Model='{model}', Prompt='{prompt}', Additional Text='{additional_text}', "
                f"Seed='{seed}', Web Search Enabled={enable_web_search}, Keep Context={keep_context}, "
                f"Custom Instruction='{custom_instruction}'"
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
                    if Console_log:
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
            if Console_log:
                print(augmented_prompt)
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
            pil_image = self.tensor_to_pil(image)
            b64 = self.encode_image(pil_image)

        response = self.polymath_interaction(
            base_url, api_key, selected_model_value, augmented_prompt,
            Console_log, keep_context, custom_instruction, b64=b64, 
        )
        return response

    def polymath_interaction(self, base_url, api_key, model_value, prompt, Console_log, keep_context, custom_instruction, b64=None):
        
        if model_value.startswith(('gpt', 'o1', 'o3')):        
            client = openai.OpenAI(api_key=api_key, base_url=base_url)
            messages = []
            if custom_instruction != "None":
                instruction_path = os.path.join(custom_instructions_directory, f"{custom_instruction}.txt")
                try:
                    with open(instruction_path, 'r') as f:
                        system_instruction = f.read()
                        messages.append({"role": "system", "content": system_instruction})
                except FileNotFoundError:
                    print(f"Error: Custom instruction file '{instruction_path}' not found.")
                except Exception as e:
                    print(f"Error reading custom instruction file: {e}")

            if keep_context and self.chat_history:
                messages.extend(self.chat_history)
            if b64:
                image_data = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                user_content = [
                    {"type": "text", "text": prompt},
                    image_data
                ]
                messages.append({"role": "user", "content": user_content})
            else:
                messages.append({"role": "user", "content": prompt})

            completion = client.chat.completions.create(
                model=model_value,
                messages=messages,
                stream=False
            )
            output_text = completion.choices[0].message.content
            if Console_log:
                print(f"Polymath Chat Response: {output_text}")
            self.chat_history.append({"role": "user", "content": prompt})
            self.chat_history.append({"role": "assistant", "content": output_text})
            return (output_text,)
        
        else:
            
            ollama_api = OllamaAPI()
            images = None

            if b64:
                images = [b64]

            response = ollama_api.generate_completion(
                model=model_value,
                prompt=prompt,
                stream=False,
                keep_alive=5,
                images=images
            )
            output_text = response.get('response', '')
            if Console_log:
                print(f"Ollama API Response: {output_text}")
            return (output_text,)


NODE_CLASS_MAPPINGS = {
    "polymath_chat": Polymath
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "polymath_chat": "LLM Polymath Chat with Advanced Web and Link Search"
}
