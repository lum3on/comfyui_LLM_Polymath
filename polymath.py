import subprocess
import sys
import json
import os, glob
import re
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import openai
from googlesearch import search
import requests
from bs4 import BeautifulSoup

# For the Scraper
import torch
import folder_paths
import shutil

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
    print("\033[92mPolymath loaded these models: \033[0m", f"\033[38;5;99m{models_list}\033[0m")

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
        self.chat_history = []

    def generate_completion(self, model, prompt, stream=False, keep_alive=5, ollama_chat_mode=False, images=None):
        url = f"{self.base_url}/api/chat" if ollama_chat_mode else f"{self.base_url}/api/generate"

        if ollama_chat_mode:
            # Append the new user message to the chat history
            self.chat_history.append({"role": "user", "content": prompt})
            payload = {
                "model": model,
                "messages": self.chat_history,  # Include the entire chat history
                "stream": stream,
                "keep_alive": keep_alive
            }
        else:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": stream,
                "keep_alive": keep_alive
            }

        if images:
            payload["images"] = images

        response = requests.post(url, json=payload)
        response.raise_for_status()
        response_json = response.json()

        if ollama_chat_mode:
            # Append the model's response to the chat history
            self.chat_history.append({"role": "assistant", "content": response_json.get('message', {}).get('content', '')})

        return response_json

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
                "seed": ("INT", {"default": 0, "min": 0, "step": 1}),
            },
            "optional": {
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
        enable_web_search=False, list_sources=False, num_search_results=3, keep_context=True, ollama_chat_mode=False, compress=False, compression_level="soft", console_log=True, image=None
    ):
        selected_model_value = models_dict.get(model)
        if not selected_model_value:
            print(f"Error: Model '{model}' not found in config.json. Using default.")
            selected_model_value = list(models_dict.values())[0] if models_dict else None
            if not selected_model_value:
                return ("Error: No valid model found.",)

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
            pil_image = self.tensor_to_pil(image)
            b64 = self.encode_image(pil_image)

        response = self.polymath_interaction(
            base_url, api_key, selected_model_value, augmented_prompt,
            console_log, keep_context, ollama_chat_mode, custom_instruction, b64=b64, 
        )
        return response

    def polymath_interaction(self, base_url, api_key, model_value, prompt, console_log, keep_context, ollama_chat_mode, custom_instruction, b64=None):
        
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

            if console_log:
                print("\033[92mPolymath Chat Response:\033[0m", output_text)
            self.chat_history.append({"role": "user", "content": prompt})
            self.chat_history.append({"role": "assistant", "content": output_text})

            return (output_text,)
        
        else:
            images = None

            if b64:
                images = [b64]

            response = self.ollama_api.generate_completion(
                ollama_chat_mode=ollama_chat_mode,
                model=model_value,
                prompt=prompt,
                stream=False,
                keep_alive=5,
                images=images
            )

            if ollama_chat_mode:
                output_text = response.get('message', {}).get('content', '')
            else:
                output_text = response.get('response', '')

            if console_log:
                print("\033[92mOllama API Response:\033[0m", output_text)
                
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
    "polymath_chat": Polymath,
    "polymath_scraper": MediaScraper
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "polymath_chat": "LLM Polymath Chat with Advanced Web and Link Search",
    "polymath_scraper": "LLM Polymath Scraper for various sites"
}
