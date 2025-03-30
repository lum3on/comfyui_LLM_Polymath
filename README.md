# LLM Polymath Chat Node
An advanced Chat Node for ComfyUI that integrates large language models to build text-driven applications and automate data processes (RAGs), enhancing prompt responses by optionally incorporating real-time web search, linked content extraction, and custom agent instructions. It supports both OpenAI’s GPT-like models and alternative models served via a local Ollama API. At its core, two essential instructions—the Comfy Node Finder, which retrieves relevant custom nodes from a the ComfyUi- Manager Custom-node-JSON database based on your queries, and the Smart Assistant, which ingests your workflow JSON to deliver tailored, actionable recommendations—drive its powerful, context-aware functionality. Additionally, a range of other agents such as Flux Prompter, several Custom Instructors, a Python debugger and scripter, and many more further extend its capabilities.

<img width="1263" alt="Bildschirmfoto 2025-02-06 um 11 05 19" src="https://github.com/user-attachments/assets/d4622bfe-c358-4f51-8dc4-cbf3d9880a70" />
---

## Features

### Prompt Processing
- **Placeholder Replacement:** If your prompt contains the placeholder `{additional_text}`, the node replaces it with the provided additional text. Otherwise, any additional text is appended.
- **Dynamic Augmentation:** Depending on the settings, the node can automatically augment your prompt with web-fetched content or search results.

### Web Search Integration
- **URL Extraction:** Scans the input prompt for URLs and uses BeautifulSoup to extract text from the linked pages.
- **Google Search Results:** If enabled, it performs a Google search for your query, retrieves a specified number of results (controlled via `num_search_results`), and appends the extracted content to the prompt.
- **Source Listing:** Optionally appends all fetched sources to the prompt so that the language model’s response can reference them.

### Model & API Integration
- **Model Loading:** Loads model configurations from a local `config.json` file and fetches additional models from an Ollama API endpoint (`http://127.0.0.1:11434/api/tags`).
- **API Selection:**  
  - The model automatically selects the API depending on which model is selected from the list. The polymath currently supports Grok, Gemini, Gemma, Deepseek, and Claude.
  - If Ollama is installed and running, the node uses the Ollama API.
- **Chat History:** Optionally retains context from previous interactions to allow for more natural, continuous conversations.

### Custom Instructions
- **Instruction Files:** The node scans a `custom_instructions` directory for `.txt` files and makes them available as options.  
- **Node Finder Specialization:** If the custom instruction named “node finder” is selected, the node loads and appends information from a JSON file (`custom-node-list.json`) to aid in finding specific nodes.

### Image Handling
- **Image Conversion:** Converts provided image tensors into PIL images and encodes them as base64 strings. These are then included in the payload sent to the language model, enabling multimodal interactions.

### Logging & Debugging
- **Console Logging:** When enabled (`Console_log`), detailed information about the prompt augmentation process and API calls is printed to the console.
- **Seed Control:** Uses a user-provided seed to help manage randomness and reproducibility.

### Output Compression
- **Compression Options:** If compression is enabled, the node appends a directive to the prompt that instructs the model to produce concise output. Three levels are available:
  - **Soft:** Maximum output size ~250 characters.
  - **Medium:** Maximum output size ~150 characters.
  - **Hard:** Maximum output size ~75 characters.

---

## USP Comfy Agents

**ComfyUI Node Assistant**

An advanced Agent that analyzes your specific use case and strictly uses the provided ../ComfyUI-Manager/custom-node-list.json reference to deliver consistent structured, ranked recommendations featuring node names, detailed descriptions, categories, inputs/outputs, and usage notes; it dynamically refines suggestions based on your requirements, ensuring you access both top-performing and underrated nodes categorized as Best Image Processing Nodes, Top Text-to-Image Nodes, Essential Utility Nodes, Best Inpainting Nodes, Advanced Control Nodes, Performance Optimization Nodes, Hidden Gems, Latent Processing Nodes, Mathematical Nodes, Noise Processing Nodes, Randomization Nodes, and Display & Show Nodes for optimal functionality, efficiency, and compatibility.

<img width="1151" alt="image" src="https://github.com/user-attachments/assets/dbf27e20-4eff-454c-9a9a-16045e67bae3" />



**ComfyUI Smart Assistant**

ComfyUI Smart Assistant Instruction: An advanced, context-aware AI integration that ingests your workflow JSON to thoroughly analyze your unique use case and deliver tailored, high-impact recommendations presented as structured, ranked insights—with each recommendation accompanied by names, detailed descriptions, categorical breakdowns, input/output specifications, and usage notes—while dynamically adapting to your evolving requirements through in-depth comparisons, alternative methodologies, and layered workflow enhancements; its robust capabilities extend to executing wildcard searches, deploying comprehensive error-handling strategies, offering real-time monitoring insights, and providing seamless integration guidance, all meticulously organized into key sections such as "Best Workflow Enhancements," "Essential Automation Tools," "Performance Optimization Strategies," "Advanced Customization Tips," "Hidden Gems & Lesser-Known Features," "Troubleshooting & Debugging," "Integration & Compatibility Advice," "Wildcard & Exploratory Searches," "Security & Compliance Measures," and "Real-Time Feedback & Monitoring"—ensuring peak functionality, efficiency, and compatibility while maximizing productivity and driving continuous improvement.

<img width="662" alt="image" src="https://github.com/user-attachments/assets/3230a6cf-a783-4914-ba8f-f580c2f971d0" />


**Polymath Scraper**

An automated web scraper node designed for seamless gallery extraction, allowing users to input a gallery website URL and retrieve image data efficiently. Built on gallery-dl, it supports all websites listed in the official repository. with key keatures such as:

- **URL-Based Extraction:** Simply input a gallery URL to fetch images.  
- **Wide Website Support:** Compatible with all sites supported by gallery-dl.  
- **Output-Ready for Training:** Provides structured outputs:  
  - **List of Image Files:** Downloaded images ready for use.  
  - **List of Filenames:** Organized for captioning and dataset creation.  
- **Modular Integration:** Stack with the LLM Polymath Node for automated captioning, enabling end-to-end dataset preparation.  

Ideal for creating large, labeled datasets for AI model training, reducing manual effort and streamlining workflow efficiency.

![image](https://github.com/user-attachments/assets/e4b0d279-d04c-475d-82ae-500b70e415aa)

### Polymath Settings Node

A versatile configuration node providing essential settings for language models (LLMs) and image generation workflows. Designed for maximum flexibility and control, it allows fine-tuning of generative behavior across multiple engines including text and image generation APIs.

- **Comprehensive LLM Controls:** Fine-tune generative text outputs with key parameters:  
  - **Temperature:** Adjusts randomness in output (0.0–2.0, default 0.8).  
  - **Top-p (Nucleus Sampling):** Controls diversity via probability mass (0.0–1.0, default 0.95).  
  - **Top-k:** Limits to top-k most likely tokens (0–100, default 40).  
  - **Max Output Tokens:** Sets maximum length of output (-1 to 65536, default 1024).  
  - **Response Format JSON:** Toggle structured JSON output (default: False).  
  - **Ollama Keep Alive:** Controls idle timeout for Ollama connections (1–10, default 5).  
  - **Request Timeout:** Timeout for generation requests (0–600 sec, default 120).

- **DALL·E Image Settings:** Customize generation style and quality:  
  - **Quality:** Choose between `"standard"` and `"hd"` (default: standard).  
  - **Style:** Select image tone, either `"vivid"` or `"natural"` (default: vivid).  
  - **Size:** Specify dimensions (1024x1024, 1792x1024, 1024x1792; default: 1024x1024).  
  - **Number of Images:** Set number of outputs per request (1–4, default: 1).


<img width="1411" alt="image" src="https://github.com/user-attachments/assets/3e74f014-0713-4518-89d4-3a827c5bfa7c" />

---

## Usage

### Input Options

The node exposes a range of configurable inputs:
- **Prompt:** Main query text. Supports `{additional_text}` placeholders.
- **Additional Text:** Extra text that supplements or replaces the placeholder in the prompt.
- **Seed:** Integer seed for reproducibility.
- **Model:** Dropdown of available models (merged from `config.json` and Ollama API).
- **Custom Instruction:** Choose from available instruction files.
- **Enable Web Search:** Toggle for fetching web content.
- **List Sources:** Whether to append the fetched sources to the prompt.
- **Number of Search Results:** Determines how many search results to process.
- **Keep Context:** Whether to retain chat history across interactions.
- **Compress & Compression Level:** Enable output compression and choose the level.
- **Console Log:** Toggle detailed logging.
- **Image:** Optionally pass an image tensor for multimodal input.

---

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/lum3on/comfyui_LLM_Polymath.git
   cd comfyui_LLM_Polymath
   ```

2. **Install Dependencies:**
   The node automatically attempts to install missing Python packages (such as `googlesearch`, `requests`, and `bs4`). However, you can also manually install dependencies using:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set the key in your Environment Variables:**
   create a .env file in your comfy root folder and set your api-keys in the file like this:
   ```bash
   OPENAI_API_KEY="your_api_key_here"
   ANTHROPIC_API_KEY="your_anthropic_api_key_here"
   XAI_API_KEY="your_xai_api_key_here"
   DEEPSEEK_API_KEY="your_deepseek_api_key_here"
   GEMINI_API_KEY="your_gemini_api_key_here"
   ```

---

## Ollama Installation & Model Download

OLLAMA (Ollama) enables you to run large language models locally with a few simple commands. Follow these instructions to install OLLAMA and download models.

### Installing OLLAMA

#### On macOS:
Download the installer from the official website or install via Homebrew:

```bash
brew install ollama
```

#### On Linux:
Run the installation script directly from your terminal:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### On Windows:
Visit the [Ollama Download Page](https://ollama.com/download) and run the provided installer.

### Downloading Models

Once OLLAMA is installed, you can easily pull and run models. For example, to download the lightweight Gemma 2B model:

```bash
ollama pull gemma:2b
```

After downloading, you can start interacting with the model using:

```bash
ollama run gemma:2b
```

For a full list of available models (including various sizes and specialized variants), please visit the official [Ollama Model Library](https://ollama.com/library).

### Model Availability in Comfy

After you download a model via Ollama, it will automatically be listed in the model dropdown in Comfy after you restart it. This seamless integration means you don’t need to perform any additional configuration—the model is ready for use immediately within Comfy.

### Example Workflow

1. **Install OLLAMA** on your system using the method appropriate for your operating system.
2. **Download a Model** with the `ollama pull` command or use the run command and the model gets auto downloaded.
3. **Run the Model** with `ollama run <model-name>` to start a REPL and interact with it.
4. **Keep the Cli open** so that Comfy can acess the local Olama api
5. **Restart Comfy** to have the downloaded model automatically appear in the model dropdown for easy selection.

By following these steps, you can quickly set up OLLAMA on your machine and begin experimenting with different large language models locally.

For further details on model customization and advanced usage, refer to the official documentation at [Ollama Docs](https://docs.ollama.com).

---

## Planned Updates

The following features are planned for the next Update.

- [ ] **Node Finder Implementation in ComfyUI Manager:** Integrate a full-featured node finder in the Comfy Manager
- [X] **Advanced Parameter Node:** Introduce an enhanced parameter node offering additional customization and control.
- [ ] **Speed Improvements:** Optimize processing and API response times for a more fluid user experience.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

This node integrates several libraries and APIs to deliver an advanced multimodal, web-augmented chat experience. Special thanks to all contributors and open source projects that made this work possible.

---

*For any questions or further assistance, please open an issue on GitHub or contact the maintainer.*
