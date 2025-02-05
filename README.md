# LLM Polymath Chat Node
An advanced Chat Node designed for ComfyUI that integrates with large language models and enhances prompt responses by optionally incorporating real-time web search, linked content extraction, and custom instructions. It supports both OpenAI’s GPT-like models and alternative models served via a local Ollama API. At its core, two essential instructions—the Comfy Node Finder, which retrieves relevant custom nodes from a curated JSON database based on your queries, and the Smart Assistant, which ingests your workflow JSON to deliver tailored, actionable recommendations—drive its powerful, context-aware functionality. Additionally, a range of other agents such as Flux Prompter, several Custom Instructors, a Python debugger and scripter, and many more further extend its capabilities.

<img width="1446" alt="image" src="https://github.com/user-attachments/assets/4740493e-ab05-4a6b-b3c4-753351d38318" />

---

## Overview

The LLM Polymath Chat Node is built to serve as a “polymath” in the digital world, able to fetch additional context from the web and merge that data with a user’s query. Whether you’re working with conversational prompts, visual inputs, or integrating specialized instructions, this node streamlines interactions with LLM`s while maintaining context and flexibility.

Key capabilities include:

- **Prompt Augmentation:** Inserts additional text into your prompt dynamically.
- **Web Integration:** Searches for URLs in your prompt and fetches content from them. It can also perform a Google search (using the `googlesearch` library) to retrieve supplementary results.
- **Model Selection:** Merges model configurations from a local `config.json` file and from an Ollama API endpoint, providing a list of available models.
- **Custom Instructions:** Supports loading extra directives from a `custom_instructions` folder. When using the “node finder” instruction, it can even load data from a custom node list.
- **Image Support:** Accepts image tensors, converts them to a PIL image, and encodes them in base64 for inclusion in the prompt.
- **Context Preservation:** Maintains a chat history if enabled so that interactions remain coherent over multiple exchanges.
- **Output Compression:** Optionally instructs the language model to generate more concise responses by enforcing character limits.

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
  - If the selected model’s name starts with `gpt`, `o1`, or `o3`, the node uses the OpenAI API (configured via an API key and base URL).
  - For other model identifiers, the node falls back to using the Ollama API.
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

## Planned Updates

The following features are planned for the next Update.

- [ ] **Node Finder Implementation in ComfyUI Manager:** Integrate a full-featured node finder in the Comfy Manager
- [ ] **Advanced Parameter Node:** Introduce an enhanced parameter node offering additional customization and control.
- [ ] **Speed Improvements:** Optimize processing and API response times for a more fluid user experience.

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

3. **Set Up Environment Variables:**
   Make sure your OpenAI API key is available in your environment:
   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```
---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

This node integrates several libraries and APIs to deliver an advanced multimodal, web-augmented chat experience. Special thanks to all contributors and open source projects that made this work possible.

---

*For any questions or further assistance, please open an issue on GitHub or contact the maintainer.*

---

### References

- The core logic and functionalities are defined in the [`polymath.py`](https://github.com/lum3on/comfyui_LLM_Polymath/blob/main/polymath.py) file. cite60
- Model configuration and merging from `config.json` and the Ollama API. cite60

This README should help users and developers understand how to install, configure, and effectively use the LLM Polymath Chat Node within ComfyUI.
