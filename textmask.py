import torch
import numpy as np
import cv2
import os

# Language mapping: Full Name -> Code
# Keep this dictionary as defined previously
language_map = {
    "English": "en",
    "简体中文": "ch_sim",
    "繁體中文": "ch_tra",
    "العربية": "ar",
    "Azərbaycan": "az",
    "Euskal": "eu",
    "Bosanski": "bs",
    "Български": "bg",
    "Català": "ca",
    "Hrvatski": "hr",
    "Čeština": "cs",
    "Dansk": "da",
    "Nederlands": "nl",
    "Eesti": "et",
    "Suomi": "fi",
    "Français": "fr",
    "Galego": "gl",
    "Deutsch": "de",
    "Ελληνικά": "el",
    "עברית": "he",
    "हिन्दी": "hi",
    "Magyar": "hu",
    "Íslenska": "is",
    "Indonesia": "id",
    "Italiano": "it",
    "日本語": "ja",
    "한국어": "ko",
    "Latviešu": "lv",
    "Lietuvių": "lt",
    "Македонски": "mk",
    "Norsk": "no",
    "Polski": "pl",
    "Português": "pt",
    "Română": "ro",
    "Русский": "ru",
    "Српски": "sr",
    "Slovenčina": "sk",
    "Slovenščina": "sl",
    "Español": "es",
    "Svenska": "sv",
    "ไทย": "th",
    "Türkçe": "tr",
    "Українська": "uk",
    "Tiếng Việt": "vi",
}

# Function to get just the full names for the dropdown
def get_language_names():
    return list(language_map.keys())

# Check for EasyOCR and attempt import
# ... (rest of the imports and helper functions: EASYOCR_AVAILABLE, tensor2np, np_mask_to_tensor, EASYOCR_READERS, get_easyocr_reader remain the same) ...
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    print("Warning: EasyOCR not installed. OCR Refined Text Mask node will not be available.")
    print("Please install it: pip install easyocr")
    EASYOCR_AVAILABLE = False


# Tensor to NumPy (Batch)
def tensor2np(tensor):
    np_array = tensor.cpu().numpy()
    if np_array.ndim == 4 and np_array.shape[0] == 1:
        img = np.clip(np_array[0] * 255, 0, 255).astype(np.uint8)
        return img
    elif np_array.ndim == 3: # HWC without batch
        img = np.clip(np_array * 255, 0, 255).astype(np.uint8)
        return img
    else: # Handle batch > 1 - process first image
         print("Warning: Batch size > 1, processing only the first image.")
         img = np.clip(np_array[0] * 255, 0, 255).astype(np.uint8)
         return img

# NumPy Mask to Tensor Mask
def np_mask_to_tensor(np_mask):
    if np_mask.ndim == 3 and np_mask.shape[2] == 1:
        np_mask = np_mask.squeeze(axis=2)
    elif np_mask.ndim == 3 and np_mask.shape[2] > 1:
         print("Warning: NumPy mask has multiple channels, taking first channel.")
         np_mask = np_mask[:,:,0] # Take first channel

    tensor_mask = torch.from_numpy(np_mask.astype(np.float32) / 255.0)
    tensor_mask = tensor_mask.unsqueeze(0) # Add batch dim -> (1, H, W)
    return tensor_mask

# --- Global EasyOCR Reader ---
EASYOCR_READERS = {}

def get_easyocr_reader(lang_codes=['en'], gpu=True): # Expecting a list of codes now
    """Gets or initializes an EasyOCR reader for the given language codes."""
    lang_key = "_".join(sorted(lang_codes)) + ("_gpu" if gpu else "_cpu")
    if lang_key not in EASYOCR_READERS:
        print(f"Initializing EasyOCR Reader for languages: {lang_codes} (GPU: {gpu})")
        try:
            EASYOCR_READERS[lang_key] = easyocr.Reader(lang_codes, gpu=gpu)
            print("EasyOCR Reader initialized.")
        except Exception as e:
            print(f"Error initializing EasyOCR Reader: {e}")
            if gpu:
                print("Attempting EasyOCR Reader initialization on CPU...")
                try:
                    EASYOCR_READERS[lang_key] = easyocr.Reader(lang_codes, gpu=False)
                    print("EasyOCR Reader initialized on CPU.")
                except Exception as e_cpu:
                    print(f"Error initializing EasyOCR Reader on CPU: {e_cpu}")
                    return None
            else:
                 return None
    return EASYOCR_READERS[lang_key]

# --- The Node ---

class TextMaskGeneratorNode:
    """
    Generates a text mask using OCR to locate text and adaptive thresholding
    to define the mask shape within those locations.
    """
    @classmethod
    def INPUT_TYPES(s):
        if not EASYOCR_AVAILABLE:
             return {"required": {}}

        return {
            "required": {
                "image": ("IMAGE",),
                # --- OCR Parameters ---
                # Use the full names for the dropdown display
                "language_name": (
                    get_language_names(), # Function returns list of full names
                    {"default": "English"} # Default selected item in dropdown
                ),
                "ocr_confidence_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "use_gpu": ("BOOLEAN", {"default": True}),
                # --- Thresholding Parameters ---
                "threshold_block_size": ("INT", {"default": 11, "min": 3, "max": 51, "step": 2}), # Must be odd
                "threshold_c": ("INT", {"default": 7, "min": 1, "max": 20, "step": 1}),
                # --- Optional Post-processing ---
                "dilation_iterations": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}), # Optional dilation *after* combining
                "dilation_kernel_size": ("INT", {"default": 3, "min": 1, "max": 11, "step": 2}), # Usually odd
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "generate_refined_mask"
    CATEGORY = "Masquerade/Masking" # Or your category like "polymath"

    # Update function signature to receive language_name
    def generate_refined_mask(self, image, language_name, ocr_confidence_threshold, use_gpu,
                              threshold_block_size, threshold_c,
                              dilation_iterations, dilation_kernel_size):

        if not EASYOCR_AVAILABLE:
            raise ImportError("EasyOCR library is not available.")

        # 1. Prepare Image (NumPy, Grayscale)
        # ... (image preparation code remains the same) ...
        img_np = tensor2np(image)
        if img_np is None or img_np.size == 0:
            raise ValueError("Input image is invalid.")
        if img_np.ndim == 3 and img_np.shape[2] == 3:
            gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        elif img_np.ndim == 3 and img_np.shape[2] == 4:
            gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGBA2GRAY)
        elif img_np.ndim == 2:
            gray_img = img_np
        else:
            raise ValueError("Unsupported image format for grayscale conversion.")
        h, w = gray_img.shape


        # --- Step A: Generate Pixel Shape Mask (Thresholding) ---
        # ... (thresholding code remains the same) ...
        if threshold_block_size % 2 == 0:
            threshold_block_size += 1
        shape_mask = cv2.adaptiveThreshold(
            gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, threshold_block_size, threshold_c
        )
        print("Generated initial shape mask using adaptive thresholding.")


        # --- Step B: Generate OCR Location Mask ---
        # Translate the selected language name (e.g., "English") to its code (e.g., "en")
        selected_lang_code = language_map.get(language_name, 'en') # Default to 'en' if name not found
        print(f"Selected language: {language_name} -> Code: {selected_lang_code}")

        # Prepare the list of codes for EasyOCR (currently just one)
        lang_codes_for_reader = [selected_lang_code]

        # Get the EasyOCR reader
        gpu_available = torch.cuda.is_available()
        actual_use_gpu = use_gpu and gpu_available
        reader = get_easyocr_reader(lang_codes_for_reader, actual_use_gpu) # Pass the list of codes
        if reader is None:
             raise RuntimeError("Failed to initialize EasyOCR Reader.")

        print(f"Running EasyOCR detection ({', '.join(lang_codes_for_reader)})...")
        results = reader.readtext(img_np, detail=1)
        print(f"EasyOCR found {len(results)} potential text regions.")

        # Create the OCR location mask (code remains the same)
        ocr_location_mask = np.zeros_like(gray_img, dtype=np.uint8)
        # ... (rest of the OCR mask creation loop remains the same) ...
        boxes_drawn = 0
        if results:
            for (bbox, text, conf) in results:
                if conf >= ocr_confidence_threshold:
                    points = np.array(bbox, dtype=np.int32)
                    points[:, 0] = np.clip(points[:, 0], 0, w - 1)
                    points[:, 1] = np.clip(points[:, 1], 0, h - 1)
                    cv2.fillPoly(ocr_location_mask, [points], (255))
                    boxes_drawn += 1
        print(f"Created OCR location mask with {boxes_drawn} boxes (confidence >= {ocr_confidence_threshold}).")


        # --- Step C: Combine Masks (Intersection) ---
        # ... (combining code remains the same) ...
        refined_mask = cv2.bitwise_and(shape_mask, ocr_location_mask)
        print("Combined shape mask and OCR location mask using bitwise AND.")


        # --- Step D: Optional Dilation (Post-processing) ---
        # ... (dilation code remains the same) ...
        if dilation_iterations > 0:
            kernel_s = max(1, dilation_kernel_size)
            if kernel_s % 2 == 0: kernel_s +=1
            kernel = np.ones((kernel_s, kernel_s), np.uint8)
            refined_mask = cv2.dilate(refined_mask, kernel, iterations=dilation_iterations)
            print(f"Applied dilation ({dilation_iterations} iterations, kernel {kernel_s}x{kernel_s}).")


        # --- Step E: Convert to Output Tensor ---
        output_mask = np_mask_to_tensor(refined_mask)

        return (output_mask,)

# --- ComfyUI Registration ---
if EASYOCR_AVAILABLE:
    NODE_CLASS_MAPPINGS = {
        "polymath_text_mask": TextMaskNode # Mapping node class name to class
    }

    # Optional: A dictionary that allows renaming nodes in the UI
    NODE_DISPLAY_NAME_MAPPINGS = {
        "polymath_text_mask": "Generate a mask from text" # Mapping node class name to display name
    }
else:
     NODE_CLASS_MAPPINGS = {}
     NODE_DISPLAY_NAME_MAPPINGS = {}
