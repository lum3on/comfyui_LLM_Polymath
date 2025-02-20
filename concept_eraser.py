import torch
import torch.nn as nn
from comfy.model_base import BaseModel

# Define the UCE Eraser Node class
class UCEEraserNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),  # UNet model input from ComfyUI
                "clip": ("CLIP",),    # SDXL CLIP model input (with an attached tokenizer)
                "concepts_to_erase": ("STRING", {
                    "multiline": True,
                    "default": "Kelly Mckernan, Sarah Anderson",
                    "description": "Comma-separated list of concepts to erase"
                }),
                "baseline_concept": ("STRING", {
                    "multiline": False,
                    "default": "art",
                    "description": "Baseline concept to align erased concepts to"
                }),
                "lambda_reg": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "description": "Regularization parameter for numerical stability"
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("modified_model",)
    FUNCTION = "apply_uce"
    CATEGORY = "Model Editing"

    def apply_uce(self, model, clip, concepts_to_erase, baseline_concept, lambda_reg):
        print("\033[92mStarting UCE Eraser Node execution...\033[0m")
        print("Cloning model...")
        model_clone = model.clone()
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        model_clone.model.to(device)
        unet = model_clone.model.diffusion_model
        device = next(unet.parameters()).device
        print(f"Model device: {device}")

        # Parse concepts
        concepts_list = [c.strip() for c in concepts_to_erase.split(",") if c.strip()]
        if not concepts_list:
            raise ValueError("No valid concepts to erase provided.")
        print(f"Concepts to erase: {concepts_list}")
        print(f"Baseline concept: {baseline_concept}")

        # Get tokenizer and text encoder
        tokenizer = clip.tokenizer
        text_encoder = clip
        max_length = getattr(tokenizer, "model_max_length", 77)

        def get_text_embedding(text):
            token_weight_pairs = tokenizer.tokenize_with_weights(text, return_word_ids=False)
            result = text_encoder.encode_from_tokens_scheduled(token_weight_pairs)
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list) and len(result[0]) >= 2:
                embedding = result[0][0]  # (1, max_length, 2048)
                return embedding.mean(dim=1).squeeze(0).to(device)  # (2048)
            raise RuntimeError("Unexpected CLIP output: " + str(result))

        # Compute embeddings
        erase_embeddings = torch.stack([get_text_embedding(c) for c in concepts_list])  # (num_concepts, 2048)
        baseline_embedding = get_text_embedding(baseline_concept)  # (2048)
        print(f"Erase embeddings shape: {erase_embeddings.shape}, Baseline shape: {baseline_embedding.shape}")

        # Process cross-attention layers
        for name, module in unet.named_modules():
            if "attn2" in name and hasattr(module, "to_v"):
                print(f"Processing layer: {name}")
                weight = module.to_v.weight
                print(f"Original weight shape: {weight.shape}, norm: {weight.norm():.4f}")
                out_dim, in_dim = weight.shape

                if in_dim != baseline_embedding.shape[0]:
                    print(f"Skipping layer {name}: weight in_dim {in_dim} != embedding dim {baseline_embedding.shape[0]}")
                    continue

                v_star = torch.matmul(weight, baseline_embedding).unsqueeze(-1)  # (out_dim, 1), e.g., (640, 1)
                C = erase_embeddings.T  # (in_dim, num_concepts), e.g., (2048, 1)

                # Special case for single concept
                num_concepts = erase_embeddings.shape[0]
                if num_concepts == 1:
                    # For 1 concept, project erase_embedding out of baseline direction
                    c = erase_embeddings.squeeze(0)  # (2048)
                    dot_product = torch.dot(c, baseline_embedding)
                    projection = (dot_product / baseline_embedding.norm()**2) * baseline_embedding
                    adjusted_embedding = c - projection
                    new_weight = weight - (v_star @ adjusted_embedding.unsqueeze(0)) / (adjusted_embedding.norm()**2 + lambda_reg)
                else:
                    # Multiple concepts: original UCE
                    sum_v_c_T = v_star @ C.T  # (out_dim, in_dim), e.g., (640, 2048)
                    sum_c_c_T = C @ C.T       # (num_concepts, num_concepts)
                    sum_c_c_T += lambda_reg * torch.eye(sum_c_c_T.shape[0], device=device)

                    try:
                        inverse_term = torch.linalg.inv(sum_c_c_T)
                    except RuntimeError:
                        print("Inversion failed, using pseudoinverse.")
                        inverse_term = torch.linalg.pinv(sum_c_c_T)

                    new_weight = sum_v_c_T @ inverse_term @ C  # (out_dim, in_dim)

                print(f"New weight shape: {new_weight.shape}, norm: {new_weight.norm():.4f}")
                with torch.no_grad():
                    if new_weight.shape != weight.shape:
                        print(f"Shape mismatch: {new_weight.shape} vs {weight.shape}. Skipping.")
                        continue
                    module.to_v.weight.copy_(new_weight)
                    print(f"Weight change norm: {(new_weight - weight).norm():.4f}")

        print("UCE Eraser Node execution completed.")
        return (model_clone,)

# Register the node with ComfyUI
NODE_CLASS_MAPPINGS = {"UCEEraserNode": UCEEraserNode}
NODE_DISPLAY_NAME_MAPPINGS = {"UCEEraserNode": "UCE Eraser Node"}

if __name__ == "__main__":
    print("UCE Eraser Node loaded successfully.")
