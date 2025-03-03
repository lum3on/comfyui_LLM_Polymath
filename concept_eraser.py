import torch
from torch import nn
import numpy as np

class ConceptEraserNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet": ("MODEL",),
                "clip": ("CLIP",),
                "concept_to_erase": ("STRING", {"default": "Van Gogh"}),
                "guided_concept": ("STRING", {"default": ""}),
                "erase_scale": ("FLOAT", {"default": 100.0, "min": 1.0, "max": 1000.0, "step": 1.0}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "erase_concept"
    CATEGORY = "model_patches"

    def get_embedding(self, text, tokenizer, text_encoder, device, model_dtype):
        """Generate embeddings from text using the tokenizer and text encoder."""
        if not isinstance(text, str):
            raise ValueError(f"Expected text to be a string, got {type(text)}: {text}")
        
        with torch.no_grad():
            # Tokenize the text directly as a string
            token_weight_pairs = tokenizer.tokenize_with_weights(text, return_word_ids=False)
            # Encode the tokens using the text encoder
            result = text_encoder.encode_from_tokens_scheduled(token_weight_pairs)
            
            # Extract the embedding from the result
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list) and len(result[0]) >= 2:
                embedding = result[0][0]  # Adjust based on actual output structure
                embedding = embedding.to(device)
                embedding_mean = embedding.mean(dim=1).squeeze(0).to(model_dtype)
                return embedding_mean
            else:
                raise RuntimeError(f"Unexpected CLIP output format: {result}")

    def erase_concept(self, erase_scale, unet, clip, concept_to_erase, guided_concept):
        # Clone the UNet to avoid modifying the original
        unet_clone = unet.clone()
        actual_model = unet_clone.model

        # Get device and dtype from the model
        device = next(actual_model.parameters()).device
        model_dtype = next(actual_model.parameters()).dtype
        actual_model.to(device)

        # Access tokenizer and text encoder from clip
        tokenizer = clip.tokenizer
        text_encoder = clip  # Assuming clip acts as the text encoder

        # Debugging output
        print(f"Type of concept_to_erase: {type(concept_to_erase)}")
        print(f"concept_to_erase: {concept_to_erase}")

        # Get embeddings for the concepts
        old_embedding = self.get_embedding(concept_to_erase, tokenizer, text_encoder, device, model_dtype)
        new_embedding = self.get_embedding(guided_concept, tokenizer, text_encoder, device, model_dtype)

        old_embeddings = [old_embedding]  # Support multiple concepts if needed

        # Function to compute new weights
        def compute_new_weight(weight, old_embeddings, target_embedding):
            in_dim = weight.shape[1]
            mat1 = torch.zeros_like(weight, dtype=torch.float32, device=device)
            mat2 = torch.zeros((in_dim, in_dim), dtype=torch.float32, device=device)
            for old_emb in old_embeddings:
                old_vec = old_emb.to(torch.float32).unsqueeze(1)
                target_vec = target_embedding.to(torch.float32).unsqueeze(1)
                target_proj = torch.matmul(weight.to(torch.float32), target_vec)
                mat1 += erase_scale * torch.matmul(target_proj, old_vec.T)
                mat2 += torch.matmul(old_vec, old_vec.T)
            mat2 += 1e-4 * torch.eye(in_dim, device=device, dtype=torch.float32)  # Regularization
            mat2_inv = torch.linalg.pinv(mat2)
            new_weight = torch.matmul(mat1, mat2_inv)
            return new_weight.to(weight.dtype)

        # Modify cross-attention layers
        for name, module in actual_model.named_modules():
            if "attn2" in name:
                if hasattr(module, "to_k"):
                    weight = module.to_k.weight
                    new_weight = compute_new_weight(weight, old_embeddings, new_embedding)
                    with torch.no_grad():
                        module.to_k.weight.copy_(new_weight)
                if hasattr(module, "to_v"):
                    weight = module.to_v.weight
                    new_weight = compute_new_weight(weight, old_embeddings, new_embedding)
                    with torch.no_grad():
                        module.to_v.weight.copy_(new_weight)

        return (unet_clone,)

# Register the nodes with ComfyUI
NODE_CLASS_MAPPINGS = {
    "ConceptEraserNode": ConceptEraserNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConceptEraserNode": "Advanced Concept Eraser",
}

if __name__ == "__main__":
    print("Concept Eraser Nodes loaded successfully.")