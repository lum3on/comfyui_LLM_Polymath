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
                "concept_to_erase": ("STRING", {"default": "fox"}),
                "guided_concept": ("STRING", {"default": ""}),
                "erase_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
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
            token_weight_pairs = tokenizer.tokenize_with_weights(text, return_word_ids=False)
            result = text_encoder.encode_from_tokens_scheduled(token_weight_pairs)
            
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list) and len(result[0]) >= 2:
                embedding = result[0][0]
                embedding = embedding.to(device)
                embedding_mean = embedding.mean(dim=1).squeeze(0).to(model_dtype)
                # Normalize embeddings to prevent extreme updates
                norm = embedding_mean.norm()
                if norm > 1e-6:
                    embedding_mean = embedding_mean / norm
                return embedding_mean
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

        # Get embeddings for the concepts, normalized
        old_embedding = self.get_embedding(concept_to_erase, tokenizer, text_encoder, device, model_dtype)
        new_embedding = self.get_embedding(guided_concept, tokenizer, text_encoder, device, model_dtype)

        old_embeddings = [old_embedding]  # Support multiple concepts if needed

        # Function to compute new weights with residual blending
        def compute_new_weight(weight, old_embeddings, target_embedding, erase_scale):
            in_dim = weight.shape[1]
            mat1 = torch.zeros_like(weight, dtype=torch.float32, device=device)
            mat2 = torch.zeros((in_dim, in_dim), dtype=torch.float32, device=device)
            for old_emb in old_embeddings:
                old_vec = old_emb.to(torch.float32).unsqueeze(1)
                target_vec = target_embedding.to(torch.float32).unsqueeze(1)
                target_proj = torch.matmul(weight.to(torch.float32), target_vec)
                mat1 += erase_scale * torch.matmul(target_proj, old_vec.T)
                mat2 += torch.matmul(old_vec, old_vec.T)
            
            # Increase regularization for stability
            mat2 += 1e-2 * torch.eye(in_dim, device=device, dtype=torch.float32)
            mat2_inv = torch.linalg.pinv(mat2)
            new_update = torch.matmul(mat1, mat2_inv)

            # Blend with original weight to preserve stability
            alpha = min(erase_scale / 10.0, 1.0)  # Controlled blending
            new_weight = (1 - alpha) * weight.to(torch.float32) + alpha * new_update
            print(f"Weight update norm: {(new_weight - weight).norm().item():.4f}")
            return new_weight.to(weight.dtype)

        # Modify cross-attention layers, focusing on output blocks
        for name, module in actual_model.named_modules():
            if "attn2" in name and "output_blocks" in name:  # Focus on later layers
                if hasattr(module, "to_k"):
                    weight = module.to_k.weight
                    new_weight = compute_new_weight(weight, old_embeddings, new_embedding, erase_scale)
                    with torch.no_grad():
                        module.to_k.weight.copy_(new_weight)
                if hasattr(module, "to_v"):
                    weight = module.to_v.weight
                    new_weight = compute_new_weight(weight, old_embeddings, new_embedding, erase_scale)
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