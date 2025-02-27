import torch
import torch.nn as nn
from comfy.model_base import BaseModel

# Define the Advanced Concept Eraser Node
class ConceptEraserNode:
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
                "guided_concept": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "description": "Concept to guide erased concepts towards (empty for unconditioned)"
                }),
                "preserve_concepts": ("STRING", {
                    "multiline": True,
                    "default": "art, painting",
                    "description": "Comma-separated list of concepts to preserve"
                }),
                "erase_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "description": "Scale factor for concept erasure (higher = stronger erasure)"
                }),
                "preserve_scale": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "description": "Scale factor for concept preservation (higher = stronger preservation)"
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("modified_model",)
    FUNCTION = "apply_concept_erasure"
    CATEGORY = "Model Editing"

    def apply_concept_erasure(self, model, clip, concepts_to_erase, guided_concept, preserve_concepts, erase_scale, preserve_scale):
        print("\033[92mStarting Advanced Concept Eraser execution...\033[0m")
        print("Cloning model...")
        model_clone = model.clone()
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        model_clone.model.to(device)
        unet = model_clone.model.diffusion_model
        device = next(unet.parameters()).device
        model_dtype = next(unet.parameters()).dtype  # Get model's dtype (e.g., torch.float16)
        print(f"Model device: {device}, dtype: {model_dtype}")
        
        # Parse concepts
        concepts_list = [c.strip() for c in concepts_to_erase.split(",") if c.strip()]
        if not concepts_list:
            raise ValueError("No valid concepts to erase provided.")
        print(f"Concepts to erase: {concepts_list}")
        
        # Parse guided concept (target to transform erased concepts into)
        guided_concept = guided_concept.strip()
        if not guided_concept:
            guided_concept = " "  # Use empty string as unconditioned
        print(f"Guided concept: '{guided_concept}'")
        
        # Parse preserve concepts
        preserve_list = [c.strip() for c in preserve_concepts.split(",") if c.strip()]
        print(f"Concepts to preserve: {preserve_list}")
        
        # Get tokenizer and text encoder
        tokenizer = clip.tokenizer
        text_encoder = clip
        
        def get_text_embedding(text):
            """Get text embedding from CLIP text encoder with explicit handling for MPS compatibility"""
            token_weight_pairs = tokenizer.tokenize_with_weights(text, return_word_ids=False)
            result = text_encoder.encode_from_tokens_scheduled(token_weight_pairs)
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list) and len(result[0]) >= 2:
                embedding = result[0][0]  # (1, max_length, 2048)
                # Handle dimensions carefully for MPS compatibility
                embedding = embedding.to(device)
                # Average across sequence length
                embedding_mean = embedding.mean(dim=1)  # (1, 2048)
                embedding_flat = embedding_mean.squeeze(0).to(dtype=model_dtype)  # (2048)
                return embedding_flat  # (2048), cast to model dtype
            raise RuntimeError("Unexpected CLIP output: " + str(result))
        
        # Pre-compute all needed embeddings
        print("Computing text embeddings...")
        old_embeddings = []  # Embeddings of concepts to erase
        for concept in concepts_list:
            old_embeddings.append(get_text_embedding(concept))
        
        # Get guided concept embedding (where to guide erased concepts)
        new_embedding = get_text_embedding(guided_concept)
        
        # Get preserve concept embeddings
        retain_embeddings = []
        for concept in preserve_list:
            retain_embeddings.append(get_text_embedding(concept))
        
        # Process all cross-attention projection matrices
        projection_matrices = []
        
        # First identify all projection matrices in the model
        for name, module in unet.named_modules():
            if "attn2" in name and hasattr(module, "to_v"):
                projection_matrices.append((name, module.to_v))
                
        print(f"Found {len(projection_matrices)} cross-attention projection matrices")
        
        # Apply concept editing to each projection matrix
        for layer_idx, (name, layer) in enumerate(projection_matrices):
            weight = layer.weight
            out_dim, in_dim = weight.shape
            print(f"Processing layer {name}, shape: {weight.shape}")
            
            # Initialize accumulation matrices for computing the projection updates
            mat1 = torch.zeros_like(weight).to(torch.float32)  # Accumulator for value projections
            mat2 = torch.zeros((in_dim, in_dim), device=device, dtype=torch.float32)  # Gram matrix of concepts
            
            # Process concepts to erase
            for i, old_emb in enumerate(old_embeddings):
                # Convert embeddings to float32 for numerical stability during calculations
                old_emb = old_emb.to(torch.float32)
                
                # Get the target embedding (guided concept)
                target_emb = new_embedding.to(torch.float32)
                
                # Handle vector reshaping more carefully for MPS compatibility
                # Explicit size to avoid dimension inference issues on MPS
                old_vec = old_emb.unsqueeze(1)  # Shape: (in_dim, 1)
                old_vec_T = old_emb.unsqueeze(0)  # Shape: (1, in_dim)
                target_vec = target_emb.unsqueeze(1)  # Shape: (in_dim, 1)
                
                # Project concepts through current weight with explicit operations
                old_proj = torch.matmul(weight.to(torch.float32), old_vec)  # (out_dim, 1)
                target_proj = torch.matmul(weight.to(torch.float32), target_vec)  # (out_dim, 1)
                
                # Use batch matrix multiplication with careful explicit reshaping
                mat1_update = torch.matmul(target_proj, old_vec_T)  # (out_dim, in_dim)
                mat2_update = torch.matmul(old_vec, old_vec_T)  # (in_dim, in_dim)
                
                # Add to accumulation matrices with erase scale
                mat1 += erase_scale * mat1_update  # (out_dim, in_dim)
                mat2 += erase_scale * mat2_update  # (in_dim, in_dim)
                
                print(f"Processed erase concept {i+1}/{len(old_embeddings)}")
            
            # Process concepts to preserve
            if retain_embeddings:
                for i, retain_emb in enumerate(retain_embeddings):
                    retain_emb = retain_emb.to(torch.float32)
                    
                    # Handle vector reshaping more carefully for MPS compatibility
                    retain_vec = retain_emb.unsqueeze(1)  # Shape: (in_dim, 1)
                    retain_vec_T = retain_emb.unsqueeze(0)  # Shape: (1, in_dim)
                    
                    # Project preserve concept through current weight
                    retain_proj = torch.matmul(weight.to(torch.float32), retain_vec)  # (out_dim, 1)
                    
                    # Use batch matrix multiplication with careful explicit reshaping
                    mat1_update = torch.matmul(retain_proj, retain_vec_T)  # (out_dim, in_dim)
                    mat2_update = torch.matmul(retain_vec, retain_vec_T)  # (in_dim, in_dim)
                    
                    # Add to accumulation matrices with preserve scale
                    mat1 += preserve_scale * mat1_update  # (out_dim, in_dim)
                    mat2 += preserve_scale * mat2_update  # (in_dim, in_dim)
                    
                    print(f"Processed preserve concept {i+1}/{len(retain_embeddings)}")
            
            # Add small regularization for numerical stability
            mat2 += 1e-4 * torch.eye(in_dim, device=device, dtype=torch.float32)
            
            # Move tensors to CPU for more reliable matrix inversion
            # MPS can have issues with certain linear algebra operations
            try:
                # Move to CPU for inversion to avoid MPS issues
                mat2_cpu = mat2.to("cpu")
                mat2_inv = torch.linalg.inv(mat2_cpu)
                mat2_inv = mat2_inv.to(device)  # Move back to original device
                print("Matrix inversion successful")
            except RuntimeError as e:
                print(f"Matrix inversion failed: {e}, using pseudoinverse")
                # Use CPU for pseudoinverse as well
                mat2_cpu = mat2.to("cpu")
                mat2_inv = torch.linalg.pinv(mat2_cpu)
                mat2_inv = mat2_inv.to(device)  # Move back to original device
            
            # Calculate the new projection matrix weights
            new_weight = torch.matmul(mat1, mat2_inv)
            
            # Ensure correct dtype before updating
            new_weight = new_weight.to(dtype=weight.dtype)
            
            print(f"New weight norm: {new_weight.norm().item():.4f}, Original weight norm: {weight.norm().item():.4f}")
            print(f"Weight change norm: {(new_weight - weight).norm().item():.4f}")
            
            # Update the weights
            with torch.no_grad():
                layer.weight.copy_(new_weight)
        
        print("Advanced Concept Eraser execution completed.")
        return (model_clone,)

# Define the UCE Eraser Node class (for backward compatibility)
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
                "erase_strength": ("FLOAT", {
                    "default": 1,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.01,
                    "description": "Erasing Strength"
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("modified_model",)
    FUNCTION = "apply_uce"
    CATEGORY = "Model Editing"

    def apply_uce(self, model, clip, concepts_to_erase, baseline_concept, lambda_reg, erase_strength):
        print("\033[92mStarting UCE Eraser Node execution...\033[0m")
        print("Cloning model...")
        model_clone = model.clone()
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        model_clone.model.to(device)
        unet = model_clone.model.diffusion_model
        device = next(unet.parameters()).device
        model_dtype = next(unet.parameters()).dtype  # Get model's dtype (e.g., torch.float16)
        print(f"Model device: {device}, dtype: {model_dtype}")

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
                return embedding.mean(dim=1).squeeze(0).to(device, dtype=model_dtype)  # (2048), cast to model dtype
            raise RuntimeError("Unexpected CLIP output: " + str(result))

        # Compute embeddings with model dtype
        erase_embeddings = torch.stack([get_text_embedding(c) for c in concepts_list])  # (num_concepts, 2048)
        baseline_embedding = get_text_embedding(baseline_concept)  # (2048)
        print(f"Erase embeddings shape: {erase_embeddings.shape}, dtype: {erase_embeddings.dtype}")
        print(f"Baseline shape: {baseline_embedding.shape}, dtype: {baseline_embedding.dtype}")

        # Process cross-attention layers
        for name, module in unet.named_modules():
            if "attn2" in name and hasattr(module, "to_v"):
                print(f"Processing layer: {name}")
                weight = module.to_v.weight
                print(f"Original weight shape: {weight.shape}, dtype: {weight.dtype}, norm: {weight.norm():.4f}")
                out_dim, in_dim = weight.shape

                if in_dim != baseline_embedding.shape[0]:
                    print(f"Skipping layer {name}: weight in_dim {in_dim} != embedding dim {baseline_embedding.shape[0]}")
                    continue

                v_star = torch.matmul(weight, baseline_embedding).unsqueeze(-1)  # (out_dim, 1), e.g., (640, 1)
                C = erase_embeddings.T  # (in_dim, num_concepts), e.g., (2048, 1)

                # Single vs. multi-concept case
                num_concepts = erase_embeddings.shape[0]
                if num_concepts == 1:
                    c = erase_embeddings.squeeze(0)  # (2048)
                    dot_product = torch.dot(c, baseline_embedding)
                    projection = (dot_product / baseline_embedding.norm()**2) * baseline_embedding
                    adjusted_embedding = c - projection
                    new_weight = weight - (v_star @ adjusted_embedding.unsqueeze(0)) / (adjusted_embedding.norm()**2 + lambda_reg)
                else:
                    # For multiple concepts, implement a more aggressive version of concept erasure
                    # that combines orthogonalization with direct concept removal
                    
                    # Normalize baseline embedding for more stable projections
                    baseline_norm = baseline_embedding.norm()
                    baseline_unit = baseline_embedding / baseline_norm
                    
                    # First orthogonalize concepts with respect to baseline
                    concepts_orthogonal = []
                    for i in range(num_concepts):
                        c = erase_embeddings[i]  # (in_dim)
                        # Project concept onto baseline and remove that component
                        dot_product = torch.dot(c, baseline_unit)
                        projection = dot_product * baseline_unit
                        # Make orthogonal to baseline but preserve magnitude
                        adjusted_embedding = c - projection
                        # Normalize to improve numerical stability
                        adjusted_norm = adjusted_embedding.norm()
                        if adjusted_norm > 1e-6:  # Avoid division by very small values
                            adjusted_embedding = adjusted_embedding / adjusted_norm
                        concepts_orthogonal.append(adjusted_embedding)
                    
                    # Stack orthogonalized concepts
                    C_orth = torch.stack(concepts_orthogonal, dim=0)  # (num_concepts, in_dim)
                    
                    # Apply Gram-Schmidt to further orthogonalize between concepts
                    for i in range(1, num_concepts):
                        for j in range(i):
                            # Project concept i onto concept j and remove
                            projection = torch.dot(C_orth[i], C_orth[j]) * C_orth[j]
                            C_orth[i] = C_orth[i] - projection
                        # Renormalize after orthogonalization
                        norm_i = C_orth[i].norm()
                        if norm_i > 1e-6:
                            C_orth[i] = C_orth[i] / norm_i
                    
                    C_orth_T = C_orth.T  # (in_dim, num_concepts)
                    
                    # Calculate influence of concepts on the weight matrix
                    W_c = torch.matmul(weight, C_orth_T)  # (out_dim, num_concepts)
                    
                    # Use a stronger erasure method based on direct projection
                    # This essentially removes any component of the weight that would activate the concept
                    new_weight = weight.clone()
                    
                    # Apply concept erasure with scaling factor for stronger effect
                    erasure_strength = erase_strength  # Scaling factor (higher = stronger erasure)
                    
                    for i in range(num_concepts):
                        concept_vec = C_orth[i]  # (in_dim)
                        # For each output dimension, remove component in concept direction
                        projection = torch.outer(W_c[:, i], concept_vec)  # (out_dim, in_dim)
                        # Apply erasure with minimal regularization to avoid numerical issues
                        regularized_factor = 1.0 / (concept_vec.dot(concept_vec) + max(lambda_reg, 1e-6))
                        new_weight = new_weight - erasure_strength * regularized_factor * projection
                    
                    # Add debugging information
                    print(f"Orthogonalized concepts matrix shape: {C_orth.shape}")
                    print(f"Concept weights matrix shape: {W_c.shape}")
                    print(f"Regularization factor: {lambda_reg}")

                print(f"New weight shape: {new_weight.shape}, dtype: {new_weight.dtype}, norm: {new_weight.norm():.4f}")
                with torch.no_grad():
                    if new_weight.shape != weight.shape:
                        print(f"Shape mismatch: {new_weight.shape} vs {weight.shape}. Skipping.")
                        continue
                    module.to_v.weight.copy_(new_weight.to(dtype=weight.dtype))  # Ensure dtype matches
                    print(f"Weight change norm: {(new_weight - weight).norm():.4f}")

        print("UCE Eraser Node execution completed.")
        return (model_clone,)

# Register the nodes with ComfyUI
NODE_CLASS_MAPPINGS = {
    "ConceptEraserNode": ConceptEraserNode,
    "UCEEraserNode": UCEEraserNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ConceptEraserNode": "Advanced Concept Eraser",
    "UCEEraserNode": "UCE Eraser Node (legacy)"
}

if __name__ == "__main__":
    print("Concept Eraser Nodes loaded successfully.")
