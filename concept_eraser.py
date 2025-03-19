import torch
import comfy.samplers
import comfy.model_management
import comfy.utils
from comfy.model_patcher import ModelPatcher
import random

class SDXLAblatingDataset:
    def __init__(self, images, clip, placeholder_token, vae, concept_type="object", device="cpu", batch_size=2):
        self.templates = [
            "a photo of a {}",
            "a rendering of a {}",
            "the photo of a {}",
            "a photo of the {}",
        ] if concept_type == "object" else [
            "a painting in the style of {}",
            "artwork in the style of {}",
            "a rendering in the style of {}",
        ]
        self.vae = vae
        self.clip = clip
        self.tokenizer = clip.tokenizer
        self.placeholder_token = placeholder_token
        self.device = device
        # SDXL uses 0.13025 scaling factor instead of 0.18215
        with torch.no_grad():
            self.latents = vae.encode(images[:,:,:,:3]).movedim(-1,1) * 0.13025

    def __len__(self):
        return len(self.latents)

    def get_batch(self, batch_size):
        idx = torch.randint(0, len(self.latents), (batch_size,))
        latents = self.latents[idx]
        prompts = [random.choice(self.templates).format(self.placeholder_token) for _ in range(batch_size)]
        tokens_list = []
        for prompt in prompts:
            tokenized = self.tokenizer.tokenize_with_weights(prompt, return_word_ids=True)
            # Based on your earlier output, tokenized["g"] is structured like:
            # [[<tensor_or_list>, { ... }]]
            # Extract the first element of the inner list:
            extracted = tokenized["g"][0][0]
            # If extracted is a tensor, convert its dtype to torch.long
            if isinstance(extracted, torch.Tensor):
                tokens = extracted.to(torch.long)
            else:
                # Otherwise, convert extracted to a tensor of type long
                tokens = torch.tensor(extracted, dtype=torch.long)
            tokens_list.append(tokens)
        tokens_batch = torch.stack(tokens_list, dim=0)
        return tokens_batch, latents




class ConceptEraserNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "images": ("IMAGE",),
                "concept": ("STRING", {"default": "fox"}),
                "concept_type": (["object", "style"], {"default": "object"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal"}),
                "lr": ("FLOAT", {"default": 1e-5, "min": 1e-6, "max": 1e-3}),
                "steps": ("INT", {"default": 100, "min": 10, "max": 500}),
                "batch_size": ("INT", {"default": 2, "min": 1, "max": 8}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "erase"
    CATEGORY = "model/advanced"

    def freeze_text_encoder(self, clip):
        # For SDXL, freeze both text encoders except the final normalization/layer layers
        for enc in [clip.cond_stage_model.clip_l, clip.cond_stage_model.clip_g]:
            for param in enc.parameters():
                param.requires_grad = False
            for name, module in enc.named_modules():
                if "final_layer_norm" in name or "layer_23" in name:
                    for param in module.parameters():
                        param.requires_grad = True
        return clip

    def erase(self, model, clip, vae, images, concept, concept_type, scheduler, lr, steps, batch_size):
        device = comfy.model_management.get_torch_device()
        dtype = torch.float32 if device.type == 'mps' else comfy.model_management.text_encoder_dtype()

        print(f"Using device: {device}, dtype: {dtype}")
        print(f"Starting concept erasure for: {concept}")

        # Prepare dataset with SDXL-compatible processing
        dataset = SDXLAblatingDataset(
            images=images,
            clip=clip,  # Pass the full CLIP object
            placeholder_token=concept,
            vae=vae,
            concept_type=concept_type,
            device=device,
            batch_size=batch_size
        )

        # Clone and prepare models
        clip_clone = self.freeze_text_encoder(clip.clone())
        model_clone = model.clone()
        optimizer = torch.optim.AdamW(
            list(filter(lambda p: p.requires_grad, clip_clone.cond_stage_model.parameters())),
            lr=lr,
            weight_decay=1e-4
        )
        
        # Get model sampling parameters
        model_sampling = model.get_model_object("model_sampling")
        sigmas = comfy.samplers.calculate_sigmas(
                    model_sampling, 
                    scheduler, 
                    1000  # Total timesteps
                )
        
        # Training loop
        pbar = comfy.utils.ProgressBar(steps)
        for step in range(steps):
            try:
                # Get batch data
                tokens, clean_latents = dataset.get_batch(batch_size)
                noise = torch.randn_like(clean_latents).to(device)
                clean_latents = clean_latents.to(device)
                
                # Sample random timesteps
                bsz = clean_latents.shape[0]
                timesteps = torch.randint(1, 1000, (bsz,), device=device)
                
                # Add noise to latents
                batch_sigmas = sigmas[timesteps.cpu()].view(bsz, 1, 1, 1).to(device)
                noisy_latents = clean_latents + noise * batch_sigmas

                # Encode text embeddings in batch
                tokens_batch = tokens
                text_conds = clip_clone.encode_from_tokens(tokens_batch).to(device)
                
                # Forward pass through UNet
                with torch.cuda.amp.autocast(dtype=dtype):
                    timestep_embed = model_sampling.timestep(timesteps)
                    pred = model_clone.model.apply_model(
                        noisy_latents, 
                        timesteps, 
                        {"c_crossattn": [text_conds], "timestep_embed": timestep_embed}
                    )

                    # Negative MSE loss for concept erasure
                    loss = -torch.nn.functional.mse_loss(pred, noise)

                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(filter(lambda p: p.requires_grad, clip_clone.cond_stage_model.parameters())), 1.0)
                optimizer.step()
                
                pbar.update_absolute(step + 1)
                print(f"Step {step+1}/{steps}, Loss: {loss.item():.6f}")
                
            except Exception as e:
                print(f"Error during step {step+1}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        print(f"Concept erasure complete for: {concept}")
        return (ModelPatcher(model_clone.model, load_device=device, offload_device=comfy.model_management.unet_offload_device()),
                clip_clone)

# Register the nodes with ComfyUI
NODE_CLASS_MAPPINGS = {
    "ConceptEraserNode": ConceptEraserNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConceptEraserNode": "Concept Eraser"
}
