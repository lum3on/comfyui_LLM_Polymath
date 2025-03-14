import torch
from torch import nn
import numpy as np
import comfy.samplers
import comfy.model_management


dummy_positive = ""  # or an appropriate empty condition
dummy_negative = ""
dummy_cfg = 1.0

class ConceptEraserNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),              # Add VAE input
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal", "tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
                "concept_to_erase": ("STRING", {"default": "fox"}),
                "few_shot_images": ("IMAGE",),  # Few-shot images input
                "lr": ("FLOAT", {"default": 1e-5, "min": 1e-6, "max": 1e-4}),
                "epochs": ("INT", {"default": 4, "min": 1, "max": 10}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "erase_concept"
    CATEGORY = "model_patches"

    def erase_concept(self, unet, clip, vae, scheduler, concept_to_erase, few_shot_images, lr, epochs):
        # Clone CLIP to avoid modifying the original
        clip_clone = clip.clone()
        text_encoder = clip_clone.cond_stage_model  # Access CLIP's text encoder

        # Identify target layers: MLP blocks and final self-attention
        target_layers = []
        for name, module in text_encoder.named_modules():
            if "mlp" in name or ("self_attn" in name and "final_layer_norm" in name):
                target_layers.append(name)
        
        # Freeze all parameters except target layers
        for param in text_encoder.parameters():
            param.requires_grad = False
        for name, param in text_encoder.named_parameters():
            if any(layer in name for layer in target_layers):
                param.requires_grad = True

        # Set up optimizer
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, text_encoder.parameters()),
            lr=lr,
            betas=(0.9, 0.98),
            weight_decay=1e-8
        )

        # Preprocess few-shot images into latents (assuming VAE is available)
        with torch.no_grad():
            latents = vae.encode(few_shot_images) * 0.18215
            print("Latents shape:", latents.shape)

        sampler = comfy.samplers.KSampler(
            unet,
            steps=1,  # use one step so we only perform the noise addition part
            device = comfy.model_management.intermediate_device(),
            sampler="Euler",  # or another valid sampler name from KSampler.SAMPLERS
            scheduler=scheduler,     # pass the scheduler string here
            denoise=1.0,
            model_options=unet.model_options
)
        # Training loop
        for epoch in range(epochs):
            # Add noise to latents

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (latents.shape[0],))

            noisy_latents = sampler.sample(
                noise,
                dummy_positive,
                dummy_negative,
                cfg=dummy_cfg,
                latent_image=latents
            )

            # Forward pass through UNet with updated text encoder
            tokens = clip_clone.tokenize(concept_to_erase)
            print(dir(text_encoder))
            text_embeds = text_encoder.encode_from_tokens(tokens)[0]            
            noise_pred = unet(noisy_latents, timesteps, text_embeds).sample

            # Maximize the loss (negative L2 loss)
            loss = -torch.nn.functional.mse_loss(noise_pred, noise)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return (unet, clip_clone)  # Return updated CLIP

# Register the nodes with ComfyUI
NODE_CLASS_MAPPINGS = {
    "ConceptEraserNode": ConceptEraserNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConceptEraserNode": "Advanced Concept Eraser",
}

if __name__ == "__main__":
    print("Concept Eraser Nodes loaded successfully.")