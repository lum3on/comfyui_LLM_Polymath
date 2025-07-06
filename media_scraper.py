import subprocess
import os, glob
import shutil
import torch
import folder_paths
from PIL import Image
import numpy as np

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

    def execute(self, urls, output_file_path, file_name, image_load_cap=0, keep_temp_path=False):
       
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