import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import base64,os,random
import folder_paths
import json,io
from comfy.cli_args import args
import math,glob

class SaveAbs:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"images": ("IMAGE", ),
                     "file_path": ("STRING",{"multiline": True,"default": "","dynamicPrompts": False}),
                     },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "Polymath/helper"

    def save_images(self, images,file_path , prompt=None, extra_pnginfo=None):
        filename_prefix = os.path.basename(file_path)
        if file_path=='':
            filename_prefix="ComfyUI"
        
        filename_prefix, _ = os.path.splitext(filename_prefix)

        _, extension = os.path.splitext(file_path)

        if extension:
            # is the file name and needs to be processed
            file_path=os.path.dirname(file_path)
            # filename_prefix=

            
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        
    
        if not os.path.exists(file_path):
            # Create a new directory using the os.makedirs function
            os.makedirs(file_path)
            print("dir created")
        else:
            print("dir already exists")

        # Use the glob module to get all files in the current directory
        if file_path=="":
            files = glob.glob(full_output_folder + '/*')
        else:
            files = glob.glob(file_path + '/*')
        # Number of statistical files
        file_count = len(files)
        counter+=file_count
        print('number of files',file_count,counter)

        results = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            file = f"{filename}_{counter:05}.png"
            
            if file_path=="":
                fp=os.path.join(full_output_folder, file)
                if os.path.exists(fp):
                    file = f"{filename}_{counter:05}_{generate_random_string(8)}.png"
                    fp=os.path.join(full_output_folder, file)
                img.save(fp, pnginfo=metadata, compress_level=self.compress_level)
                results.append({
                    "filename": file,
                    "subfolder": subfolder,
                    "type": self.type
                })
            
            else:

                fp=os.path.join(file_path, file)
                if os.path.exists(fp):
                    file = f"{filename}_{counter:05}_{generate_random_string(8)}.png"
                    fp=os.path.join(file_path, file)

                img.save(os.path.join(file_path, file), pnginfo=metadata, compress_level=self.compress_level)
                results.append({
                    "filename": file,
                    "subfolder": file_path,
                    "type": self.type
                })
            counter += 1

        return ()
 
 
# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "polymath_SaveAbsolute": SaveAbs
}
 
# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "polymath_SaveAbsolute": "Save Image to Absolute Path"
}