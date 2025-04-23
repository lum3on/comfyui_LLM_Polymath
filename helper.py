import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import base64,os,random, string
import folder_paths
import json,io
from comfy.cli_args import args
import math,glob

def generate_random_string(length):
        characters = string.ascii_letters + string.digits
        return ''.join(random.choice(characters) for _ in range(length))

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
 
class TextSplitter:
    DELIMITER_NEWLINE = "\\n"  # Display as "\n" in UI
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_string": ("STRING", {
                    "multiline": True,
                    "forceInput": True
                }),
                "delimiter": ("STRING", {
                    "default": cls.DELIMITER_NEWLINE,
                    "multiline": False
                }),
                "ignore_before_equals": ("BOOLEAN", {
                    "default": False
                }),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("splitted_texts",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "split_string"
    CATEGORY = "Polymath/helper"
    
    def split_string(s3elf, input_string, delimiter, ignore_before_equals):
        # Handle the special case for newline delimiter
        actual_delimiter = "\n" if delimiter == self.DELIMITER_NEWLINE else delimiter
        
        # Split the string and process parts
        parts = input_string.split(actual_delimiter)
        result = []
        
        # Process each part
        for part in parts:
            cleaned_part = part.strip()
            if ignore_before_equals and '=' in cleaned_part:
                cleaned_part = cleaned_part.split('=', 1)[1].strip()
            result.append(cleaned_part)
        
        return (result,)
    
def wrapIndex(index, length):
    """Helper function to handle index wrapping for out-of-bounds indices."""
    if length == 0:
        return 0, 0
    wraps = index // length if index >= 0 else (index + 1) // length - 1
    wrapped_index = index % length if index >= 0 else (index % length + length) % length
    return wrapped_index, wraps

class StringListPicker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list_input": ("STRING", {
                    "forceInput": True,
                }),
                "index": ("INT", {
                    "default": 0,
                    "min": -999,
                    "max": 999,
                    "step": 1
                }),
            },
        }
    
    RETURN_TYPES = ("STRING", "INT", "INT",)
    RETURN_NAMES = ("list_item", "size", "wraps",)
    
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, False, True)
    
    FUNCTION = "pick"
    CATEGORY = "Polymath/helper"
    
    def pick(self, list_input, index):
        # Ensure list_input is a list of strings
        if not list_input:
            return ([], 0, [0] * len(index))
        
        length = len(list_input)
        
        # Process each index
        item_list = []
        wraps_list = []
        for i in index:
            index_mod, wraps = wrapIndex(i, length)
            item_list.append(list_input[index_mod] if length > 0 else "")
            wraps_list.append(wraps)
        
        return (item_list, length, wraps_list,)
 
# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "polymath_SaveAbsolute": SaveAbs,
    "polymath_TextSplitter": TextSplitter,
    "polymath_StringListPicker": StringListPicker
}
 
# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "polymath_SaveAbsolute": "Save Image to Absolute Path",
    "polymath_TextSplitter": "Split Texts by Specified Delimiter",
    "polymath_StringListPicker": "Picks Texts from a List by Index"
}