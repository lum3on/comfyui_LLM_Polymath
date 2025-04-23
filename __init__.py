from .polymath import PolymathSettings, Polymath, MediaScraper
from .helper import SaveAbs, TextSplitter, StringListPicker
#from .concept_eraser import ConceptEraserNode
from .textmask import TextMaskNode

NODE_CLASS_MAPPINGS = {
    "polymath_settings": PolymathSettings,
    "polymath_chat": Polymath,
    "polymath_scraper": MediaScraper,
    "polymath_helper": SaveAbs,
    "polymath_TextSplitter": TextSplitter,
    "polymath_StringListPicker": StringListPicker,
    #"polymath_concept_eraser": ConceptEraserNode,
    "polymath_text_mask": TextMaskNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "polymath_chat": "LLM Polymath Chat with Advanced Web and Link Search",
    "polymath_scraper": "LLM Polymath Scraper for various sites",
    "polymath_helper": "Save Image to Absolute Path",
    "polymath_TextSplitter": "Split Texts by Specified Delimiter",
    "polymath_StringListPicker": "Picks Texts from a List by Index",
    "polymath_concept_eraser": "Erase Concept from Model"
    #"polymath_text_mask": "Generate mask from text"
}

ascii_art = """
Polymath is brought to you by

       _  _                                    _  _  _  _                                   
      (▒)(▒)                                 _(▒)(▒)(▒)(▒)_                                 
         (▒)   _         _     _  _   _  _  (▒)          (▒)    _  _  _     _  _  _  _      
         (▒)  (▒)       (▒)   (▒)(▒)_(▒)(▒)          _  _(▒) _ (▒)(▒)(▒) _ (▒)(▒)(▒)(▒)_    
         (▒)  (▒)       (▒)  (▒)   (▒)   (▒)        (▒)(▒)_ (▒)         (▒)(▒)        (▒)   
         (▒)  (▒)       (▒)  (▒)   (▒)   (▒) _           (▒)(▒)         (▒)(▒)        (▒)   
       _ (▒) _(▒)_  _  _(▒)_ (▒)   (▒)   (▒)(▒)_  _  _  _(▒)(▒) _  _  _ (▒)(▒)        (▒)   
      (▒)(▒)(▒) (▒)(▒)(▒) (▒)(▒)   (▒)   (▒)  (▒)(▒)(▒)(▒)     (▒)(▒)(▒)   (▒)        (▒)                                                                                              
"""
print(f"\033[92m{ascii_art}\033[0m")

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
