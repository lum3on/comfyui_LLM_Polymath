from .polymath import Polymath, MediaScraper

NODE_CLASS_MAPPINGS = {
    "polymath_chat": Polymath,
    "polymath_scraper": MediaScraper
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "polymath_chat": "LLM Polymath Chat with Advanced Web and Link Search",
    "polymath_scraper": "LLM Polymath Scraper for various sites"
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
