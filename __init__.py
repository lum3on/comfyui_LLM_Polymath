# Import FluxContextPreset first (always works)
try:
    from .flux_context_preset import FluxContextPreset
except ImportError:
    from flux_context_preset import FluxContextPreset

# Initialize mappings with FluxContextPreset
NODE_CLASS_MAPPINGS = {
    "flux_context_preset": FluxContextPreset,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "flux_context_preset": "Flux Context Preset",
}

# Try to import other nodes
try:
    try:
        from .polymath import PolymathSettings, Polymath
        from .media_scraper import MediaScraper
        from .helper import SaveAbs, TextSplitter, StringListPicker
        from .textmask import TextMaskNode
    except ImportError:
        from polymath import PolymathSettings, Polymath
        from media_scraper import MediaScraper
        from helper import SaveAbs, TextSplitter, StringListPicker
        from textmask import TextMaskNode

    # Add other nodes to mappings if imports successful
    NODE_CLASS_MAPPINGS.update({
        "polymath_settings": PolymathSettings,
        "polymath_chat": Polymath,
        "polymath_scraper": MediaScraper,
        "polymath_helper": SaveAbs,
        "polymath_TextSplitter": TextSplitter,
        "polymath_StringListPicker": StringListPicker,
        "polymath_text_mask": TextMaskNode,
    })

    NODE_DISPLAY_NAME_MAPPINGS.update({
        "polymath_settings": "Polymath Settings",
        "polymath_chat": "LLM Polymath Chat with Advanced Web and Link Search",
        "polymath_scraper": "Media Scraper",
        "polymath_helper": "Save Image to Absolute Path",
        "polymath_TextSplitter": "Split Texts by Specified Delimiter",
        "polymath_StringListPicker": "Picks Texts from a List by Index",
        "polymath_text_mask": "Generate mask from text",
    })

except ImportError as e:
    print(f"Warning: Some Polymath nodes could not be imported: {e}")
    print("Only FluxContextPreset will be available.")

ascii_art = """
Polymath is brought to you by

       _  _                                 _  _
      (▒)(▒)   _      _     _  _   _  _   _(▒)(▒)_     _  _     _  _
         (▒)  (▒)    (▒)   (▒)(▒)_(▒)(▒) (▒)   _(▒)   (▒)(▒)_  (▒)(▒)
         (▒)  (▒)    (▒)  (▒)   (▒)   (▒)     (▒)(▒)(▒)    (▒)(▒)  (▒)
       _ (▒) _(▒)_  _(▒)_ (▒)   (▒)   (▒)(▒)_  _(▒) (▒)_  _(▒)(▒)  (▒)
      (▒)(▒)(▒) (▒)(▒) (▒)(▒)   (▒)   (▒)  (▒)(▒)     (▒)(▒)  (▒)  (▒)
"""
print(f"\033[92m{ascii_art}\033[0m")

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
