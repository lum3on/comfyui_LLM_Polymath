from .polymath import Polymath

NODE_CLASS_MAPPINGS = {
    "polymath_chat": Polymath
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "polymath_chat": "LLM Polymath Chat with Advanced Web and Link Search",
}

WEB_DIRECTORY = "web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]