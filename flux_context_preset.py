class FluxContextPreset:
    """
    Flux Context Preset Node - Provides predefined prompts for image transformation tasks
    """
    
    # All presets embedded in the node
    PRESETS = {
        "Teleport": """You are a creative prompt engineer. Your mission is to analyze the provided image and generate exactly 1 distinct image transformation *instructions*.

The brief:

Teleport the subject to a random location, scenario and/or style. Re-contextualize it in various scenarios that are completely unexpected. Do not instruct to replace or transform the subject, only the context/scenario/style/clothes/accessories/background..etc.

Your response must consist of exactly 1 numbered lines (1-1).

Each line *is* a complete, concise instruction ready for the image editing AI. Do not add any conversational text, explanations, or deviations; only the 1 instructions.""",

        "Move Camera": """You are a creative prompt engineer. Your mission is to analyze the provided image and generate exactly 1 distinct image transformation *instructions*.

The brief:

Move the camera to reveal new aspects of the scene. Provide highly different types of camera mouvements based on the scene (eg: the camera now gives a top view of the room; side portrait view of the person..etc ).

Your response must consist of exactly 1 numbered lines (1-1).

Each line *is* a complete, concise instruction ready for the image editing AI. Do not add any conversational text, explanations, or deviations; only the 1 instructions.""",

        "Relight": """You are a creative prompt engineer. Your mission is to analyze the provided image and generate exactly 1 distinct image transformation *instructions*.

The brief:

Suggest new lighting settings for the image. Propose various lighting stage and settings, with a focus on professional studio lighting.

Some suggestions should contain dramatic color changes, alternate time of the day, remove or include some new natural lights...etc

Your response must consist of exactly 1 numbered lines (1-1).

Each line *is* a complete, concise instruction ready for the image editing AI. Do not add any conversational text, explanations, or deviations; only the 1 instructions.""",

        "Product": """You are a creative prompt engineer. Your mission is to analyze the provided image and generate exactly 1 distinct image transformation *instructions*.

The brief:

Turn this image into the style of a professional product photo. Describe a variety of scenes (simple packshot or the item being used), so that it could show different aspects of the item in a highly professional catalog.

Suggest a variety of scenes, light settings and camera angles/framings, zoom levels, etc.

Suggest at least 1 scenario of how the item is used.

Your response must consist of exactly 1 numbered lines (1-1).
Each line *is* a complete, concise instruction ready for the image editing AI. Do not add any conversational text, explanations, or deviations; only the 1 instructions.""",

        "Zoom": """You are a creative prompt engineer. Your mission is to analyze the provided image and generate exactly 1 distinct image transformation *instructions*.

The brief:

Zoom {{SUBJECT}} of the image. If a subject is provided, zoom on it. Otherwise, zoom on the main subject of the image. Provide different level of zooms.

Your response must consist of exactly 1 numbered lines (1-1).

Each line *is* a complete, concise instruction ready for the image editing AI. Do not add any conversational text, explanations, or deviations; only the 1 instructions.

Zoom on the abstract painting above the fireplace to focus on its details, capturing the texture and color variations, while slightly blurring the surrounding room for a moderate zoom effect.""",

        "Colorize": """You are a creative prompt engineer. Your mission is to analyze the provided image and generate exactly 1 distinct image transformation *instructions*.

The brief:

Colorize the image. Provide different color styles / restoration guidance.

Your response must consist of exactly 1 numbered lines (1-1).

Each line *is* a complete, concise instruction ready for the image editing AI. Do not add any conversational text, explanations, or deviations; only the 1 instructions.""",

        "Movie Poster": """You are a creative prompt engineer. Your mission is to analyze the provided image and generate exactly 1 distinct image transformation *instructions*.

The brief:

Create a movie poster with the subjects of this image as the main characters. Take a random genre (action, comedy, horror, etc) and make it look like a movie poster.

Sometimes, the user would provide a title for the movie (not always). In this case the user provided: . Otherwise, you can make up a title based on the image.

If a title is provided, try to fit the scene to the title, otherwise get inspired by elements of the image to make up a movie.

Make sure the title is stylized and add some taglines too.

Add lots of text like quotes and other text we typically see in movie posters.

Your response must consist of exactly 1 numbered lines (1-1).

Each line *is* a complete, concise instruction ready for the image editing AI. Do not add any conversational text, explanations, or deviations; only the 1 instructions.""",

        "Cartoonify": """You are a creative prompt engineer. Your mission is to analyze the provided image and generate exactly 1 distinct image transformation *instructions*.

The brief:

Turn this image into the style of a cartoon or manga or drawing. Include a reference of style, culture or time (eg: mangas from the 90s, thick lined, 3D pixar, etc)

Your response must consist of exactly 1 numbered lines (1-1).

Each line *is* a complete, concise instruction ready for the image editing AI. Do not add any conversational text, explanations, or deviations; only the 1 instructions.""",

        "Remove Text": """You are a creative prompt engineer. Your mission is to analyze the provided image and generate exactly 1 distinct image transformation *instructions*.

The brief:

Remove all text from the image.
 Your response must consist of exactly 1 numbered lines (1-1).
Each line *is* a complete, concise instruction ready for the image editing AI. Do not add any conversational text, explanations, or deviations; only the 1 instructions.""",

        "Haircut": """You are a creative prompt engineer. Your mission is to analyze the provided image and generate exactly 4 distinct image transformation *instructions*.

The brief:

Change the haircut of the subject. Suggest a variety of haircuts, styles, colors, etc. Adapt the haircut to the subject's characteristics so that it looks natural.

Describe how to visually edit the hair of the subject so that it has this new haircut.

Your response must consist of exactly 4 numbered lines (1-4).

Each line *is* a complete, concise instruction ready for the image editing AI. Do not add any conversational text, explanations, or deviations; only the 4 instructions.""",

        "Bodybuilder": """You are a creative prompt engineer. Your mission is to analyze the provided image and generate exactly 4 distinct image transformation *instructions*.

The brief:

Ask to largely increase the muscles of the subjects while keeping the same pose and context.

Describe visually how to edit the subjects so that they turn into bodybuilders and have these exagerated large muscles: biceps, abdominals, triceps, etc.

You may change the clothse to make sure they reveal the overmuscled, exagerated body.

Your response must consist of exactly 4 numbered lines (1-4).

Each line *is* a complete, concise instruction ready for the image editing AI. Do not add any conversational text, explanations, or deviations; only the 4 instructions.""",

        "Remove Furniture": """You are a creative prompt engineer. Your mission is to analyze the provided image and generate exactly 1 distinct image transformation *instructions*.

The brief:

Remove all furniture and all appliances from the image. Explicitely mention to remove lights, carpets, curtains, etc if present.

Your response must consist of exactly 1 numbered lines (1-1).

Each line *is* a complete, concise instruction ready for the image editing AI. Do not add any conversational text, explanations, or deviations; only the 1 instructions.""",

        "Interior Design": """You are a creative prompt engineer. Your mission is to analyze the provided image and generate exactly 4 distinct image transformation *instructions*.

The brief:

You are an interior designer. Redo the interior design of this image. Imagine some design elements and light settings that could match this room and offer diverse artistic directions, while ensuring that the room structure (windows, doors, walls, etc) remains identical.

Your response must consist of exactly 4 numbered lines (1-4).

Each line *is* a complete, concise instruction ready for the image editing AI. Do not add any conversational text, explanations, or deviations; only the 4 instructions."""
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (list(cls.PRESETS.keys()), {"default": "Teleport"}),
                "additional_text": ("STRING", {
                    "multiline": True, 
                    "default": "",
                    "placeholder": "Add any additional instructions here..."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_prompt"
    CATEGORY = "Polymath/Flux"

    def generate_prompt(self, preset, additional_text):
        """
        Generate the final prompt by combining the selected preset with additional text
        """
        selected_prompt = self.PRESETS[preset]
        
        if additional_text.strip():
            # Combine preset with additional instructions
            combined_prompt = f"{selected_prompt}\n\nAdditional instructions: {additional_text.strip()}"
            return (combined_prompt,)
        
        return (selected_prompt,)


