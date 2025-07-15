import { app } from "../../scripts/app.js";

// All presets data for the revelation window
const FLUX_PRESETS = {
    "Teleport": `You are a creative prompt engineer. Your mission is to analyze the provided image and generate exactly 1 distinct image transformation *instructions*.

The brief:

Teleport the subject to a random location, scenario and/or style. Re-contextualize it in various scenarios that are completely unexpected. Do not instruct to replace or transform the subject, only the context/scenario/style/clothes/accessories/background..etc.

Your response must consist of exactly 1 numbered lines (1-1).

Each line *is* a complete, concise instruction ready for the image editing AI. Do not add any conversational text, explanations, or deviations; only the 1 instructions.`,

    "Move Camera": `You are a creative prompt engineer. Your mission is to analyze the provided image and generate exactly 1 distinct image transformation *instructions*.

The brief:

Move the camera to reveal new aspects of the scene. Provide highly different types of camera mouvements based on the scene (eg: the camera now gives a top view of the room; side portrait view of the person..etc ).

Your response must consist of exactly 1 numbered lines (1-1).

Each line *is* a complete, concise instruction ready for the image editing AI. Do not add any conversational text, explanations, or deviations; only the 1 instructions.`,

    "Relight": `You are a creative prompt engineer. Your mission is to analyze the provided image and generate exactly 1 distinct image transformation *instructions*.

The brief:

Suggest new lighting settings for the image. Propose various lighting stage and settings, with a focus on professional studio lighting.

Some suggestions should contain dramatic color changes, alternate time of the day, remove or include some new natural lights...etc

Your response must consist of exactly 1 numbered lines (1-1).

Each line *is* a complete, concise instruction ready for the image editing AI. Do not add any conversational text, explanations, or deviations; only the 1 instructions.`,

    "Product": `You are a creative prompt engineer. Your mission is to analyze the provided image and generate exactly 1 distinct image transformation *instructions*.

The brief:

Turn this image into the style of a professional product photo. Describe a variety of scenes (simple packshot or the item being used), so that it could show different aspects of the item in a highly professional catalog.

Suggest a variety of scenes, light settings and camera angles/framings, zoom levels, etc.

Suggest at least 1 scenario of how the item is used.

Your response must consist of exactly 1 numbered lines (1-1).
Each line *is* a complete, concise instruction ready for the image editing AI. Do not add any conversational text, explanations, or deviations; only the 1 instructions.`,

    "Zoom": `You are a creative prompt engineer. Your mission is to analyze the provided image and generate exactly 1 distinct image transformation *instructions*.

The brief:

Zoom {{SUBJECT}} of the image. If a subject is provided, zoom on it. Otherwise, zoom on the main subject of the image. Provide different level of zooms.

Your response must consist of exactly 1 numbered lines (1-1).

Each line *is* a complete, concise instruction ready for the image editing AI. Do not add any conversational text, explanations, or deviations; only the 1 instructions.

Zoom on the abstract painting above the fireplace to focus on its details, capturing the texture and color variations, while slightly blurring the surrounding room for a moderate zoom effect.`,

    "Colorize": `You are a creative prompt engineer. Your mission is to analyze the provided image and generate exactly 1 distinct image transformation *instructions*.

The brief:

Colorize the image. Provide different color styles / restoration guidance.

Your response must consist of exactly 1 numbered lines (1-1).

Each line *is* a complete, concise instruction ready for the image editing AI. Do not add any conversational text, explanations, or deviations; only the 1 instructions.`,

    "Movie Poster": `You are a creative prompt engineer. Your mission is to analyze the provided image and generate exactly 1 distinct image transformation *instructions*.

The brief:

Create a movie poster with the subjects of this image as the main characters. Take a random genre (action, comedy, horror, etc) and make it look like a movie poster.

Sometimes, the user would provide a title for the movie (not always). In this case the user provided: . Otherwise, you can make up a title based on the image.

If a title is provided, try to fit the scene to the title, otherwise get inspired by elements of the image to make up a movie.

Make sure the title is stylized and add some taglines too.

Add lots of text like quotes and other text we typically see in movie posters.

Your response must consist of exactly 1 numbered lines (1-1).

Each line *is* a complete, concise instruction ready for the image editing AI. Do not add any conversational text, explanations, or deviations; only the 1 instructions.`,

    "Cartoonify": `You are a creative prompt engineer. Your mission is to analyze the provided image and generate exactly 1 distinct image transformation *instructions*.

The brief:

Turn this image into the style of a cartoon or manga or drawing. Include a reference of style, culture or time (eg: mangas from the 90s, thick lined, 3D pixar, etc)

Your response must consist of exactly 1 numbered lines (1-1).

Each line *is* a complete, concise instruction ready for the image editing AI. Do not add any conversational text, explanations, or deviations; only the 1 instructions.`,

    "Remove Text": `You are a creative prompt engineer. Your mission is to analyze the provided image and generate exactly 1 distinct image transformation *instructions*.

The brief:

Remove all text from the image.
 Your response must consist of exactly 1 numbered lines (1-1).
Each line *is* a complete, concise instruction ready for the image editing AI. Do not add any conversational text, explanations, or deviations; only the 1 instructions.`,

    "Haircut": `You are a creative prompt engineer. Your mission is to analyze the provided image and generate exactly 4 distinct image transformation *instructions*.

The brief:

Change the haircut of the subject. Suggest a variety of haircuts, styles, colors, etc. Adapt the haircut to the subject's characteristics so that it looks natural.

Describe how to visually edit the hair of the subject so that it has this new haircut.

Your response must consist of exactly 4 numbered lines (1-4).

Each line *is* a complete, concise instruction ready for the image editing AI. Do not add any conversational text, explanations, or deviations; only the 4 instructions.`,

    "Bodybuilder": `You are a creative prompt engineer. Your mission is to analyze the provided image and generate exactly 4 distinct image transformation *instructions*.

The brief:

Ask to largely increase the muscles of the subjects while keeping the same pose and context.

Describe visually how to edit the subjects so that they turn into bodybuilders and have these exagerated large muscles: biceps, abdominals, triceps, etc.

You may change the clothse to make sure they reveal the overmuscled, exagerated body.

Your response must consist of exactly 4 numbered lines (1-4).

Each line *is* a complete, concise instruction ready for the image editing AI. Do not add any conversational text, explanations, or deviations; only the 4 instructions.`,

    "Remove Furniture": `You are a creative prompt engineer. Your mission is to analyze the provided image and generate exactly 1 distinct image transformation *instructions*.

The brief:

Remove all furniture and all appliances from the image. Explicitely mention to remove lights, carpets, curtains, etc if present.

Your response must consist of exactly 1 numbered lines (1-1).

Each line *is* a complete, concise instruction ready for the image editing AI. Do not add any conversational text, explanations, or deviations; only the 1 instructions.`,

    "Interior Design": `You are a creative prompt engineer. Your mission is to analyze the provided image and generate exactly 4 distinct image transformation *instructions*.

The brief:

You are an interior designer. Redo the interior design of this image. Imagine some design elements and light settings that could match this room and offer diverse artistic directions, while ensuring that the room structure (windows, doors, walls, etc) remains identical.

Your response must consist of exactly 4 numbered lines (1-4).

Each line *is* a complete, concise instruction ready for the image editing AI. Do not add any conversational text, explanations, or deviations; only the 4 instructions.`
};

// Function to create and show the preset revelation modal
function showPresetModal() {
    // Create modal overlay
    const modal = document.createElement('div');
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.8);
        z-index: 10000;
        display: flex;
        justify-content: center;
        align-items: center;
    `;

    // Create modal content
    const modalContent = document.createElement('div');
    modalContent.style.cssText = `
        background-color: #2a2a2a;
        border: 2px solid #555;
        border-radius: 8px;
        width: 90%;
        max-width: 1000px;
        max-height: 90%;
        overflow-y: auto;
        padding: 20px;
        color: #fff;
        font-family: monospace;
    `;

    // Create header
    const header = document.createElement('div');
    header.style.cssText = `
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
        border-bottom: 1px solid #555;
        padding-bottom: 10px;
    `;

    const title = document.createElement('h2');
    title.textContent = 'Flux Context Presets - System Prompts';
    title.style.cssText = `
        margin: 0;
        color: #9968f3;
        font-size: 24px;
    `;

    const closeButton = document.createElement('button');
    closeButton.textContent = '✕';
    closeButton.style.cssText = `
        background: #ff4444;
        border: none;
        color: white;
        font-size: 20px;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
    `;
    closeButton.onclick = () => document.body.removeChild(modal);

    header.appendChild(title);
    header.appendChild(closeButton);

    // Create content area
    const content = document.createElement('div');
    
    // Add each preset
    Object.entries(FLUX_PRESETS).forEach(([name, prompt]) => {
        const presetDiv = document.createElement('div');
        presetDiv.style.cssText = `
            margin-bottom: 25px;
            border: 1px solid #444;
            border-radius: 5px;
            overflow: hidden;
        `;

        const presetHeader = document.createElement('div');
        presetHeader.style.cssText = `
            background-color: #9968f3;
            color: white;
            padding: 10px 15px;
            font-weight: bold;
            font-size: 16px;
        `;
        presetHeader.textContent = name;

        const presetContent = document.createElement('pre');
        presetContent.style.cssText = `
            background-color: #1a1a1a;
            color: #e0e0e0;
            padding: 15px;
            margin: 0;
            white-space: pre-wrap;
            font-size: 12px;
            line-height: 1.4;
            border: none;
        `;
        presetContent.textContent = prompt;

        presetDiv.appendChild(presetHeader);
        presetDiv.appendChild(presetContent);
        content.appendChild(presetDiv);
    });

    modalContent.appendChild(header);
    modalContent.appendChild(content);
    modal.appendChild(modalContent);

    // Close modal when clicking outside
    modal.onclick = (e) => {
        if (e.target === modal) {
            document.body.removeChild(modal);
        }
    };

    document.body.appendChild(modal);
}

// Register the extension
app.registerExtension({
    name: "flux_context_preset.revelation",
    async nodeCreated(node) {
        if (node.comfyClass === "flux_context_preset") {
            // Apply styling
            node.color = "#9968f3";
            node.bgcolor = "#2a2a2a";

            // Create clickable icon in the node title area
            const createRevealIcon = () => {
                const icon = document.createElement('div');
                icon.innerHTML = '📋';
                icon.title = 'Show All Presets';
                icon.style.cssText = `
                    position: absolute;
                    top: 8px;
                    right: 8px;
                    width: 24px;
                    height: 24px;
                    background: linear-gradient(45deg, #9968f3, #7c4dff);
                    border: 2px solid #fff;
                    border-radius: 50%;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 12px;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.4);
                    transition: all 0.2s ease;
                    z-index: 1000;
                    user-select: none;
                `;

                icon.onmouseover = () => {
                    icon.style.transform = 'scale(1.1)';
                    icon.style.boxShadow = '0 4px 12px rgba(153, 104, 243, 0.6)';
                };

                icon.onmouseout = () => {
                    icon.style.transform = 'scale(1)';
                    icon.style.boxShadow = '0 2px 6px rgba(0,0,0,0.4)';
                };

                icon.onclick = (e) => {
                    e.stopPropagation();
                    showPresetModal();
                };

                return icon;
            };

            // Add icon to node after it's rendered
            setTimeout(() => {
                const nodeElement = node.canvas?.canvas?.parentElement?.querySelector(`[data-id="${node.id}"]`) ||
                                  document.querySelector(`[data-id="${node.id}"]`);

                if (nodeElement) {
                    // Make sure the node has relative positioning for absolute positioning of icon
                    nodeElement.style.position = 'relative';

                    // Remove any existing icon
                    const existingIcon = nodeElement.querySelector('.preset-reveal-icon');
                    if (existingIcon) {
                        existingIcon.remove();
                    }

                    // Add the new icon
                    const icon = createRevealIcon();
                    icon.classList.add('preset-reveal-icon');
                    nodeElement.appendChild(icon);
                } else {
                    // Fallback: try to find the node element in the DOM
                    const observer = new MutationObserver((mutations) => {
                        const nodeEl = document.querySelector(`[data-id="${node.id}"]`);
                        if (nodeEl) {
                            nodeEl.style.position = 'relative';
                            const icon = createRevealIcon();
                            icon.classList.add('preset-reveal-icon');
                            nodeEl.appendChild(icon);
                            observer.disconnect();
                        }
                    });
                    observer.observe(document.body, { childList: true, subtree: true });

                    // Stop observing after 5 seconds to prevent memory leaks
                    setTimeout(() => observer.disconnect(), 5000);
                }
            }, 200);
        }
    }
});
