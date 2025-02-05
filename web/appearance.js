import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "llm_polymath.appearance", // Registrierter Extenstionname von Comfy
    async nodeCreated(node) {
        if (node.comfyClass === "polymath_chat") { // das was in __init__ unter NODE_CLASS_MAPPINGS steht
            // Apply styling
            node.color = "#9968f3";
            node.bgcolor = "#519b36";
        }
    }
});