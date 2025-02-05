import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "polymath.appearance", // Extension name - was ist der richtige Name hier?
    async nodeCreated(node) {
        if (node.comfyClass === "polymath") { // was ist der richtige Name hier?
            // Apply styling
            node.color = "#16727c";
            node.bgcolor = "#4F0074";
        }
    }
});