import { app } from "../../scripts/app.js";

app.registerExtension({
<<<<<<< HEAD
    name: "LLM-Polymath.appearance", // Extension name - was ist der richtige Name hier?
    async nodeCreated(node) {
=======
    name: "polymath.appearance", // Extension name - was ist der richtige Name hier?
    nodeCreated(node) {
>>>>>>> da240e8 (first commit)
        if (node.comfyClass === "polymath") {
            // Apply styling
            node.color = "#16727c";
            node.bgcolor = "#4F0074";
        }
    }
});