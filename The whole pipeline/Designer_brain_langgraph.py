import json
from pathlib import Path
import re
from graph_layout import app


# ----------------------------
# CONFIG (keep aligned with your existing script)
# ----------------------------
MODEL_PATH = "/teamspace/studios/this_studio/Designer/models/Qwen3-8B-Q4_K_M.gguf"
INPUT_JSON_PATH = "/teamspace/studios/this_studio/Designer/The whole pipeline/noha.json"
# INPUT_JSON_PATH = "8_slides.json"
OUTPUT_JSON_PATH = "powerpoint_layout_29_12.json"
MAX_TOKENS = 4096 * 2 * 2 * 2
N_GPU_LAYERS = -1


with open("config.json", "r", encoding="utf-8") as f:
    config_data = json.load(f)

img_gen_config = config_data.get("configuration", {}).get("generate_images", False)
print("image generation choice is:", img_gen_config)
# ----------------------------
# IMPORTANT:
# Do NOT change your system/user prompts here.
# Paste your existing system_message and user_message EXACTLY as-is from Designer_brain.py.
# ----------------------------

# NOTE: You asked not to change prompts. This file expects you to paste them in.
# To keep this file runnable out-of-the-box without touching your prompt content,
# we load them directly from Designer_brain.py as plain text (no execution).
def _load_prompts_from_designer_brain(path="Designer_brain.py"):
    text = Path(path).read_text(encoding="utf-8")

    # Extract system_message triple-quoted string
    sm = re.search(r'system_message\s*=\s*"""(.*?)"""', text, re.DOTALL)
    um = re.search(r'user_message\s*=\s*f?"""(.*?)"""', text, re.DOTALL)

    if not sm:
        raise RuntimeError("Could not find system_message in Designer_brain.py")
    if not um:
        raise RuntimeError("Could not find user_message in Designer_brain.py")

    system_message = sm.group(1)
    user_message_template = um.group(1)
    return system_message, user_message_template


def main():
    # Load input JSON
    input_data = json.loads(Path(INPUT_JSON_PATH).read_text(encoding="utf-8"))

    system_message, user_message_template = _load_prompts_from_designer_brain("Designer_brain.py")

    # Your existing Designer_brain.py uses an f-string template for user_message.
    # We reproduce that behavior WITHOUT changing the text itself:

    user_message = user_message_template.replace(
        "{input_data}",
        json.dumps(input_data, indent=2)
    )


    state = {
        "input_data": input_data,
        "system_message": system_message,
        "user_message": user_message,
        "model_path": MODEL_PATH,
        "max_tokens": MAX_TOKENS,
        "n_gpu_layers": N_GPU_LAYERS,
        "raw_model_output": "",
        "layout_json": {},
        "errors": [],
    }

    final_state = app.invoke(state)

    if final_state["errors"]:
        print("Errors:", final_state["errors"])
        print("\nRAW OUTPUT:\n", final_state["raw_model_output"])
        Path("raw_output_debug.txt").write_text(final_state["raw_model_output"], encoding="utf-8")
        print("\nSaved raw output to raw_output_debug.txt")
        return

    layout_json = final_state["layout_json"]
    Path(OUTPUT_JSON_PATH).write_text(json.dumps(layout_json, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()