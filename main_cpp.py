import json
import time
from llama_cpp import Llama

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = "/teamspace/studios/this_studio/Designer/models/qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf"  # Update this path
INPUT_JSON_PATH = "Test1.json"
OUTPUT_JSON_PATH = "powerpoint_layout.json"
MAX_TOKENS = 4096*2*2

# GPU layers - adjust based on your VRAM (0 = CPU only, -1 = all layers on GPU)
N_GPU_LAYERS = -1  # Use -1 for full GPU, 0 for CPU only, or specific number like 20

print("=" * 60)
print("STARTING LLAMA.CPP INFERENCE")
print("=" * 60)

# ----------------------------
# LOAD MODEL
# ----------------------------
print("\n[1/4] Loading model...")
print(f"Model path: {MODEL_PATH}")
print(f"GPU layers: {N_GPU_LAYERS}")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,          # Context window
    n_gpu_layers=N_GPU_LAYERS,
    n_threads=8,         # CPU threads
    verbose=False
)
print("✓ Model loaded successfully")

# ----------------------------
# LOAD INPUT JSON
# ----------------------------
print("\n[2/4] Loading input JSON...")
with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
    input_data = json.load(f)
print(f"✓ Loaded {len(input_data.get('slides', []))} slides")

# ----------------------------
# SYSTEM + USER MESSAGES
# ----------------------------
system_message = """
You are a PowerPoint design agent that generates complete slide specifications.

Your task:
- Convert slide content into detailed PowerPoint-ready JSON
- Include ALL design specifications: fonts, sizes, colors, positions, dimensions
- Specify exact layout coordinates and styling for each element
- Output VALID JSON ONLY - no explanations, no markdown

Design Guidelines:
- Slide dimensions: 10 inches width × 7.5 inches height (standard PowerPoint)
- Use professional fonts: Arial, Calibri, or Times New Roman
- Title font size: 32-44pt
- Subtitle font size: 20-28pt
- Body text font size: 14-18pt
- Use consistent color schemes
- Position elements with x, y coordinates (in inches from top-left)
- Specify width and height for all elements
"""

user_message = f"""
Input presentation content:
{json.dumps(input_data, indent=2)}

Generate a PowerPoint-ready JSON with this EXACT structure for each slide:

{{
  "presentation": {{
    "title": "Presentation Title",
    "dimensions": {{"width": 10, "height": 7.5, "unit": "inches"}},
    "theme": {{
      "primary_color": "#1F4E78",
      "secondary_color": "#FFFFFF",
      "accent_color": "#FFC000",
      "background_color": "#FFFFFF"
    }},
    "slides": [
      {{
        "slide_number": 1,
        "layout_type": "title_slide | content | two_column | image_with_text | bullet_list",
        "background": {{
          "color": "#FFFFFF",
          "image": null
        }},
        "elements": [
          {{
            "type": "text",
            "role": "title | subtitle | body | bullet_point",
            "content": "Text content here",
            "position": {{"x": 0.5, "y": 1.0, "unit": "inches"}},
            "size": {{"width": 9.0, "height": 1.5, "unit": "inches"}},
            "font": {{
              "family": "Arial",
              "size": 44,
              "bold": true,
              "italic": false,
              "color": "#1F4E78",
              "alignment": "center | left | right"
            }},
            "z_index": 1
          }},
          {{
            "type": "image",
            "content": null,
            "position": {{"x": 5.5, "y": 2.0, "unit": "inches"}},
            "size": {{"width": 4.0, "height": 3.0, "unit": "inches"}},
            "source": "generate",
            "image_prompt": "A professional image showing...",
            "border": {{"width": 1, "color": "#CCCCCC"}},
            "z_index": 2
          }},
          {{
            "type": "shape",
            "shape_type": "rectangle | circle | line",
            "position": {{"x": 0.5, "y": 6.5, "unit": "inches"}},
            "size": {{"width": 9.0, "height": 0.1, "unit": "inches"}},
            "fill_color": "#1F4E78",
            "border": {{"width": 0, "color": null}},
            "z_index": 0
          }}
        ]
      }}
    ]
  }}
}}

IMPORTANT:
- Include position (x, y) and size (width, height) for EVERY element
- Use consistent color scheme throughout
- Z-index determines layering (0=back, higher=front)
- For bullet lists, create separate text elements for each bullet
- Ensure no elements overlap unless intentional
- Leave margins: 0.5 inches from edges
"""

# ----------------------------
# BUILD PROMPT
# ----------------------------
print("\n[3/4] Preparing prompt...")

# Qwen2.5 chat template format
prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""

print(f"✓ Prompt prepared")
print(f"  Max tokens to generate: {MAX_TOKENS}")

# ----------------------------
# GENERATE WITH STREAMING
# ----------------------------
print("\n[4/4] Generating response...")
print("=" * 60)
print("LIVE OUTPUT (streaming):")
print("-" * 60)

start_time = time.time()
full_response = ""
token_count = 0

# Stream generation with progress
for output in llm(
    prompt,
    max_tokens=MAX_TOKENS,
    temperature=0.1,
    top_p=0.9,
    echo=False,
    stream=True,
    stop=["<|im_end|>", "<|endoftext|>"]
):
    chunk = output['choices'][0]['text']
    print(chunk, end='', flush=True)
    full_response += chunk
    token_count += 1

generation_time = time.time() - start_time

print("\n" + "-" * 60)
print(f"✓ Generation complete in {generation_time:.2f}s")

# ----------------------------
# STATISTICS
# ----------------------------
print(f"\n📊 Generation Stats:")
print(f"   - Tokens generated: {token_count}")
print(f"   - Tokens/second: {token_count/generation_time:.2f}")
print(f"   - Total time: {generation_time:.2f}s")

# ----------------------------
# PARSE AND SAVE JSON
# ----------------------------
print("\n" + "=" * 60)
print("PARSING OUTPUT")
print("=" * 60)

def extract_json(text):
    # Remove markdown code blocks if present
    text = text.replace("```json", "").replace("```", "")
    
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model output")
    return json.loads(text[start:end])

try:
    layout_json = extract_json(full_response)
    print("✓ Successfully parsed JSON")
    
    # Validate structure
    if "presentation" in layout_json:
        slides = layout_json["presentation"].get("slides", [])
        print(f"✓ Found {len(slides)} slides in output")
        
        # Validate each slide has required design specs
        for i, slide in enumerate(slides):
            elements = slide.get("elements", [])
            print(f"  Slide {i+1}: {len(elements)} elements")
    
    # Save to JSON file
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(layout_json, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Successfully saved to {OUTPUT_JSON_PATH}")
    
    print("\n" + "=" * 60)
    print("PREVIEW OF POWERPOINT LAYOUT JSON")
    print("=" * 60)
    
    # Show a preview (first 50 lines)
    preview = json.dumps(layout_json, indent=2, ensure_ascii=False)
    preview_lines = preview.split('\n')
    if len(preview_lines) > 50:
        print('\n'.join(preview_lines[:50]))
        print(f"\n... ({len(preview_lines) - 50} more lines)")
        print(f"\nFull output saved to: {OUTPUT_JSON_PATH}")
    else:
        print(preview)
    
except Exception as e:
    print(f"✗ Error parsing JSON: {e}")
    print("\nRAW OUTPUT:")
    print(full_response)
    
    # Save raw output for debugging
    debug_file = "raw_output_debug.txt"
    with open(debug_file, "w", encoding="utf-8") as f:
        f.write(full_response)
    print(f"\n✓ Raw output saved to {debug_file} for debugging")