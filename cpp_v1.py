import json
import time
from llama_cpp import Llama

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = "/teamspace/studios/this_studio/Designer/models/qwen2.5-3b-instruct-q8_0.gguf"  # Update this path
INPUT_JSON_PATH = "Test1.json"
OUTPUT_JSON_PATH = "powerpoint_layout.json"
MAX_TOKENS = 2000

# GPU layers - T4 has 16GB VRAM, safe to use all layers
N_GPU_LAYERS = 35  # Qwen 2.5-3B has ~26-35 layers, this ensures all on GPU

print("=" * 60)
print("STARTING LLAMA.CPP INFERENCE")
print("=" * 60)

# ----------------------------
# LOAD MODEL
# ----------------------------
print("\n[1/4] Loading model...")
print(f"Model path: {MODEL_PATH}")
print(f"GPU layers: {N_GPU_LAYERS}")

# Check if GPU support is available
try:
    import llama_cpp
    print(f"llama-cpp-python version: {llama_cpp.__version__}")
except:
    pass

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,          # Context window
    n_gpu_layers=N_GPU_LAYERS,
    n_threads=8,         # CPU threads
    verbose=True         # Show detailed loading info
)
print("✓ Model loaded successfully")
print(f"  Backend: {'GPU (CUDA)' if N_GPU_LAYERS > 0 else 'CPU'}")
print(f"  GPU layers loaded: {llm.n_gpu_layers if hasattr(llm, 'n_gpu_layers') else 'N/A'}")

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
You are a PowerPoint Slides designer agent that generates complete slide Layout and the colours of all it's elements.

Your task:
- Based on slides content Generate PowerPointLayout-ready as JSON
- Include ALL design specifications: fonts, sizes, colors, positions, dimensions
- Specify exact layout coordinates and styling for each element
- Output VALID JSON ONLY - no explanations, no markdown

Design Guidelines:
- Slide dimensions: 13 inches width × 7.5 inches height (standard PowerPoint)
- Use professional fonts: Arial, Calibri, or Times New Roman
- Use consistent color schemes
- Position elements with x, y coordinates (in inches from top-left)
- Specify each slide number 
- Specify each element Text Size
- Specify width and height for all elements
- Specify color as hex like #1F4E78
- Specify the layout_type type like Title slide,Bullet point slide,Comparison slide,Paragraph slide
- Specify the role type like Title slide,Bullet point slide,Comparison slide,Paragraph slide

"""

user_message = f"""
Input presentation content:
{json.dumps(input_data, indent=2)}

Generate a PowerPoint-ready JSON with this EXACT structure for each slide:

{{
  "presentation": {{
    "title": "Presentation Title",
    "dimensions": {{"width": , "height": , "unit": "inches"}},
    "theme": {{
      "primary_color": "",
      "secondary_color": "",
      "accent_color": "",
      "background_color": ""
    }},
    "slides": [
      {{
        "slide_number": ,
        "layout_type": "",
        "background": {{
          "color": "",
          "image": null
        }},
        "elements": [
          {{
            "type": "text",
            "role": "",
            "position": {{"x":, "y": , "unit": "inches"}},
            "size": {{"width": , "height": , "unit": "inches"}},
            "font": {{
              "family": "",
              "size": ,
              "bold": ,
              "italic": ,
              "color": "",
              "alignment": "center | left | right"
            }},
            "z_index": 1
          }},
          {{
            "type": "image",
            "content": null,
            "position": {{"x": , "y": , "unit": "inches"}},
            "size": {{"width": , "height": , "unit": "inches"}},
            "source": "generate",
            "image_prompt": "A professional image showing...",
            "z_index": 
          }},
          {{
            "type": "shape",
            "shape_type": "rectangle | circle | line",
            "position": {{"x": , "y": , "unit": "inches"}},
            "size": {{"width": , "height": , "unit": "inches"}},
            "fill_color": "",
            "border": {{"width": , "color": }},
            "z_index": 
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