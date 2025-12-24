import json
import time
from llama_cpp import Llama

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = "/teamspace/studios/this_studio/Designer/models/qwen2.5-3b-instruct-q8_0.gguf"  # Update this path
INPUT_JSON_PATH = "noha.json"
OUTPUT_JSON_PATH = "powerpoint_layout_10.json"
MAX_TOKENS = 4096*2

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
    n_ctx=4096*2,          # Context window
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

system_message = """
You are a deterministic PowerPoint Layout Generator Agent.

OUTPUT RULES (CRITICAL):
- Output STRICT JSON only. No markdown, no commentary.
- Do NOT generate or paraphrase slide content.
- All text must be placeholders only, and must appear ONLY inside shapes as a "text" field.

PLACEHOLDERS (ONLY):
- {{TITLE}}
- {{CONTENT}}

SLIDE RULES (CRITICAL):
- input_data.slides is an array.
- Generate EXACTLY one output slide per input slide.
- Preserve order.
- slide_number starts at 1 and increments sequentially.
- Do NOT merge, drop, or add slides.

ANALYSIS & IMAGE RULE (CRITICAL):
- Carefully analyze each input slide's structure/metadata to decide the best layout_type.
- IMAGE EVALUATION: For each slide, check if an image adds value. 
- If "layout_type" is 'bullet' or 'paragraph', prioritize a split-layout (text on one side, image on the other).
- If an image is included, you MUST adjust the size and position of the "content" shape to ensure it does not overlap with the image.
- If an image is not clearly beneficial for the specific structure, OMIT the image element.

ELEMENT & INDEXING RULES (CRITICAL):
- Each slide MUST have exactly 2 shapes: 1 title shape and 1 content shape.
- UNIQUE IDs: Every element "id" MUST be unique across the entire presentation. 
- You MUST append the current slide_number to every ID (e.g., "title_shape_1", "content_shape_1", "image_1", then "title_shape_2", etc.).

DIFFUSION PROMPT (CRITICAL):

INTENT:
- image_prompt must describe a visually rich SCENE, not a concept label.
- Think like a visual art director: describe what is visible, how it is composed, and how it feels.

STRUCTURE (REQUIRED):
Each image_prompt MUST include, in natural language:
1. Subject / scene description (abstract or realistic, but visual)
2. Composition & camera (e.g., wide shot, isometric, close-up, depth)
3. Style (e.g., minimalist illustration, cinematic photo, 3D render, flat design)
4. Lighting & color mood (e.g., soft light, high contrast, muted palette)
5. Quality modifiers (e.g., high detail, professional, clean)
6. Negative constraints (MANDATORY phrase at the end):
   "no text, no logos, no labels, no watermark, no symbols, no icons"

STRICT RULES:
- image_prompt must be diffusion-ready prose (NOT a title, keyword list, or diagram name).
- NEVER include slide content, brands, proper nouns, numbers, charts, or readable symbols.
- Prompts MUST vary semantically across slides (scene, style, or composition must change).
- Prefer abstract or symbolic visuals over literal diagrams.
- Avoid instructional visuals (no arrows, no callouts, no UI elements).

LAYOUT AWARENESS:
- Image composition must match placement:
  - Side image → balanced negative space toward text side
  - Smaller image → focused subject, low clutter
- NEVER assume full-slide usage.


CANVAS:
- 13.33 × 7.5 inches.
- Use numeric coordinates (in inches) for positions and sizes, but choose values yourself.
- Do not specify font sizes, font families, or exact color codes unless strongly necessary.
"""



user_message = """
Input Data (structure only, content is external):
{input_data}

TASK:
Generate ONLY the layout JSON for all slides.

OUTPUT FORMAT (STRICT):

{
  "presentation": {
    "dimensions": { "width": 13.33, "height": 7.5, "unit": "inches" },
    "slides": [
      {
        "slide_number": 1,
        "source_slide_index": 0,
        "layout_type": "title | bullet | paragraph | comparison",
        "elements": [
          {
            "id": "title_shape_1",
            "type": "shape",
            "role": "title",
            "shape_type": "rectangle | rounded_rectangle | other",
            "text": "{{TITLE}}",
            "position": { "x": 0, "y": 0, "unit": "inches" },
            "size": { "width": 0, "height": 0, "unit": "inches" }
          },
          {
            "id": "content_shape_1",
            "type": "shape",
            "role": "content",
            "shape_type": "rectangle | rounded_rectangle | other",
            "text": "{{CONTENT}}",
            "position": { "x": 0, "y": 0, "unit": "inches" },
            "size": { "width": 0, "height": 0, "unit": "inches" }
          }

          // OPTIONAL: include this element ONLY if needed; otherwise omit it entirely
          ,
          {
            "id": "image_1",
            "type": "image",
            "source": "generate",
            "image_prompt": "diffusion-ready prompt fit the slide content ...",
            "position": { "x": 0, "y": 0, "unit": "inches" },
            "size": { "width": 0, "height": 0, "unit": "inches" }
          }
        ]
      }
    ]
  }
}


NOTES (STRICT):
- "elements" must contain EXACTLY 2 shapes and 0 or 1 image.
- Do NOT include any "text" elements anywhere.
- If no image is needed for a slide, OMIT the image element entirely from the "elements" array.
- If included, only ONE image per slide and it must fit the reserved space (do NOT default to full-slide).
- Replace all 0 values with appropriate numeric coordinates/sizes.
- image_prompt must be diffusion-ready, varied, and aligned to placement. Include: composition, style, lighting, mood, quality, "no text".
"""


# ----------------------------
# BUILD PROMPT (FIX: Uncomment this line!)
# ----------------------------
print("\n[3/4] Preparing prompt...")

# Format the user message with actual input data - use simple replacement to avoid brace escaping issues
user_message_formatted = user_message.replace(
    "{input_data}",
    json.dumps(input_data, indent=2)
)

# Qwen2.5 chat template format
prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message_formatted}<|im_end|>
<|im_start|>assistant
"""

print(f"✓ Prompt prepared")
print(f"  Max tokens to generate: {MAX_TOKENS}")
print(f"  Input slides to process: {len(input_data.get('slides', []))}")

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
        input_slides = len(input_data.get('slides', []))
        output_slides = len(slides)
        
        print(f"✓ Input slides: {input_slides}")
        print(f"✓ Output slides: {output_slides}")
        
        if input_slides == output_slides:
            print("✓ Slide count matches!")
        else:
            print(f"⚠ WARNING: Slide count mismatch! Expected {input_slides}, got {output_slides}")
        
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