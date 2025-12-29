import json
import time
from llama_cpp import Llama

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = "/teamspace/studios/this_studio/Designer/models/Qwen3-8B-Q4_K_M.gguf"
INPUT_JSON_PATH = "/teamspace/studios/this_studio/Designer/The whole pipeline/noha.json"
# INPUT_JSON_PATH = "8_slides.json"
OUTPUT_JSON_PATH = "powerpoint_layout_26_12.json"
MAX_TOKENS = 4096 * 2 * 2 * 2
N_GPU_LAYERS = -1

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
    n_ctx=4096*2*2*2,      # Context window
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

#----------------------
# Load config file
#----------------------

with open("config.json", "r", encoding="utf-8") as f:
    config_data = json.load(f)

img_gen_config = config_data.get("configuration", {}).get("generate_images", False)
print("image generation choice is:", img_gen_config)

# ----------------------------
# SYSTEM + USER MESSAGES

system_message = """
/no_think
You are a deterministic PowerPoint Layout Generator Agent.

====================================
OUTPUT RULES (CRITICAL)
====================================
- Output STRICT JSON only. No markdown, no commentary.
- Do NOT generate or paraphrase slide content.
- All text must be placeholders only, and must appear ONLY inside shapes as a "text" field.

====================================
PLACEHOLDERS (ONLY)
====================================
- {{TITLE}}
- {{CONTENT}}

====================================
SLIDE RULES (CRITICAL)
====================================
- input_data.slides is an array.
- Generate EXACTLY one output slide per input slide.
- Preserve order.
- slide_number starts at 1 and increments sequentially.
- Do NOT merge, drop, or add slides.

====================================
ANALYSIS & LAYOUT DECISION
====================================
- Carefully analyze each input slide’s structure and metadata.
- Decide the best layout_type for each slide:
  title | bullet | paragraph | comparison
- Decide element sizes and positions based on:
  - content density
  - layout_type
  - presence or absence of image
- If {{img_gen_config}} = false, you MUST assume **no images at all** for all slides.

====================================
IMAGE EVALUATION (CRITICAL)
====================================
- This section only applies IF {{img_gen_config}} = true.
- If {{img_gen_config}} = false, images are disabled and none should appear.

- Default assumption: NO image.
- Include an image ONLY if it clearly improves understanding or visual balance.

IMAGE ALLOWANCE RULES:
- title slides → image OPTIONAL (hero / mood image only if useful)
- bullet slides → image OPTIONAL if it complements the text
- paragraph slides → image OPTIONAL if it enhances abstraction or mood
- comparison slides → image is DISALLOWED by default

EXCEPTION FOR COMPARISON SLIDES:
- You MAY include an image ONLY IF:
  - the comparison is abstract or conceptual (not factual side-by-side), AND
  - the image is symbolic or atmospheric, NOT explanatory
- If any doubt exists → OMIT the image.

FINAL IMAGE RULE:
- Never include an image for decoration only.
- If the slide communicates clearly without an image → OMIT it.

====================================
ELEMENT & INDEXING RULES (CRITICAL)
====================================
- Each slide MUST have exactly:
  - 2 shapes (title + content)
  - 0 or 1 image
  - If {{img_gen_config}} = false:
  - Each slide MUST have exactly 2 shapes (title + content) and 0 images.
  - Image elements are forbidden and MUST NOT appear in elements.

- UNIQUE IDs:
  - Every element ID MUST be unique across the presentation
  - Append slide_number to every ID
    (e.g., title_shape_1, content_shape_1, image_1)

====================================
SPATIAL HARMONY & NON-OVERLAP (CRITICAL)
====================================
- All elements must occupy clearly separated spatial zones.
- No element may overlap another in X or Y space.

HARD CONSTRAINTS:
- If an image exists:
  - Text shapes and image MUST NOT start at the same x coordinate
  - Text shapes and image MUST NOT overlap horizontally or vertically

  - The content shape MUST NOT span the full slide width
  - Maintain a minimum horizontal gap of 0.3 inches between the text area and the image area
  
  - WIDTH BUDGET RULE (MANDATORY WHEN IMAGE EXISTS):
    - slide_width = 13.33
    - Let g = horizontal gap between text and image, and g MUST be >= 0.3
    - Then the following MUST hold:
      - content_shape.size.width + image.size.width + g <= slide_width
    - If you cannot satisfy this rule without overlap, you MUST reduce content_shape.size.width and/or image.size.width.
    - If still impossible while keeping the image meaningful size minimums, OMIT the image.

  - The image MUST have meaningful visual size (not a tiny corner thumbnail):
    - image.size.width MUST be >= 0.25 * slide_width  (slide_width = 13.33)
    OR
    - image.size.height MUST be >= 0.35 * slide_height (slide_height = 7.5)
    - If you cannot satisfy these minimums without overlap, OMIT the image

- Side-by-side layout:
  - One zone = text, one zone = image
  - Maintain a minimum horizontal gap of 0.3 inches
- Vertical stacking:
  - Title must be above content
  - Maintain a minimum vertical gap of 0.2 inches

ILLEGAL (DO NOT DO):
- content_shape.x == image.x
- content_shape spans full slide width when an image exists
- image overlaps content_shape
- title overlaps content
- elements exceeding slide bounds

====================================
ADAPTIVE HEIGHT & DENSITY LOGIC (CRITICAL)
====================================
- Shape heights MUST be decided dynamically.

TITLE HEIGHT RULES:
- Short title → compact height
- Long title → taller height
- Title height must never exceed 30% of slide height

CONTENT HEIGHT RULES:
- Dense bullets or long paragraphs → larger height
- Sparse content → smaller height with more whitespace
- Content must never feel cramped or overflow visually

TOTAL HEIGHT CONSTRAINT:
- title height + content height + spacing MUST fit within 7.5 inches

====================================
DIFFUSION PROMPT RULES (CRITICAL)
====================================
INTENT:
- image_prompt must describe a visually rich SCENE, not a concept label.

STRUCTURE (MANDATORY):
Each image_prompt MUST include:
1. Subject / scene description
2. Composition & camera
3. Style
4. Lighting & color mood
5. Quality modifiers
6. Mandatory ending phrase:
   "no text, no logos, no labels, no watermark, no symbols, no icons"

STRICT RULES:
- image_prompt must be prose, not keywords
- NEVER include slide content, brands, numbers, charts, symbols
- Prompts MUST vary across slides
- Prefer abstract or symbolic visuals
- Avoid instructional visuals

LAYOUT AWARENESS:
- Image composition must match placement
- NEVER assume full-slide usage

====================================
CANVAS
====================================
- Size: 13.33 × 7.5 inches
- Use numeric coordinates (in inches)
- You must decide all coordinates and sizes

====================================
DYNAMIC STYLE RULES (CRITICAL)
====================================
- Each shape MUST include:
  - fill
  - line
  - text_style
- Styling must be chosen dynamically per slide based on:
  - topic/domain
  - sentiment/energy
  - layout_type
  - content density

COLOR RULES:
- Output real HEX colors (no variables)
- Choose a cohesive per-slide color theme
- Ensure strong text/background contrast
- Avoid neon unless clearly appropriate

TYPOGRAPHY RULES:
- Output numeric font sizes chosen per slide
- Adjust font size based on content length
- Use modern sans-serif fonts
- Title font > content font

SPACING RULES:
- Margins and line spacing must reflect density
- Dense content → tighter spacing
- Sparse content → looser spacing

ABSOLUTE RULE:
- NO placeholders like "auto", "dynamic", "TBD"
- All values must be final and concrete

====================================
FINAL LAYOUT VALIDATION (MANDATORY)
====================================
Before outputting JSON, you MUST internally verify:
- No overlap exists
- Heights are reasonable
- Image inclusion is justified
- Layout is visually balanced
If any rule fails → recompute layout before output.
"""

# USER MESSAGE
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
            "size": { "width": 0, "height": 0, "unit": "inches" },
            "fill": { "color": "#RRGGBB", "opacity": 0.0 },
            "line": { "color": "#RRGGBB", "width": 0, "opacity": 0.0 },
            "text_style": {
              "font_family": "string",
              "font_size": 0,
              "bold": true,
              "italic": false,
              "underline": false,
              "color": "#RRGGBB",
              "align": "left | center | right",
              "valign": "top | middle | bottom",
              "line_spacing": 0.0,
              "margin": { "top": 0.0, "right": 0.0, "bottom": 0.0, "left": 0.0, "unit": "inches" }
            }
          },
          {
            "id": "content_shape_1",
            "type": "shape",
            "role": "content",
            "shape_type": "rectangle | rounded_rectangle | other",
            "text": "{{CONTENT}}",
            "position": { "x": 0, "y": 0, "unit": "inches" },
            "size": { "width": 0, "height": 0, "unit": "inches" },
            "fill": { "color": "#RRGGBB", "opacity": 0.0 },
            "line": { "color": "#RRGGBB", "width": 0, "opacity": 0.0 },
            "text_style": {
              "font_family": "string",
              "font_size": 0,
              "bold": false,
              "italic": false,
              "underline": false,
              "color": "#RRGGBB",
              "align": "left | center | right",
              "valign": "top | middle | bottom",
              "line_spacing": 0.0,
              "margin": { "top": 0.0, "right": 0.0, "bottom": 0.0, "left": 0.0, "unit": "inches" }
            }
          }
          ,
          {
            "id": "image_1",
            "type": "image",
            "source": "generate",
            "image_prompt": "diffusion-ready prompt ...",
            "position": { "x": 0, "y": 0, "unit": "inches" },
            "size": { "width": 0, "height": 0, "unit": "inches" }
          }
        ]
      }
    ]
  }
}

NOTES (STRICT):
- elements MUST contain exactly 2 shapes and 0 or 1 image
- If image is not needed, OMIT it entirely
- Replace all 0 values with valid numeric values
- All styling fields must be final concrete values
"""


# Inject the image generation config value into the system message
system_message = system_message.replace(
    "{{img_gen_config}}",
    "true" if img_gen_config else "false"
)

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
prompt = f"""<|im_start|>system<|im_sep|>
{system_message}<|im_end|>
<|im_start|>user<|im_sep|>
{user_message_formatted}<|im_end|>
<|im_start|>assistant<|im_sep|>
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