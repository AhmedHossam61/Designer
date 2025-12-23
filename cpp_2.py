import json
import time
from llama_cpp import Llama

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = "/teamspace/studios/this_studio/Designer/models/qwen2.5-3b-instruct-q8_0.gguf"  # Update this path
INPUT_JSON_PATH = "Test1.json"
OUTPUT_JSON_PATH = "powerpoint_layout.json"
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

You MUST output STRICT JSON only.
You MUST NOT generate or paraphrase slide content.
You ONLY generate layout, structure, positions, and placeholders.

CONTENT RULE (CRITICAL):
- DO NOT write real text, summaries, bullets, or sentences.
- ALL text fields MUST be placeholders.
- Valid placeholders ONLY:
  - {{TITLE}}
  - {{SUBTITLE}}
  - {{CONTENT}}
  - {{LEFT_CONTENT}}
  - {{RIGHT_CONTENT}}

If you generate real semantic text, the output is INVALID.

ITERATION RULE (CRITICAL):
- The input JSON contains an array called: input_data.slides
- You MUST generate EXACTLY ONE output slide for EACH input slide
- If input contains N slides, output MUST contain N slides
- Slide order MUST be preserved
- slide_number MUST start from 1 and increment sequentially
- Do NOT merge slides
- Do NOT drop slides
- Do NOT add extra slides

YOUR RESPONSIBILITIES:
1. Choose layout_type for each slide (title, bullet, paragraph, comparison).
2. Place shapes, text containers, and optional images.
3. Explicitly assign which placeholder belongs to which shape or region.
4. Use fixed coordinates on a 13.33 × 7.5 inch canvas.

THEME (FIXED):
- Primary: #2C3E50
- Secondary: #E74C3C
- Background: #FFFFFF

CANVAS:
- Width: 13.33 inches
- Height: 7.5 inches

GLOBAL RULES:
- All Bullet / Paragraph / Comparison slides MUST have:
  - A rectangle header shape at y = 0
  - Title text placed ON TOP of that rectangle
- Every text element MUST declare:
  - content_placeholder
  - assigned_shape_id (or null)

IMAGE RULES:
- Images are OPTIONAL
- If generated, image_prompt MUST be generic and abstract
- NEVER include slide content inside image_prompt

VALID image_prompt example:
"A clean minimal illustration representing a generic technology concept, flat style, blue and white palette."

"""



user_message = """
Input Data (structure only, content is external):
{input_data}

TASK:
Generate ONLY the layout JSON.

STRICT RULES:
1. DO NOT generate real text.
2. Use placeholders ONLY.
3. Generate ONE output slide per input slide.
4. Preserve slide order.
5. Every text element MUST include:
   - content_placeholder
   - assigned_shape_id
6. Image elements are OPTIONAL.
   - If an image is not needed, DO NOT include an image element.
   - If included, image_prompt MUST be generic and abstract.

OUTPUT FORMAT (DO NOT CHANGE STRUCTURE):

{
  "presentation": {
    "title": "{{TITLE}}",
    "dimensions": {
      "width": 13.33,
      "height": 7.5,
      "unit": "inches"
    },
    "slides": [
      {
        "slide_number": 1,
        "source_slide_index": 0,
        "layout_type": "title | bullet | paragraph | comparison",
        "elements": [
          {
            "id": "shape_header_1",
            "type": "shape",
            "shape_type": "rectangle",
            "fill_color": "#2C3E50",
            "position": { "x": 0, "y": 0, "unit": "inches" },
            "size": { "width": 13.33, "height": 1.2, "unit": "inches" },
            "z_index": 0
          },
          {
            "id": "text_title_1",
            "type": "text",
            "role": "title",
            "content_placeholder": "{{TITLE}}",
            "assigned_shape_id": "shape_header_1",
            "position": { "x": 0.5, "y": 0.1, "unit": "inches" },
            "size": { "width": 12, "height": 1, "unit": "inches" },
            "font": {
              "family": "Arial",
              "size": 32,
              "color": "#FFFFFF"
            },
            "z_index": 1
          },
          {
            "id": "text_content_1",
            "type": "text",
            "role": "content",
            "content_placeholder": "{{CONTENT}}",
            "assigned_shape_id": null,
            "position": { "x": 0.5, "y": 1.5, "unit": "inches" },
            "size": { "width": 7, "height": 5, "unit": "inches" },
            "font": {
              "family": "Arial",
              "size": 20,
              "color": "#000000"
            },
            "z_index": 1
          },
          {
            "id": "image_1",
            "type": "image",
            "source": "generate",
            "image_prompt": "A clean minimal illustration representing a generic concept, flat style, blue and white color palette.",
            "position": { "x": 8.0, "y": 1.5, "unit": "inches" },
            "size": { "width": 4.5, "height": 4.5, "unit": "inches" },
            "z_index": 1
          }
        ]
      }
    ]
  }
}
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