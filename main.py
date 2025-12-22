# import json
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
# import time

# # ----------------------------
# # CONFIG
# # ----------------------------
# MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
# INPUT_JSON_PATH = "Test1.json"
# MAX_NEW_TOKENS = 4000

# print("=" * 60)
# print("STARTING MODEL INFERENCE")
# print("=" * 60)

# # ----------------------------
# # LOAD MODEL & TOKENIZER
# # ----------------------------
# print("\n[1/5] Loading tokenizer...")
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# print("✓ Tokenizer loaded")

# print("\n[2/5] Loading model (this may take a while)...")
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     torch_dtype=torch.float16,
#     device_map="auto"
# )
# print("✓ Model loaded")

# # ----------------------------
# # LOAD INPUT JSON
# # ----------------------------
# print("\n[3/5] Loading input JSON...")
# with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
#     input_data = json.load(f)
# print(f"✓ Loaded {len(input_data.get('slides', []))} slides")

# # ----------------------------
# # SYSTEM + USER MESSAGES
# # ----------------------------
# system_message = """
# You are a PowerPoint design agent that generates complete slide specifications.

# Your task:
# - Convert slide content into detailed PowerPoint-ready JSON
# - Include ALL design specifications: fonts, sizes, colors, positions, dimensions
# - Specify exact layout coordinates and styling for each element
# - Output VALID JSON ONLY - no explanations, no markdown

# Design Guidelines:
# - Slide dimensions: 10 inches width × 7.5 inches height (standard PowerPoint)
# - Use professional fonts: Arial, Calibri, or Times New Roman
# - Title font size: 32-44pt
# - Subtitle font size: 20-28pt
# - Body text font size: 14-18pt
# - Use consistent color schemes
# - Position elements with x, y coordinates (in inches from top-left)
# - Specify width and height for all elements
# """

# user_message = f"""
# Input presentation content:
# {json.dumps(input_data, indent=2)}

# Generate a PowerPoint-ready JSON with this EXACT structure for each slide:

# {{
#   "presentation": {{
#     "title": "Presentation Title",
#     "dimensions": {{"width": 10, "height": 7.5, "unit": "inches"}},
#     "theme": {{
#       "primary_color": "#1F4E78",
#       "secondary_color": "#FFFFFF",
#       "accent_color": "#FFC000",
#       "background_color": "#FFFFFF"
#     }},
#     "slides": [
#       {{
#         "slide_number": 1,
#         "layout_type": "title_slide | content | two_column | image_with_text | bullet_list",
#         "background": {{
#           "color": "#FFFFFF",
#           "image": null
#         }},
#         "elements": [
#           {{
#             "type": "text",
#             "role": "title | subtitle | body | bullet_point",
#             "content": "Text content here",
#             "position": {{"x": 0.5, "y": 1.0, "unit": "inches"}},
#             "size": {{"width": 9.0, "height": 1.5, "unit": "inches"}},
#             "font": {{
#               "family": "Arial",
#               "size": 44,
#               "bold": true,
#               "italic": false,
#               "color": "#1F4E78",
#               "alignment": "center | left | right"
#             }},
#             "z_index": 1
#           }},
#           {{
#             "type": "image",
#             "content": null,
#             "position": {{"x": 5.5, "y": 2.0, "unit": "inches"}},
#             "size": {{"width": 4.0, "height": 3.0, "unit": "inches"}},
#             "source": "generate",
#             "image_prompt": "A professional image showing...",
#             "border": {{"width": 1, "color": "#CCCCCC"}},
#             "z_index": 2
#           }},
#           {{
#             "type": "shape",
#             "shape_type": "rectangle | circle | line",
#             "position": {{"x": 0.5, "y": 6.5, "unit": "inches"}},
#             "size": {{"width": 9.0, "height": 0.1, "unit": "inches"}},
#             "fill_color": "#1F4E78",
#             "border": {{"width": 0, "color": null}},
#             "z_index": 0
#           }}
#         ]
#       }}
#     ]
#   }}
# }}

# IMPORTANT:
# - Include position (x, y) and size (width, height) for EVERY element
# - Use consistent color scheme throughout
# - Z-index determines layering (0=back, higher=front)
# - For bullet lists, create separate text elements for each bullet
# - Ensure no elements overlap unless intentional
# - Leave margins: 0.5 inches from edges
# """

# messages = [
#     {"role": "system", "content": system_message},
#     {"role": "user", "content": user_message}
# ]

# # ----------------------------
# # APPLY CHAT TEMPLATE
# # ----------------------------
# print("\n[4/5] Preparing prompt...")
# prompt = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True
# )

# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
# input_token_count = inputs["input_ids"].shape[1]

# print(f"✓ Prompt prepared: {input_token_count} tokens")
# print(f"  Max new tokens to generate: {MAX_NEW_TOKENS}")
# print(f"  Total max tokens: {input_token_count + MAX_NEW_TOKENS}")

# # ----------------------------
# # GENERATE WITH PROGRESS
# # ----------------------------
# print("\n[5/5] Generating response...")
# print("=" * 60)
# print("LIVE OUTPUT (streaming):")
# print("-" * 60)

# # Create a custom streamer for real-time output
# streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# start_time = time.time()

# with torch.no_grad():
#     output = model.generate(
#         **inputs,
#         max_new_tokens=MAX_NEW_TOKENS,
#         do_sample=False,
#         streamer=streamer,
#         pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
#     )

# generation_time = time.time() - start_time

# print("-" * 60)
# print(f"✓ Generation complete in {generation_time:.2f}s")

# # ----------------------------
# # DECODE FULL OUTPUT
# # ----------------------------
# input_len = inputs["input_ids"].shape[1]
# generated_tokens = output[0][input_len:]
# tokens_generated = len(generated_tokens)

# response_text = tokenizer.decode(
#     generated_tokens,
#     skip_special_tokens=True
# )

# print(f"\n📊 Generation Stats:")
# print(f"   - Tokens generated: {tokens_generated}")
# print(f"   - Tokens/second: {tokens_generated/generation_time:.2f}")
# print(f"   - Total time: {generation_time:.2f}s")

# # ----------------------------
# # PARSE JSON
# # ----------------------------
# print("\n" + "=" * 60)
# print("PARSING OUTPUT")
# print("=" * 60)

# def extract_json(text):
#     start = text.find("{")
#     end = text.rfind("}") + 1
#     if start == -1 or end == -1:
#         raise ValueError("No JSON object found in model output")
#     return json.loads(text[start:end])

# try:
#     layout_json = extract_json(response_text)
#     print("✓ Successfully parsed JSON")
    
#     # Validate structure
#     if "presentation" in layout_json:
#         slides = layout_json["presentation"].get("slides", [])
#         print(f"✓ Found {len(slides)} slides in output")
        
#         # Validate each slide has required design specs
#         for i, slide in enumerate(slides):
#             elements = slide.get("elements", [])
#             print(f"  Slide {i+1}: {len(elements)} elements")
    
#     # Save to file
#     output_file = "powerpoint_layout.json"
#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(layout_json, f, indent=2)
#     print(f"\n✓ Saved to {output_file}")
    
#     print("\n" + "=" * 60)
#     print("FINAL POWERPOINT LAYOUT JSON")
#     print("=" * 60)
#     print(json.dumps(layout_json, indent=2))
    
# except Exception as e:
#     print(f"✗ Error parsing JSON: {e}")
#     print("\nRAW OUTPUT:")
#     print(response_text)
    
#     # Save raw output for debugging
#     with open("raw_output.txt", "w", encoding="utf-8") as f:
#         f.write(response_text)
#     print("\n✓ Raw output saved to raw_output.txt")


import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import time

# ----------------------------
# CONFIG
# ----------------------------
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
INPUT_JSON_PATH = "slides.json"
OUTPUT_JSON_PATH = "powerpoint_layout.json"
MAX_NEW_TOKENS = 4096*2

print("=" * 60)
print("STARTING MODEL INFERENCE")
print("=" * 60)

# ----------------------------
# LOAD MODEL & TOKENIZER
# ----------------------------
print("\n[1/5] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("✓ Tokenizer loaded")

print("\n[2/5] Loading model (this may take a while)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("✓ Model loaded")

# ----------------------------
# LOAD INPUT JSON
# ----------------------------
print("\n[3/5] Loading input JSON...")
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

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_message}
]

# ----------------------------
# APPLY CHAT TEMPLATE
# ----------------------------
print("\n[4/5] Preparing prompt...")
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
input_token_count = inputs["input_ids"].shape[1]

print(f"✓ Prompt prepared: {input_token_count} tokens")
print(f"  Max new tokens to generate: {MAX_NEW_TOKENS}")
print(f"  Total max tokens: {input_token_count + MAX_NEW_TOKENS}")

# ----------------------------
# GENERATE WITH PROGRESS
# ----------------------------
print("\n[5/5] Generating response...")
print("=" * 60)
print("LIVE OUTPUT (streaming):")
print("-" * 60)

# Create a custom streamer for real-time output
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

start_time = time.time()

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        streamer=streamer,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
    )

generation_time = time.time() - start_time

print("-" * 60)
print(f"✓ Generation complete in {generation_time:.2f}s")

# ----------------------------
# DECODE FULL OUTPUT
# ----------------------------
input_len = inputs["input_ids"].shape[1]
generated_tokens = output[0][input_len:]
tokens_generated = len(generated_tokens)

response_text = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
)

print(f"\n📊 Generation Stats:")
print(f"   - Tokens generated: {tokens_generated}")
print(f"   - Tokens/second: {tokens_generated/generation_time:.2f}")
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
    layout_json = extract_json(response_text)
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
    print(response_text)
    
    # Save raw output for debugging (as txt)
    debug_file = "raw_output_debug.txt"
    with open(debug_file, "w", encoding="utf-8") as f:
        f.write(response_text)
    print(f"\n✓ Raw output saved to {debug_file} for debugging")