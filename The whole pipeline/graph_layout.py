from __future__ import annotations

import json
from typing import Any, Dict, List, TypedDict, Optional

from langgraph.graph import StateGraph, END

from llama_cpp import Llama

from tools_designer_opt import designer_opt_generate_image


# ----------------------------
# State
# ----------------------------
class LayoutState(TypedDict):
    # Inputs
    input_data: Dict[str, Any]
    system_message: str
    user_message: str

    # LLM config
    model_path: str
    max_tokens: int
    n_gpu_layers: int

    # Outputs / intermediate
    raw_model_output: str
    layout_json: Dict[str, Any]
    errors: List[str]


# ----------------------------
# Helpers
# ----------------------------
def extract_json(text: str) -> Dict[str, Any]:
    """
    Extract the first JSON object from model output.
    Keeps behavior simple and strict; replace with your existing extract_json if preferred.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    return json.loads(text[start:end + 1])


def find_generated_images(layout_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Returns a list of dicts containing slide/element indices and prompts for any
    element that looks like an image request:
      - type == "image"
      - source == "generate"
      - image_prompt exists and non-empty
    """
    reqs: List[Dict[str, Any]] = []
    pres = layout_json.get("presentation", {})
    slides = pres.get("slides", [])
    for s_idx, slide in enumerate(slides):
        for e_idx, el in enumerate(slide.get("elements", [])):
            if (
                el.get("type") == "image"
                and el.get("source") == "generate"
                and isinstance(el.get("image_prompt"), str)
                and el["image_prompt"].strip()
            ):
                reqs.append(
                    {
                        "slide_index": s_idx,
                        "element_index": e_idx,
                        "prompt": el["image_prompt"].strip(),
                        "width": int(el.get("width", 1024)) if isinstance(el.get("width"), (int, float, str)) else 1024,
                        "height": int(el.get("height", 1024)) if isinstance(el.get("height"), (int, float, str)) else 1024,
                    }
                )
    return reqs


def patch_image_path(layout_json: Dict[str, Any], slide_index: int, element_index: int, path: str) -> None:
    el = layout_json["presentation"]["slides"][slide_index]["elements"][element_index]
    el["source"] = "file"
    el["path"] = path


# ----------------------------
# Nodes
# ----------------------------
def node_generate_layout(state: LayoutState) -> LayoutState:
    """
    Mirror the exact behavior of Designer_brain.py:
    - Qwen chat template
    - stream=True
    - stop=["<|im_end|>", "<|endoftext|>"]
    - n_ctx=8192, n_threads=8
    """
    try:
        llm = Llama(
            model_path=state["model_path"],
            n_ctx=  20480 ,
            n_gpu_layers=state["n_gpu_layers"],
            n_threads=8,
            verbose=False,
        )

        prompt = f"""<|im_start|>system
{state["system_message"]}<|im_end|>
<|im_start|>user
{state["user_message"]}<|im_end|>
<|im_start|>assistant
"""

        full_response = ""
        token_count = 0

        for output in llm(
            prompt,
            max_tokens=state["max_tokens"],
            temperature=0.1,
            top_p=0.9,
            echo=False,
            stream=True,
            stop=["<|im_end|>", "<|endoftext|>"],
        ):
            chunk = output["choices"][0]["text"]
            full_response += chunk
            token_count += 1

        state["raw_model_output"] = full_response

    except Exception as e:
        state["errors"].append(f"generate_layout: {e}")

    return state


def node_parse_json(state: LayoutState) -> LayoutState:
    try:
        state["layout_json"] = extract_json(state["raw_model_output"])
    except Exception as e:
        state["errors"].append(f"parse_json: {e}")
    return state


def node_materialize_images(state: LayoutState) -> LayoutState:
    """
    For each image element with source=generate, call designer_opt tool and patch JSON.
    This uses LangChain tool invocation so it is a real 'tool call' in code terms.
    """
    try:
        reqs = find_generated_images(state["layout_json"])
        for r in reqs:
            tool_args = {
                "prompt": r["prompt"],
                "width": r.get("width", 1024),
                "height": r.get("height", 1024),
            }
            img_path = designer_opt_generate_image.invoke(tool_args)
            patch_image_path(state["layout_json"], r["slide_index"], r["element_index"], str(img_path))
    except Exception as e:
        state["errors"].append(f"materialize_images: {e}")
    return state


def route_after_parse(state: LayoutState) -> str:
    if state["errors"]:
        return "end"
    if find_generated_images(state["layout_json"]):
        return "do_images"
    return "end"


# ----------------------------
# Graph
# ----------------------------
graph = StateGraph(LayoutState)

graph.add_node("generate_layout", node_generate_layout)
graph.add_node("parse_json", node_parse_json)
graph.add_node("materialize_images", node_materialize_images)

graph.set_entry_point("generate_layout")
graph.add_edge("generate_layout", "parse_json")

graph.add_conditional_edges(
    "parse_json",
    route_after_parse,
    {"do_images": "materialize_images", "end": END},
)

graph.add_edge("materialize_images", END)

app = graph.compile()
