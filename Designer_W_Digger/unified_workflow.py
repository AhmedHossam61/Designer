"""
Unified Workflow: Research Agent + Designer Agent
==================================================
Flow: User Topic → Research → Content JSON → Layout JSON → Images (optional)
"""

import json
from pathlib import Path
from typing import TypedDict, List, Literal
from datetime import datetime

from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END, START

# Import your existing agents
from Agentswithtools import agent_executor as research_agent
from graph_layout import app as designer_app, LayoutState


# ============================================================================
# UNIFIED STATE
# ============================================================================
class UnifiedState(TypedDict):
    """Shared state across both agents"""
    
    # Input
    user_topic: str
    
    # Research Agent outputs
    research_messages: List[str]
    content_json_path: str  # Path to generated content JSON (like noha.json)
    
    # Designer Agent outputs  
    layout_json_path: str   # Path to layout JSON (with placeholders/paths)
    
    # Configuration
    generate_images: bool
    
    # Errors
    errors: List[str]


# ============================================================================
# NODE 1: RESEARCH & CONTENT GENERATION
# ============================================================================
def research_node(state: UnifiedState) -> UnifiedState:
    """
    Use the research agent to generate slide content.
    Saves result to content_json_path.
    """
    print("\n" + "="*60)
    print("NODE 1: RESEARCH & CONTENT GENERATION")
    print("="*60)
    
    try:
        print(f"📚 Researching topic: {state['user_topic']}")
        
        # Run research agent
        result = research_agent.invoke({"input": state["user_topic"]})
        
        # Extract the final output
        # The research agent should return JSON with slides
        output = result.get("output", "")
        
        print(f"\n✅ Research completed. Output length: {len(output)} chars")
        
        # Try to parse the JSON from the output
        # The agent might wrap it in text, so we need to extract it
        try:
            # Try direct parse first
            content_json = json.loads(output)
        except json.JSONDecodeError:
            # Extract JSON from text (find first { to last })
            start = output.find("{")
            end = output.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError("No valid JSON found in research agent output")
            json_str = output[start:end]
            content_json = json.loads(json_str)
        
        # Validate structure
        if "slides" not in content_json:
            raise ValueError("Research output missing 'slides' key")
        
        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        content_path = f"content_{timestamp}.json"
        
        # Save content JSON
        Path(content_path).write_text(
            json.dumps(content_json, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        
        print(f"💾 Content saved to: {content_path}")
        print(f"   Slides generated: {len(content_json.get('slides', []))}")
        
        state["content_json_path"] = content_path
        state["research_messages"].append(f"Research completed: {len(content_json['slides'])} slides")
        
    except Exception as e:
        error_msg = f"Research node error: {str(e)}"
        print(f"❌ {error_msg}")
        state["errors"].append(error_msg)
    
    return state


# ============================================================================
# NODE 2: DESIGNER & LAYOUT GENERATION
# ============================================================================
def designer_node(state: UnifiedState) -> UnifiedState:
    """
    Use the designer agent to generate PowerPoint layout.
    Takes content_json_path as input, outputs layout_json_path.
    """
    print("\n" + "="*60)
    print("NODE 2: DESIGNER & LAYOUT GENERATION")
    print("="*60)
    
    try:
        if not state["content_json_path"]:
            raise ValueError("No content JSON path provided")
        
        content_path = Path(state["content_json_path"])
        if not content_path.exists():
            raise FileNotFoundError(f"Content JSON not found: {content_path}")
        
        print(f"🎨 Designing layout from: {content_path}")
        
        # Load content JSON
        input_data = json.loads(content_path.read_text(encoding="utf-8"))
        
        # Load config for image generation
        config_path = Path("config.json")
        if config_path.exists():
            config_data = json.load(config_path.open())
            img_gen_config = config_data.get("configuration", {}).get("generate_images", False)
        else:
            img_gen_config = state.get("generate_images", False)
        
        print(f"   Image generation: {'ENABLED' if img_gen_config else 'DISABLED'}")
        
        # Load prompts from Designer_brain.py (we'll use the fixed version)
        from Designer_brain_langgraph import _load_prompts_from_designer_brain
        
        system_message, user_message_template = _load_prompts_from_designer_brain(
            "Designer_brain_FIXED.py"
        )
        
        # Inject config
        system_message = system_message.replace(
            "{{img_gen_config}}",
            "true" if img_gen_config else "false"
        )
        
        user_message = user_message_template.replace(
            "{input_data}",
            json.dumps(input_data, indent=2)
        )
        
        # Prepare designer state
        designer_state: LayoutState = {
            "input_data": input_data,
            "system_message": system_message,
            "user_message": user_message,
            "model_path": "/teamspace/studios/this_studio/Designer/models/Qwen3-8B-Q4_K_M.gguf",
            "max_tokens": 4096 * 8,
            "n_gpu_layers": -1,
            "raw_model_output": "",
            "layout_json": {},
            "errors": [],
        }
        
        print("🚀 Running designer workflow...")
        
        # Run designer app
        final_designer_state = designer_app.invoke(designer_state)
        
        if final_designer_state["errors"]:
            for err in final_designer_state["errors"]:
                state["errors"].append(f"Designer: {err}")
            print(f"❌ Designer errors: {final_designer_state['errors']}")
            return state
        
        # Generate timestamped filename for layout
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        layout_path = f"layout_{timestamp}.json"
        
        # Save layout JSON
        layout_json = final_designer_state["layout_json"]
        Path(layout_path).write_text(
            json.dumps(layout_json, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        
        print(f"✅ Layout saved to: {layout_path}")
        
        # Count images that were generated
        slides = layout_json.get("presentation", {}).get("slides", [])
        image_count = sum(
            1 for slide in slides
            for elem in slide.get("elements", [])
            if elem.get("type") == "image" and elem.get("source") == "file"
        )
        
        print(f"   Slides: {len(slides)}")
        print(f"   Images generated: {image_count}")
        
        state["layout_json_path"] = layout_path
        state["research_messages"].append(f"Layout generated: {layout_path}")
        
    except Exception as e:
        error_msg = f"Designer node error: {str(e)}"
        print(f"❌ {error_msg}")
        state["errors"].append(error_msg)
    
    return state


# ============================================================================
# ROUTING LOGIC
# ============================================================================
def should_continue_to_designer(state: UnifiedState) -> Literal["designer", "end"]:
    """Route to designer if research was successful"""
    if state["errors"] or not state["content_json_path"]:
        return "end"
    return "designer"


def check_final_status(state: UnifiedState) -> Literal["end"]:
    """Always end after designer"""
    return "end"


# ============================================================================
# BUILD GRAPH
# ============================================================================
def build_unified_graph() -> StateGraph:
    """
    Build the unified workflow graph:
    START → research → designer → END
    """
    
    graph = StateGraph(UnifiedState)
    
    # Add nodes
    graph.add_node("research", research_node)
    graph.add_node("designer", designer_node)
    
    # Set entry point
    graph.set_entry_point("research")
    
    # Add edges with routing
    graph.add_conditional_edges(
        "research",
        should_continue_to_designer,
        {
            "designer": "designer",
            "end": END
        }
    )
    
    graph.add_edge("designer", END)
    
    return graph.compile()


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("\n" + "="*60)
    print("UNIFIED RESEARCH + DESIGNER WORKFLOW")
    print("="*60)
    
    # Get user input
    user_topic = input("\n📝 Enter research topic: ").strip()
    if not user_topic:
        print("❌ No topic entered. Exiting...")
        return
    
    # Ask about image generation
    gen_images = input("🖼️  Generate images? (y/n): ").strip().lower() == 'y'
    
    # Build workflow
    print("\n🔧 Building workflow graph...")
    workflow = build_unified_graph()
    
    # Initial state
    initial_state: UnifiedState = {
        "user_topic": user_topic,
        "research_messages": [],
        "content_json_path": "",
        "layout_json_path": "",
        "generate_images": gen_images,
        "errors": []
    }
    
    # Run workflow
    print("\n🚀 Starting workflow...\n")
    
    try:
        final_state = workflow.invoke(initial_state)
        
        # Display results
        print("\n" + "="*60)
        print("WORKFLOW COMPLETED")
        print("="*60)
        
        if final_state["errors"]:
            print("\n⚠️  Errors encountered:")
            for err in final_state["errors"]:
                print(f"   - {err}")
        else:
            print("\n✅ Success!")
        
        print(f"\n📄 Generated files:")
        if final_state["content_json_path"]:
            print(f"   Content JSON: {final_state['content_json_path']}")
        if final_state["layout_json_path"]:
            print(f"   Layout JSON:  {final_state['layout_json_path']}")
        
        print("\n💬 Process log:")
        for msg in final_state["research_messages"]:
            print(f"   - {msg}")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Workflow interrupted by user")
    except Exception as e:
        print(f"\n❌ Workflow failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
