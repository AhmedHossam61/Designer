"""
Quick Test Script for Unified Workflow
=======================================
This script provides hardcoded examples to test the workflow without
needing to input topics manually.
"""

from unified_workflow import build_unified_graph, UnifiedState
import json
from pathlib import Path


def test_simple_topic():
    """Test with a simple topic - no image generation"""
    print("\n" + "="*70)
    print("TEST 1: Simple Topic (No Images)")
    print("="*70)
    
    workflow = build_unified_graph()
    
    state: UnifiedState = {
        "user_topic": "Explain photosynthesis in simple terms for 5 slides",
        "research_messages": [],
        "content_json_path": "",
        "layout_json_path": "",
        "generate_images": False,
        "errors": []
    }
    
    result = workflow.invoke(state)
    
    print_results(result)
    return result


def test_with_images():
    """Test with image generation enabled"""
    print("\n" + "="*70)
    print("TEST 2: Topic with Image Generation")
    print("="*70)
    
    workflow = build_unified_graph()
    
    state: UnifiedState = {
        "user_topic": "Solar system planets explained in 5 slides",
        "research_messages": [],
        "content_json_path": "",
        "layout_json_path": "",
        "generate_images": True,
        "errors": []
    }
    
    result = workflow.invoke(state)
    
    print_results(result)
    return result


def test_with_existing_content():
    """Test designer node directly with existing content JSON"""
    print("\n" + "="*70)
    print("TEST 3: Using Existing Content JSON")
    print("="*70)
    
    # Check if noha.json exists
    if not Path("noha.json").exists():
        print("❌ noha.json not found. Skipping this test.")
        return None
    
    # We'll manually invoke just the designer node
    from unified_workflow import designer_node
    
    state: UnifiedState = {
        "user_topic": "N/A",
        "research_messages": ["Using existing noha.json"],
        "content_json_path": "noha.json",
        "layout_json_path": "",
        "generate_images": True,
        "errors": []
    }
    
    result = designer_node(state)
    
    print_results(result)
    return result


def test_error_handling():
    """Test error handling with invalid input"""
    print("\n" + "="*70)
    print("TEST 4: Error Handling")
    print("="*70)
    
    workflow = build_unified_graph()
    
    state: UnifiedState = {
        "user_topic": "",  # Empty topic should cause error
        "research_messages": [],
        "content_json_path": "",
        "layout_json_path": "",
        "generate_images": False,
        "errors": []
    }
    
    result = workflow.invoke(state)
    
    print_results(result)
    return result


def print_results(state: UnifiedState):
    """Pretty print the workflow results"""
    print("\n" + "-"*70)
    print("RESULTS")
    print("-"*70)
    
    if state["errors"]:
        print("\n❌ Errors:")
        for err in state["errors"]:
            print(f"   {err}")
    else:
        print("\n✅ Success!")
    
    print(f"\n📊 State:")
    print(f"   Topic: {state['user_topic']}")
    print(f"   Content JSON: {state['content_json_path'] or 'Not generated'}")
    print(f"   Layout JSON: {state['layout_json_path'] or 'Not generated'}")
    print(f"   Images enabled: {state['generate_images']}")
    
    if state["research_messages"]:
        print(f"\n💬 Messages:")
        for msg in state["research_messages"]:
            print(f"   - {msg}")
    
    # Show file contents if they exist
    if state["content_json_path"] and Path(state["content_json_path"]).exists():
        print(f"\n📄 Content JSON Preview:")
        content = json.loads(Path(state["content_json_path"]).read_text())
        print(f"   Slides: {len(content.get('slides', []))}")
        if content.get('slides'):
            first_slide = content['slides'][0]
            print(f"   First slide type: {first_slide.get('slide_type')}")
            print(f"   First slide title: {first_slide.get('slide_title', '')[:50]}...")
    
    if state["layout_json_path"] and Path(state["layout_json_path"]).exists():
        print(f"\n🎨 Layout JSON Preview:")
        layout = json.loads(Path(state["layout_json_path"]).read_text())
        slides = layout.get("presentation", {}).get("slides", [])
        print(f"   Slides: {len(slides)}")
        if slides:
            elements = slides[0].get("elements", [])
            print(f"   First slide elements: {len(elements)}")
            image_count = sum(1 for s in slides for e in s.get("elements", []) if e.get("type") == "image")
            print(f"   Total images: {image_count}")


def run_all_tests():
    """Run all test cases"""
    print("\n" + "="*70)
    print("RUNNING ALL TESTS")
    print("="*70)
    
    tests = [
        ("Simple Topic", test_simple_topic),
        ("With Images", test_with_images),
        ("Existing Content", test_with_existing_content),
        ("Error Handling", test_error_handling)
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            print(f"\n{'='*70}")
            print(f"Running: {name}")
            print(f"{'='*70}")
            result = test_func()
            results[name] = {
                "status": "✅ PASSED" if result and not result.get("errors") else "❌ FAILED",
                "result": result
            }
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results[name] = {
                "status": "💥 CRASHED",
                "result": None
            }
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, test_result in results.items():
        print(f"{test_result['status']} - {test_name}")
    
    passed = sum(1 for r in results.values() if "PASSED" in r["status"])
    total = len(results)
    
    print(f"\n{passed}/{total} tests passed")


def interactive_menu():
    """Interactive test menu"""
    while True:
        print("\n" + "="*70)
        print("UNIFIED WORKFLOW - TEST MENU")
        print("="*70)
        print("\n1. Test simple topic (no images)")
        print("2. Test with images")
        print("3. Test with existing content JSON")
        print("4. Test error handling")
        print("5. Run all tests")
        print("6. Exit")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == "1":
            test_simple_topic()
        elif choice == "2":
            test_with_images()
        elif choice == "3":
            test_with_existing_content()
        elif choice == "4":
            test_error_handling()
        elif choice == "5":
            run_all_tests()
        elif choice == "6":
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        run_all_tests()
    else:
        interactive_menu()
