# main.py
from workflow import build_graph
from workflow import AgentState

def main():
    print("=== Agentic AI Research Assistant ===\n")
    
   
    user_topic = input("Enter the topic you want the agent to research: ").strip()
    if not user_topic:
        print("No topic entered. Exiting...")
        return
   
    workflow_app = build_graph()
    initial_state: AgentState = {
        "messages": [],
        "user_input": user_topic
    }
    
    print("\n--- Running the workflow ---\n")
    final_state = workflow_app.invoke(initial_state)
    
    # 3️⃣ عرض النتائج
    print("\n--- Final Workflow State ---\n")
    for i, msg in enumerate(final_state["messages"], 1):
        print(f"[{i}] {msg}\n")

if __name__ == "__main__":
    main()
