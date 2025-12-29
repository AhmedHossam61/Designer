# agent_setup.py
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_classic.agents import Tool
from Tools import duckduckgo_search, youtube_search_tool, fetch_url, extract_pdf_text, extract_docx_text
# from prompt import prompt_template
tools = [
    Tool(
        name="duckduckgo_search",
        func=duckduckgo_search,
        description="Search the web for news, articles, and information"
    ),
    Tool(
        name="youtube_search_tool",
        func=youtube_search_tool,
        description="Search YouTube for videos, tutorials, and visual content"
    ),
    Tool(
        name="fetch_url",
        func=fetch_url,
        description="Fetch full content from a specific URL"
    ),
    Tool(
        name="extract_pdf_text",
        func=extract_pdf_text,
        description=" Extract and read text from PDF files (local path or URL)"
    ),
    Tool(
        name="extract_docx_text",
        func=extract_docx_text,
        description="Extract text from DOCX files (local path or URL)"
    ),
]


# 2. SETUP MODEL (Fix: Temperature and Context)
llm = LlamaCpp(
    model_path="/teamspace/studios/this_studio/Designer/models/Qwen3-8B-Q4_K_M.gguf",
    temperature=0.1,  # FIXED: 9.0 will output pure gibberish. 0.1 is best for following JSON rules.
    max_tokens=4096,
    n_ctx=8192,  # Reduced for stability (40k is very high for local execution unless you have 24GB+ VRAM)
    n_threads=8,  # Adjusted: 64 threads often slows down consumer CPUs (usually physical cores - 2 is best)
    n_gpu_layers=30,  # Offload as much as possible
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    verbose=False,

)

# 3. SETUP PROMPT (Fix: Syntax and Variables)
# LangChain uses single curly braces {var} for variables.
# We need {tools}, {tool_names}, {agent_scratchpad}, and {input}.


template = """
/no think
<|im_start|>system
You are an expert Research Assistant and Presentation Designer. Your goal is to produce a JSON object with presentation slides based on the user input.

────────────────────────────
INPUT ANALYSIS
────────────────────────────
Detect type of user input:

DOCUMENT INPUT (PDF/DOCX):
- Indicators: .pdf, .docx extensions, or phrases like "summarize this document"
- Action: Use extract_pdf_text() or extract_docx_text()
- Goal: Extract content and create summary slides

SEARCH QUERY:
- Indicators: Questions, topics, requests for information
- Keywords: "video", "tutorial" → youtube_search
- Keywords: "article", "news", "information", "explain" → duckduckgo_search
- Goal: Research and create informative slides

────────────────────────────
TOOLS
────────────────────────────
Available tools: {tools}

To use a tool, follow this format:

Thought: Do I need to use a tool? Yes
Action: <tool name> (choose from [{tool_names}])
Action Input: <input for the tool>
Observation: <tool output>

────────────────────────────
TASK
────────────────────────────
1. Research the topic or document content
2. Synthesize exactly 5 slides covering the topic comprehensively
3. Format strictly as valid JSON

Slide types to use (choose appropriately):
- Title slide: Introduction & subtitle
- Bullet point slide: Key points or lists
- Comparison slide: Pros vs. Cons or contrasts
- Paragraph slide: Detailed explanation

────────────────────────────
OUTPUT FORMAT
────────────────────────────
Produce a single JSON object with no extra text:

{{
  "slides": [
    {{
      "slide_type": "Title slide / Bullet point slide / Comparison slide / Paragraph slide",
      "slide_title": "Engaging headline",
      "slide_content": "Body text with \\n for line breaks and **bold** for emphasis"
    }}
  ],
  "summary": "Comprehensive executive summary tying all slides together"
}}

Requirements:
- Tone: Professional, insightful, educational
- Length: ~50-70 words per slide
- Summary must connect all slides
- Ensure JSON is valid
- Follow user's requested slide count

<|im_end|>
<|im_start|>user
Topic: {input}
Thought:{agent_scratchpad}
<|im_end|>
"""


prompt = PromptTemplate(
    input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
    template=template
)


# 4. CREATE AGENT
agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=1
)

# # 5. EXECUTION (Fix: Dictionary Key)
# try:
#     print("--- Starting Agent ---")
#     query = "Agentic AI Explain for non-tech people in 5 slides."

#     # The key here must match the input variable in the PromptTemplate (which is "input")
#     response = agent_executor.invoke({"input": query})

#     print("\n\n--- Final Result ---")
#     print(response)

# except Exception as e:
#     print(f"\nError: {e}")



# # -------- Initialize AgentState --------
# # -------- AgentState --------
# class AgentState(TypedDict):
#     messages: Annotated[Sequence[BaseMessage], "add_messages"]
# agent_state: AgentState = {"messages": []}

# # -------- Execute Query --------
# query = "Agentic AI explained for non-tech people in 5 slides"

# try:
#     print("--- Starting Agent ---")
#     response = agent_executor.invoke({"input": query})
#     # Add agent messages to AgentState
#     if isinstance(response, dict) and "messages" in response:
#         agent_state["messages"].extend(response["messages"])
    
#     print("\n--- Final Result ---")
#     print(response)

# except Exception as e:
#     print(f"Error: {e}")







