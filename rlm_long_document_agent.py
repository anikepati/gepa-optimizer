#!/usr/bin/env python3
"""
RLM-Powered Long Document Analysis Agent
========================================
A small, production-ready style agent that uses Recursive Language Model (RLM)
principles to analyze very long documents (10k–1M+ tokens) without hitting
context window limits or suffering from "context rot".

Built with Google Gemini + tool calling (fully compatible with Google Agent 
Development Kit / Vertex AI Agent Builder patterns).

Key RLM Features Implemented:
- External DocumentStore (full context never in LLM prompt)
- Programmatic exploration via tools (peek, search, recurse)
- Recursive sub-analysis on intelligently chosen chunks
- Stateful aggregation in REPL-like buffers
- Final answer only after thorough exploration

Author: Grok (xAI) — May 2026
Requires: pip install google-generativeai
Get API key: https://aistudio.google.com/app/apikey
"""

import os
import re
import json
import textwrap
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from google.generativeai.types import Tool, FunctionDeclaration

# ============================================================
# 1. CONFIGURATION
# ============================================================
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or "YOUR_GOOGLE_API_KEY_HERE"
MODEL_NAME = "gemini-2.0-flash"          # Fast & capable; swap to gemini-1.5-pro or gemini-2.0-flash-thinking for harder tasks
MAX_SUBCALL_DEPTH = 4                    # Prevent runaway recursion
CHUNK_SIZE = 4000                        # Tokens per recursive chunk (approx)

if GEMINI_API_KEY == "YOUR_GOOGLE_API_KEY_HERE":
    print("⚠️  Please set GOOGLE_API_KEY environment variable or edit the script.")
    print("   Get a free key at: https://aistudio.google.com/app/apikey")

genai.configure(api_key=GEMINI_API_KEY)

# ============================================================
# 2. DOCUMENT STORE (External Context — Core of RLM)
# ============================================================
class DocumentStore:
    """Holds the full document outside the LLM's context window."""
    def __init__(self, text: str, doc_id: str = "long_doc"):
        self.doc_id = doc_id
        self.full_text = text
        self.length = len(text)
        self.chunks: List[str] = self._create_chunks()

    def _create_chunks(self, size: int = CHUNK_SIZE) -> List[str]:
        """Simple overlapping chunks for demo. In production use semantic chunking."""
        chunks = []
        for i in range(0, self.length, size - 200):  # 200 char overlap
            chunk = self.full_text[i:i + size]
            chunks.append(chunk)
        return chunks

    def peek(self, start: int = 0, length: int = 2000) -> str:
        """Peek at a section of the document."""
        return self.full_text[start : start + length]

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Simple keyword + regex search (RLM-style programmatic access)."""
        results = []
        query_lower = query.lower()
        for i, chunk in enumerate(self.chunks):
            if query_lower in chunk.lower():
                # Find best snippet
                match = re.search(re.escape(query), chunk, re.IGNORECASE)
                start = max(0, match.start() - 150) if match else 0
                snippet = chunk[start : start + 400]
                results.append({
                    "chunk_index": i,
                    "snippet": snippet,
                    "relevance": "high" if query_lower in snippet.lower() else "medium"
                })
                if len(results) >= max_results:
                    break
        return results

    def get_chunk(self, index: int) -> str:
        if 0 <= index < len(self.chunks):
            return self.chunks[index]
        return ""

    def stats(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "total_chars": self.length,
            "total_chunks": len(self.chunks),
            "approx_tokens": self.length // 4
        }


# ============================================================
# 3. RLM AGENT (Recursive Reasoning Loop)
# ============================================================
class RLMLongDocAgent:
    """
    RLM-powered agent for long document analysis.
    Uses Gemini tool calling to simulate REPL + recursive sub-calls.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model = genai.GenerativeModel(model_name)
        self.document: Optional[DocumentStore] = None
        self.buffers: Dict[str, str] = {}          # RLM state (like REPL variables)
        self.trace: List[str] = []                 # For transparency

    def load_document(self, text: str, doc_id: str = "user_document"):
        self.document = DocumentStore(text, doc_id)
        self.buffers = {}
        self.trace = []
        print(f"✅ Loaded document: {self.document.stats()}")

    # ---------- TOOL DEFINITIONS (Google-style Function Calling) ----------
    def _get_tools(self) -> List[Tool]:
        """Tools the RLM agent can call — mimics REPL actions."""
        return [
            Tool(
                function_declarations=[
                    FunctionDeclaration(
                        name="peek_document",
                        description="Peek at a specific section of the full document (start char, length). Use for initial exploration.",
                        parameters={
                            "type": "OBJECT",
                            "properties": {
                                "start": {"type": "INTEGER", "description": "Starting character position"},
                                "length": {"type": "INTEGER", "description": "Number of characters to read"}
                            },
                            "required": ["start", "length"]
                        }
                    ),
                    FunctionDeclaration(
                        name="search_document",
                        description="Search the document for keywords or phrases. Returns relevant snippets with chunk indices.",
                        parameters={
                            "type": "OBJECT",
                            "properties": {
                                "query": {"type": "STRING", "description": "Search term or phrase"},
                                "max_results": {"type": "INTEGER", "description": "Max snippets to return"}
                            },
                            "required": ["query"]
                        }
                    ),
                    FunctionDeclaration(
                        name="analyze_chunk",
                        description="Recursively analyze a specific chunk (by index) with a focused sub-question. This is the 'recursive call' in RLM.",
                        parameters={
                            "type": "OBJECT",
                            "properties": {
                                "chunk_index": {"type": "INTEGER", "description": "Index of chunk to analyze"},
                                "sub_query": {"type": "STRING", "description": "Focused question for this chunk"}
                            },
                            "required": ["chunk_index", "sub_query"]
                        }
                    ),
                    FunctionDeclaration(
                        name="store_buffer",
                        description="Store intermediate findings in a named buffer (RLM state management).",
                        parameters={
                            "type": "OBJECT",
                            "properties": {
                                "name": {"type": "STRING"},
                                "content": {"type": "STRING"}
                            },
                            "required": ["name", "content"]
                        }
                    ),
                    FunctionDeclaration(
                        name="finalize_answer",
                        description="Provide the final synthesized answer. Only call when you have enough information.",
                        parameters={
                            "type": "OBJECT",
                            "properties": {
                                "answer": {"type": "STRING", "description": "Complete, well-reasoned final answer"},
                                "confidence": {"type": "NUMBER", "description": "Confidence 0-1"}
                            },
                            "required": ["answer"]
                        }
                    )
                ]
            )
        ]

    # ---------- TOOL IMPLEMENTATIONS ----------
    def _peek_document(self, start: int, length: int) -> str:
        if not self.document:
            return "No document loaded."
        content = self.document.peek(start, length)
        self.trace.append(f"PEEK: chars {start}-{start+length}")
        return f"Document section [{start}:{start+length}]:\n{content}"

    def _search_document(self, query: str, max_results: int = 5) -> str:
        if not self.document:
            return "No document loaded."
        results = self.document.search(query, max_results)
        self.trace.append(f"SEARCH: '{query}' → {len(results)} hits")
        return json.dumps(results, indent=2)

    def _analyze_chunk(self, chunk_index: int, sub_query: str, depth: int = 0) -> str:
        """Recursive sub-call — the heart of RLM."""
        if depth > MAX_SUBCALL_DEPTH:
            return "Max recursion depth reached."

        if not self.document:
            return "No document loaded."

        chunk = self.document.get_chunk(chunk_index)
        if not chunk:
            return f"Invalid chunk index {chunk_index}"

        self.trace.append(f"RECURSE → chunk {chunk_index} | depth {depth} | query: {sub_query[:60]}...")

        # Build a focused sub-prompt for the recursive call
        sub_prompt = f"""You are analyzing chunk {chunk_index} of a larger document.

CHUNK CONTENT:
{chunk[:3000]}...

SUB-QUESTION: {sub_query}

Provide a concise, factual extraction or insight. If nothing relevant, say so clearly."""

        try:
            response = self.model.generate_content(sub_prompt)
            result = response.text.strip()
            # Store in buffer automatically
            buffer_name = f"chunk_{chunk_index}_{sub_query[:20].replace(' ', '_')}"
            self.buffers[buffer_name] = result
            return f"Chunk {chunk_index} analysis:\n{result}"
        except Exception as e:
            return f"Sub-call error: {str(e)}"

    def _store_buffer(self, name: str, content: str) -> str:
        self.buffers[name] = content
        self.trace.append(f"BUFFER: stored '{name}' ({len(content)} chars)")
        return f"Stored in buffer '{name}'."

    def _finalize_answer(self, answer: str, confidence: float = 0.85) -> str:
        self.trace.append("FINAL ANSWER GENERATED")
        return f"FINAL_ANSWER (confidence {confidence:.0%}):\n{answer}"

    # ---------- MAIN AGENT LOOP ----------
    def analyze(self, query: str, max_turns: int = 12) -> Dict[str, Any]:
        """
        Main RLM-powered analysis loop.
        The model decides when to peek, search, recurse, or finalize.
        """
        if not self.document:
            return {"error": "No document loaded. Call load_document() first."}

        self.trace = []
        self.buffers = {}

        system_prompt = f"""You are an expert RLM (Recursive Language Model) agent specialized in long document analysis.

DOCUMENT STATS: {self.document.stats()}

You have access to the FULL document via tools. NEVER assume you have the whole text — always use tools.

Your goal: Answer the user's query thoroughly by:
1. Exploring the document programmatically (peek, search)
2. Recursively analyzing relevant chunks with focused sub-questions
3. Storing key findings in buffers
4. Synthesizing everything into a high-quality final answer

Available tools:
- peek_document(start, length)
- search_document(query)
- analyze_chunk(chunk_index, sub_query)   ← This is your recursive power
- store_buffer(name, content)
- finalize_answer(answer, confidence)

Think step-by-step. Be strategic about which chunks to recurse on. 
Only call finalize_answer when you have strong evidence from multiple parts of the document.

Current user query: {query}
"""

        # Initial message
        messages = [{"role": "user", "parts": [system_prompt]}]

        for turn in range(max_turns):
            try:
                response = self.model.generate_content(
                    messages,
                    tools=self._get_tools(),
                    tool_config={"function_calling_config": "AUTO"}
                )

                # Check for function calls
                if response.candidates[0].content.parts[0].function_call:
                    fc = response.candidates[0].content.parts[0].function_call
                    tool_name = fc.name
                    args = {k: v for k, v in fc.args.items()}

                    # Execute the tool
                    if tool_name == "peek_document":
                        result = self._peek_document(**args)
                    elif tool_name == "search_document":
                        result = self._search_document(**args)
                    elif tool_name == "analyze_chunk":
                        result = self._analyze_chunk(**args, depth=turn)
                    elif tool_name == "store_buffer":
                        result = self._store_buffer(**args)
                    elif tool_name == "finalize_answer":
                        final = self._finalize_answer(**args)
                        return {
                            "answer": args.get("answer", final),
                            "confidence": args.get("confidence", 0.85),
                            "trace": self.trace,
                            "buffers": self.buffers,
                            "turns": turn + 1
                        }
                    else:
                        result = f"Unknown tool: {tool_name}"

                    # Feed result back
                    messages.append({"role": "model", "parts": [response.candidates[0].content]})
                    messages.append({
                        "role": "user",
                        "parts": [f"Tool result: {result}\n\nContinue reasoning or call another tool."]
                    })

                else:
                    # No tool call — model gave text response
                    text = response.text
                    messages.append({"role": "model", "parts": [text]})
                    # Force it to use tools or finalize
                    messages.append({
                        "role": "user",
                        "parts": ["Please use tools to explore the document or call finalize_answer if ready."]
                    })

            except Exception as e:
                return {"error": str(e), "trace": self.trace}

        return {
            "answer": "Analysis incomplete after max turns. Partial findings in buffers.",
            "buffers": self.buffers,
            "trace": self.trace,
            "turns": max_turns
        }


# ============================================================
# 4. EXAMPLE USAGE
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("RLM Long Document Analysis Agent — Demo")
    print("=" * 70)

    # Sample LONG document (synthetic ~8k tokens about future of AI agents)
    SAMPLE_LONG_DOC = """
    """ + textwrap.dedent("""
    # The 2026 State of Agentic AI: From Prototypes to Production

    ## Executive Summary
    Agentic AI systems have moved from experimental prototypes in 2025 to production deployments across enterprises in 2026. Adoption has reached 35% in just two years, compared to 72% for traditional AI over eight years. Key drivers include multi-agent orchestration frameworks like LangGraph and CrewAI, standardized protocols (MCP and A2A), and the emergence of Recursive Language Models (RLMs) that solve the long-context problem.

    ## 1. The Rise of Multi-Agent Orchestration
    Single agents are being replaced by coordinated teams of specialized agents. A typical production system now includes:
    - Planner Agent: Breaks down complex goals using Plan-and-Execute
    - Researcher Agent: Gathers information using live web tools and RAG
    - Executor Agent: Performs actions via APIs and tools
    - Critic/Verifier Agent: Validates outputs and triggers reflection loops
    - Governance Agent: Monitors for policy violations and safety

    LangGraph 1.0 (released October 2025) has become the dominant framework with over 400 enterprise deployments and 90 million monthly downloads. Its graph-based execution model supports cycles, conditionals, and parallel branches — essential for long-running agentic workflows.

    ## 2. Recursive Language Models (RLMs) — Solving Context Rot
    One of the biggest breakthroughs of late 2025 is Recursive Language Models. Traditional long-context models (even 1M+ token windows) suffer from "context rot" — performance degradation due to attention dilution and lost-in-the-middle effects.

    RLMs treat the entire document or conversation history as an external REPL environment. The root model only sees metadata and uses code to:
    - Programmatically peek and search sections
    - Launch recursive sub-calls on relevant chunks
    - Store findings in persistent buffers
    - Synthesize only when sufficient evidence is gathered

    The first natively trained RLM, RLM-Qwen3-8B, outperforms its base model by 28.3% on long-context benchmarks while approaching GPT-5 quality on several tasks — all at a fraction of the compute cost. This technique is particularly powerful for analyzing 10M+ token corpora such as entire codebases, legal archives, or scientific literature.

    ## 3. Protocol Standardization: The Agent Internet
    Google’s Agent-to-Agent (A2A) Protocol (v0.3 released July 2025) and Anthropic’s Model Context Protocol (MCP) have created interoperability standards. Agents can now discover each other via "Agent Cards", negotiate capabilities, and collaborate across vendors without custom integration.

    Over 150 organizations now support A2A, including Salesforce, SAP, ServiceNow, and Adobe. This has enabled "agent marketplaces" where specialized agents can be composed on the fly.

    ## 4. Enterprise Adoption & Challenges
    While experimentation is widespread, only 24% of organizations have successfully scaled agentic systems to production. Common failure modes include:
    - Layering agents on top of legacy processes instead of redesigning workflows
    - Insufficient governance and sandboxing
    - Underestimating cost (FinOps is now a first-class concern)
    - Poor evaluation — most benchmarks still focus on technical metrics rather than business outcomes

    Successful deployments (e.g., at TELUS, Rakuten, Fountain, Zapier) report 30–93% productivity gains in software engineering, candidate screening, and customer support.

    ## 5. Vertical AI Agents Outperform Generalists
    Industry-specific agents with deep domain knowledge deliver 40%+ efficiency gains:
    - Healthcare: Diagnosis support, trial matching, continuous patient monitoring
    - Finance: Real-time fraud detection, risk modeling, automated compliance
    - Manufacturing: Predictive maintenance, supply chain optimization
    - Legal: Contract analysis over millions of pages using RLMs

    ## 6. The Future: Self-Evolving Agent Swarms
    Researchers are now training agents that can improve their own scaffolding through reinforcement learning on RLM trajectories. Early results suggest that by late 2026 we will see agents capable of autonomously discovering new tools and optimizing their own multi-agent topologies.

    CLI-based coding agents (Claude Code, Cursor Composer, Windsurf) are already the dominant way many developers ship code in 2026, with reported 30% faster delivery times.

    ## Conclusion
    2026 marks the transition from "AI that chats" to "AI that acts reliably at scale." The combination of graph-based orchestration (LangGraph), recursive context management (RLMs), and standardized agent protocols (A2A/MCP) has created the foundation for hyper-autonomous enterprise systems that were science fiction just three years ago.

    Organizations that treat agentic AI as a new operating system — rather than another chatbot layer — will capture disproportionate value in the coming decade.
    """).strip()

    # Initialize agent
    agent = RLMLongDocAgent()

    # Load the long document
    agent.load_document(SAMPLE_LONG_DOC, "2026_Agentic_AI_Report")

    # Example complex query that requires synthesis across the whole document
    query = """What are the three most important technical breakthroughs in 2026 that enable production-grade agentic systems? 
For each breakthrough, explain its impact on long-context handling or multi-agent collaboration, and name at least one real-world company or framework that benefited."""

    print(f"\n🔍 Running RLM analysis on query:\n{query}\n")
    print("-" * 70)

    result = agent.analyze(query)

    print("\n" + "=" * 70)
    print("📊 FINAL RESULT")
    print("=" * 70)
    print(result.get("answer", "No answer generated."))

    if "confidence" in result:
        print(f"\nConfidence: {result['confidence']:.0%}")

    print("\n📜 Reasoning Trace (first 10 steps):")
    for i, step in enumerate(result.get("trace", [])[:10]):
        print(f"  {i+1}. {step}")

    print("\n💾 Buffers created:", list(result.get("buffers", {}).keys()))

    print("\n✅ Demo complete! Modify the query or load your own long document.")
    print("   To use with real PDFs: add PyMuPDF or pdfplumber and extract text first.")
