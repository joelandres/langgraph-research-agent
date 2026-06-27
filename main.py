from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next_step: str
    current_draft: str

llm = ChatOpenAI(model="gpt-4o", temperature=0)

def supervisor_node(state: AgentState):
    print("🧠 [Supervisor]: Planning next move...")

    history = [m.name for m in state["messages"] if m.name]
    
    if "Researcher" not in history:
        return {"next_step": "research"}
    elif "Writer" not in history:
        return {"next_step": "writer"}
    elif "QA" not in history:
        return {"next_step": "qa"}
    else:
        return {"next_step": "finish"}

def research_worker(state: AgentState):
    print("🤖 [Research Worker]: Gathering technical points...")
    last_message = state['messages'][-1].content
    response = llm.invoke(f"Provide 3 key architectural facts for: {last_message}")
    return {
        "messages": [HumanMessage(content=response.content, name="Researcher")],
        "current_draft": response.content
    }

def writer_worker(state: AgentState):
    print("🤖 [Writer Worker]: Drafting formal summary...")
    draft = state.get("current_draft", "")
    response = llm.invoke(f"Turn these facts into a cohesive professional paragraph:\n{draft}")
    return {
        "messages": [HumanMessage(content=response.content, name="Writer")],
        "current_draft": response.content
    }

def qa_worker(state: AgentState):
    print("🤖 [QA Worker]: Reviewing final output accuracy...")
    draft = state.get("current_draft", "")
    response = llm.invoke(f"Review and polish this paragraph for executive presentation:\n{draft}")
    return {
        "messages": [HumanMessage(content=response.content, name="QA")],
        "current_draft": response.content
    }

def supervisor_conditional_router(state: AgentState):
    return state["next_step"]

workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("research", research_worker)
workflow.add_node("writer", writer_worker)
workflow.add_node("qa", qa_worker)

workflow.add_edge(START, "supervisor")
workflow.add_edge("research", "supervisor")
workflow.add_edge("writer", "supervisor")
workflow.add_edge("qa", "supervisor")

workflow.add_conditional_edges(
    "supervisor",
    supervisor_conditional_router,
    {
        "research": "research",
        "writer": "writer",
        "qa": "qa",
        "finish": END
    }
)

# We intercept control immediately BEFORE the QA node runs to allow Human Approval.
# No checkpointer here: when served via `langgraph dev`/LangGraph API, persistence
# is handled by the platform, and a custom checkpointer on this graph is rejected.
app = workflow.compile(interrupt_before=["qa"])

# Only runs when executing this file directly (e.g. `python main.py`),
# not when `langgraph dev` imports this module to load `app`.
if __name__ == "__main__":
    local_app = workflow.compile(checkpointer=MemorySaver(), interrupt_before=["qa"])
    config = {"configurable": {"thread_id": "consulting_project_001"}}
    initial_input = {"messages": [HumanMessage(content="Explain LangGraph Cloud advantages.")]}

    print("\n--- PHASE 1: Initial Processing ---")
    # Loops through Supervisor -> Research -> Writer
    for event in local_app.stream(initial_input, config=config, stream_mode="updates"):
        for node, update in event.items():
            if node != "supervisor" and isinstance(update, dict) and update.get("current_draft"):
                print(f"\n📝 Draft after {node}:\n{update['current_draft']}\n")

    # State automatically pauses because of our interrupt rule.
    snapshot = local_app.get_state(config)
    print(f"\n⏸️ GRAPH PAUSED. Next node queued: {snapshot.next}")

    # Reviewing the draft and choosing to proceed.
    user_approval = input("\nType 'y' to approve draft and send to QA worker: ")

    if user_approval.strip().upper() == 'Y':
        print("\n--- PHASE 2: Resuming with Human Approval ---")
        # Resume passing None as input since we are simply triggering the queue
        for event in local_app.stream(None, config=config, stream_mode="updates"):
            for node, update in event.items():
                if node != "supervisor" and isinstance(update, dict) and update.get("current_draft"):
                    print(f"\n📝 Draft after {node}:\n{update['current_draft']}\n")

        final_state = local_app.get_state(config)
        print(f"\n🏁 Final Output:\n{final_state.values['messages'][-1].content}")
    else:
        print("Execution aborted by user.")
