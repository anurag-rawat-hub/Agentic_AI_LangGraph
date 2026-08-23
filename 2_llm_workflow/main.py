from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from typing import TypedDict
from google import genai
import os

load_dotenv()

# 1. Initialize the native Google GenAI Client
# It will automatically pick up GEMINI_API_KEY from your environment variables
client = genai.Client()

#create a state
class LLMState(TypedDict):
    question:str
    answer:str

def llm_qa(state:LLMState)->LLMState:

    #extract a question from state
    question=state['question']

    #form a prompt
    prompt=f"Answer the folowing question {question}"

    #ask the question to the llm
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
    )

    #update the answer in the state
    state['answer']=response.text

    return state



#create a graph
graph=StateGraph(LLMState)


#add nodes
graph.add_node('llm_qa',llm_qa)


#add edges
graph.add_edge(START,'llm_qa')
graph.add_edge('llm_qa',END)


#compile
workflow=graph.compile()


#execute
initial_state={'question':'How far is the moon from earth?'}

final_state=workflow.invoke(initial_state)
print(final_state)


graph_image = workflow.get_graph().draw_mermaid_png()

with open("workflow_graph.png", "wb") as f:
    f.write(graph_image)

print("Graph saved as workflow_graph.png")


