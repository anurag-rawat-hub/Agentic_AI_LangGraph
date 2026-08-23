import os
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from google import genai
from dotenv import load_dotenv

load_dotenv()

client=genai.Client()


#create state
class BlogState(TypedDict):
    title:str
    outline:str
    content:str


#create graph
graph=StateGraph(BlogState)

def create_outline(state:BlogState) -> BlogState:
    
    #fetch title
    title=state['title']

    #call llm gen outline
    prompt=f"Generate a detailed outline for a blog on the topic-{title}"

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
    )

    #update state
    state['outline']=response.text

    return state

def create_blog(state:BlogState) -> BlogState:

    #fetch title
    title=state['title']

    #fetch outline
    outline=state['outline']

    prompt=f"Write a detailed blog on the -{title} using the following outline\n {outline}"

    content = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
    )

    #update state
    state['content']=content.text

    return state


#add node
graph.add_node('create_outline',create_outline)
graph.add_node('create_blog',create_blog)


#add edges
graph.add_edge(START, 'create_outline')
graph.add_edge('create_outline','create_blog')
graph.add_edge('create_blog',END)


#compile
workflow=graph.compile()

#execute
initial_state={'title':'Rise of AI in India'}

final_state=workflow.invoke(initial_state)
print(final_state)


graph_image = workflow.get_graph().draw_mermaid_png()

with open("workflow_graph.png", "wb") as f:
    f.write(graph_image)

print("Graph saved as workflow_graph.png")



