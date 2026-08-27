from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

from langgraph.graph.message import add_messages


#state
class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


#chat niode
def chat_node(state: ChatState):

    # take user query from state
    messages = state['messages']

    # send to llm
    response = llm.invoke(messages)

    # response store state
    return {'messages': [response]}


checkpointer=MemorySaver()

graph = StateGraph(ChatState)

# add nodes
graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)


'''
initial_state = {
    'messages': [HumanMessage(content='What is the capital of india')]
}

chatbot.invoke(initial_state)['messages'][-1].content

final_state=chatbot.invoke(initial_state)
print(final_state)



graph_image = chatbot.get_graph().draw_mermaid_png()

with open("workflow_graph.png", "wb") as f:
    f.write(graph_image)

print("Graph saved as workflow_graph.png")

'''

thread_id='1'

while True:
    user_message=input('Type here: ')
    print('User:', user_message)

    if user_message.strip().lower() in ['exit', 'quit', 'bye']:
        break

    config={'configurable': {'thread_id': thread_id}}
    response=chatbot.invoke({'messages':[HumanMessage(content=user_message)]}, config=config)

    print('AI : ', response['messages'][-1].content)