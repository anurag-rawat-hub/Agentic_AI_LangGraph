from langgraph.graph import StateGraph , START, END
from typing import TypedDict

#define State
class BMIState(TypedDict):

    weight_kg:float
    height_m:float
    bmi:float
    category:str

def calculate_bmi(state:BMIState) -> BMIState:
    weight=state["weight_kg"]
    hieght=state['height_m']
    bmi=weight/(hieght**2)
    state['bmi']=round(bmi,2)
    return state

def label_bmi(state:BMIState) -> BMIState:
    bmi=state['bmi']
    if bmi<18.5:
        state["category"]="underweight"
    elif 18.5<=bmi<25:
        state['category']="Normal"
    elif 25<=bmi<30:
        state['category']="Overweight"
    else:
        state['category']="Obese"

    return state


# define your graph
graph=StateGraph(BMIState)


#add nodes to your graph
graph.add_node('calculate_bmi',calculate_bmi)
graph.add_node('label_bmi',label_bmi)


#add edges to your graph
graph.add_edge(START,'calculate_bmi')
graph.add_edge('calculate_bmi','label_bmi')
graph.add_edge('label_bmi',END)


#compile the graph
workflow=graph.compile()

#execute the graph
initial_state={'weight_kg':80, 'height_m':1.73}
final_state=workflow.invoke(initial_state)

print(final_state)

graph_image = workflow.get_graph().draw_mermaid_png()

with open("workflow_graph.png", "wb") as f:
    f.write(graph_image)

print("Graph saved as workflow_graph.png")

