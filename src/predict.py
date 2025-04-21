import torch
import numpy as np
from dqn_model import DQN
from data_loader import load_road_network
from graph_builder import build_graph

# Load model
graph = build_graph(load_road_network())
state_size = 1  # Sesuaikan jika ada fitur lain
action_size = len(graph.nodes)
model = DQN(state_size, action_size)
model.load_state_dict(torch.load("dqn_model.pth"))  # Perbaiki path
model.eval()

def predict_route(start, end, max_steps=50):
    state = start
    route = [start]

    for _ in range(max_steps):  # Batasi jumlah langkah untuk menghindari infinite loop
        state_tensor = torch.FloatTensor([[state]])  # Sesuaikan bentuk input
        with torch.no_grad():
            action = torch.argmax(model(state_tensor)).item()
        
        if action == end:
            route.append(action)
            break

        if action in route or action not in graph.nodes:  # Hindari loop
            break

        route.append(action)
        state = action

    return route

if __name__ == "__main__":
    start, end = 1, 10  # Misalnya, dari node 1 ke 10
    route = predict_route(start, end)
    print(f"Rute terbaik: {route}")
