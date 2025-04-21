import json
import networkx as nx
import os

def load_road_network(json_path):
    """
    Load road network from a JSON file and build a NetworkX graph.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File {json_path} tidak ditemukan!")

    with open(json_path, 'r', encoding='utf-8') as file:
        road_data = json.load(file)
    
    G = nx.MultiDiGraph()  # Gunakan MultiDiGraph untuk mendukung jalan dua arah

    for road in road_data["roads"]:
        start, end = road["start"], road["end"]
        weight = road.get("distance", 1)  # Default weight jika tidak ada data jarak
        G.add_edge(start, end, weight=weight)

    return G

# Cek apakah graph terbentuk dengan benar
def check_graph(graph):
    print(f"✅ Graph memiliki {graph.number_of_nodes()} nodes dan {graph.number_of_edges()} edges.")
    print("🔍 Contoh 5 edges pertama:")
    print(list(graph.edges(data=True))[:5])  # Menampilkan 5 edge pertama

if __name__ == "__main__":
    json_path = "../data/jalan_osm.json"
    try:
        graph = load_road_network(json_path)
        check_graph(graph)  # Tambahkan ini untuk mengecek graph
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
