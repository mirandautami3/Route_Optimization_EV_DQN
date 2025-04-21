import os
import pandas as pd
import osmnx as ox
import networkx as nx

# Path ke data SPKLU
spklu_file_path = "/Users/miranda/Documents/Skripsi/Code Rekomendasi Rute DQN/data/spklu_dataset.csv"

def load_spklu_data():
    """Load dataset SPKLU (Stasiun Pengisian Kendaraan Listrik Umum)."""
    if not os.path.exists(spklu_file_path):
        raise FileNotFoundError(f"File {spklu_file_path} tidak ditemukan!")
    return pd.read_csv(spklu_file_path)

def get_osm_graph(location="Surabaya, Indonesia", network_type="drive"):
    """Ambil graph jalan dari OSM untuk lokasi yang ditentukan."""
    print(f"📡 Mengambil data jalan dari OSM untuk {location}...")
    graph = ox.graph_from_place(location, network_type=network_type)
    print("✅ Data jalan berhasil diambil!")
    return graph

if __name__ == "__main__":
    # Load data SPKLU
    spklu_df = load_spklu_data()
    print("📍 SPKLU Data Head:")
    print(spklu_df.head())

    # Ambil data jalan dari OSM
    jalan_graph = get_osm_graph()

    # Cetak jumlah node dan edge pada graph jalan
    print(f"\n🌍 Graph Jalan: {len(jalan_graph.nodes)} nodes, {len(jalan_graph.edges)} edges")
