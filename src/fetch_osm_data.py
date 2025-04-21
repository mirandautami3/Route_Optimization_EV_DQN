import osmnx as ox
import json

# Ambil data jalan di Surabaya
graph = ox.graph_from_place("Surabaya, Indonesia", network_type="drive")

# Konversi ke format JSON
road_data = {"roads": []}
for u, v, data in graph.edges(data=True):
    road_data["roads"].append({
        "start": u,
        "end": v,
        "distance": data.get("length", 1)  # Default 1 jika tidak ada panjang jalan
    })

# Simpan ke file JSON
with open("data/jalan_osm.json", "w", encoding="utf-8") as f:
    json.dump(road_data, f, indent=4)

print("✅ File jalan_osm.json berhasil dibuat!")
