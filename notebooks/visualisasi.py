import streamlit as st
import pickle
import math
import osmnx as ox
import networkx as nx
import numpy as np
from geopy.geocoders import Nominatim

# Load graph
graph_path = "data/road_graph_weighted_surabaya.pkl"
try:
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    st.success("✅ Graf jalan Surabaya berhasil dimuat.")
except Exception as e:
    st.error(f"❌ Gagal memuat graf jalan: {e}")
    G = None

# Fungsi haversine
def haversine(lon1, lat1, lon2, lat2):
    R = 6371.0  # radius bumi dalam km
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# Fungsi mencari node terdekat
def find_nearest_node(G, coord):
    return ox.distance.nearest_nodes(G, X=coord[1], Y=coord[0])

# Fungsi membangun matriks jarak
def build_distance_matrix(valid_nodes, graph):
    distance_matrix = np.zeros((len(valid_nodes), len(valid_nodes)))
    node_to_index = {node: index for index, node in enumerate(valid_nodes)}  # Pemetaan node ke indeks
    for i, start_node in enumerate(valid_nodes):
        for j, end_node in enumerate(valid_nodes):
            if i != j:
                try:
                    # Mencari jarak terpendek antara start_node dan end_node dalam graph
                    path = nx.shortest_path(graph, start_node, end_node, weight='distance')
                    distance = nx.shortest_path_length(graph, start_node, end_node, weight='distance')
                    distance_matrix[i][j] = distance
                except nx.NetworkXNoPath:
                    distance_matrix[i][j] = float('inf')  # Tidak ada jalur yang tersedia
    return distance_matrix, node_to_index

# Fungsi perhitungan energi
def calculate_energy_required(distance, consumption_rate=0.2):
    return distance * consumption_rate

# UI
st.title("🔋 Estimasi Energi Perjalanan EV di Surabaya")

start_address = st.text_input("Masukkan alamat awal:")
end_address = st.text_input("Masukkan alamat tujuan:")

battery_options = {
    "Wuling Air EV Standard Range": {"capacity": 17.3, "range": 200},
    "Wuling Air EV Long Range": {"capacity": 31.9, "range": 300},
    "Wuling BinguoEV Premium Range": {"capacity": 37.9, "range": 410},
    "Wuling BinguoEV Long Range": {"capacity": 31.9, "range": 333},
    "Wuling Cloud EV": {"capacity": 50.6, "range": 460}
}

selected_model = st.selectbox("Pilih model mobil EV:", list(battery_options.keys()))
battery_capacity = battery_options[selected_model]['capacity']

battery_percent = st.slider("Persentase baterai saat ini:", 0, 100, 80)

connector_type = st.selectbox("Pilih jenis konektor:", ['CCS2', 'AC Type 2', 'GBT', 'CHAdeMO'])

if st.button("Hitung Energi"):
    if not start_address or not end_address:
        st.warning("Tolong isi alamat awal dan tujuan.")
    elif G is None:
        st.error("Graf jalan tidak tersedia.")
    else:
        geo = Nominatim(user_agent="ev_app")
        start = geo.geocode(start_address)
        end = geo.geocode(end_address)

        if not start or not end:
            st.error("Alamat tidak ditemukan. Coba lagi dengan alamat yang lebih spesifik.")
        else:
            st.success(f"Lokasi awal: {start.latitude}, {start.longitude}")
            st.success(f"Lokasi tujuan: {end.latitude}, {end.longitude}")

            start_coords = (start.latitude, start.longitude)
            end_coords = (end.latitude, end.longitude)

            start_node = find_nearest_node(G, start_coords)
            end_node = find_nearest_node(G, end_coords)

            # --- Menambahkan lebih banyak node ke valid_nodes ---
            valid_nodes = [start_node, end_node]  # Mulai dengan start_node dan end_node

            # Jika ada node SPKLU, tambahkan ke valid_nodes
            spklu_nodes = [node_id for node_id, data in G.nodes(data=True) if data.get("is_spklu")]  # Misalnya, node SPKLU
            valid_nodes.extend(spklu_nodes)  # Menambahkan SPKLU ke dalam valid_nodes jika ada

            # Membuat distance_matrix dengan valid_nodes yang lebih lengkap
            distance_matrix, node_to_index = build_distance_matrix(valid_nodes, G)
            st.write(f"✅ Matriks jarak dibangun dengan sukses, ukuran matriks: {len(distance_matrix)}x{len(distance_matrix[0])}")
            
            # --- Proses 2: Menghitung Jarak dan Energi yang Dibutuhkan ---
            distance = haversine(start_coords[1], start_coords[0], end_coords[1], end_coords[0])
            st.info(f"Jarak antar lokasi: {distance:.2f} km")

            current_energy = (battery_percent / 100.0) * battery_capacity
            required_energy = calculate_energy_required(distance)

            st.write(f"🔋 Energi yang tersedia: {current_energy:.2f} kWh")
            st.write(f"⚡ Energi yang dibutuhkan: {required_energy:.2f} kWh")

            # --- Proses 3: Mengecek apakah SPKLU diperlukan selama perjalanan ---
            spklu_nodes_list = [n for n, attr in G.nodes(data=True) if attr.get("is_spklu") == True]
            st.write(f"Total Jumlah SPKLU di Surabaya: {len(spklu_nodes_list)}")

            # Memeriksa jika jalur perjalanan melewati SPKLU
            spklu_in_route = []
            for node in spklu_nodes:
                if node in valid_nodes:
                    spklu_in_route.append(node)

            if spklu_in_route:
                st.success(f"✅ Ditemukan SPKLU pada jalur: {spklu_in_route}")
                st.info(f"Jumlah SPKLU yang tersedia di jalur perjalanan: {len(spklu_in_route)}")
            else:
                st.warning("⚠️ Tidak ada SPKLU pada jalur perjalanan ini.")

            # Menampilkan apakah perjalanan bisa selesai dengan energi yang ada atau perlu SPKLU
            if current_energy >= required_energy:
                st.success("✅ Energi cukup untuk perjalanan tanpa perlu mengisi daya.")
            else:
                st.warning("❌ Energi tidak cukup. Anda perlu mengisi daya di SPKLU.")
                if spklu_in_route:
                    st.success("✅ Ada SPKLU di jalur Anda untuk mengisi daya.")
                else:
                    st.warning("⚠️ Tidak ada SPKLU di jalur Anda. Periksa rute atau pilih rute yang memiliki SPKLU.")
            
            st.markdown("---")
            st.write(f"🔌 Jenis konektor: **{connector_type}**")
            if start_node in G.nodes and 'available_connectors' in G.nodes[start_node]:
                available = G.nodes[start_node]['available_connectors']
                if connector_type in available:
                    st.success(f"✅ Konektor {connector_type} tersedia di node awal.")
                else:
                    st.warning(f"⚠️ Konektor {connector_type} tidak tersedia di node awal.")


# # # run
# # # streamlit run notebooks/visualisasi.py


