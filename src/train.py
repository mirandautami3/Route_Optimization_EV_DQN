import numpy as np
import torch
from data_loader import load_spklu_data
from graph_builder import load_road_network
from dqn_model import DQNAgent

# Load data
spklu_data = load_spklu_data()
graph = load_road_network("../data/jalan_osm.json")  # Pastikan path benar

# Cek apakah graf berhasil dimuat
print(f"🔍 Total nodes dalam graf: {len(graph.nodes)}")
print(f"📌 Contoh nodes: {list(graph.nodes)[:10]}")  # Menampilkan 10 node pertama
if len(graph.nodes) == 0:
    raise ValueError("❌ Graf kosong! Periksa data JSON jalan.")

# Inisialisasi agent
state_size = 1  # Saat ini hanya node ID, bisa diperluas jika pakai fitur lain
action_size = len(graph.nodes)
agent = DQNAgent(state_size, action_size)

# Hyperparameters
EPISODES = 500
BATCH_SIZE = 32

# Training loop
for episode in range(EPISODES):
    state = np.random.choice(list(graph.nodes))  # Pilih titik awal dari node yang valid
    done = False
    total_reward = 0

    while not done:
        if state not in graph.nodes:
            print(f"⚠️ Node {state} tidak ditemukan dalam graf. Memilih ulang...")
            state = np.random.choice(list(graph.nodes))  # Pilih ulang node yang valid
            continue

        action = agent.act([state])

        # Validasi action sebelum digunakan
        if action not in graph.nodes:
            print(f"⚠️ Action {action} tidak valid! Memilih ulang...")
            continue

        neighbors = list(graph.neighbors(action))

        if not neighbors:
            print(f"⚠️ Node {action} tidak memiliki tetangga. Menghentikan episode...")
            break  # Hindari loop jika tidak ada tetangga

        # Pilih next_state dari tetangga yang valid
        next_state = np.random.choice(neighbors)
        reward = -graph[state][next_state].get('weight', 1)  # Pastikan 'weight' ada, default 1
        done = next_state in spklu_data  # Cek apakah sudah sampai tujuan

        # Simpan pengalaman ke memori DQN
        agent.remember([state], action, reward, [next_state], done)
        state = next_state
        total_reward += reward

    # Latih model dengan pengalaman yang dikumpulkan
    if len(agent.memory) > BATCH_SIZE:
        agent.replay(BATCH_SIZE)

    print(f"✅ Episode {episode+1}/{EPISODES}, Total Reward: {total_reward}")

# Simpan model setelah training selesai
torch.save(agent.model.state_dict(), "dqn_model.pth")
print("🎯 Model DQN berhasil disimpan!")
