import streamlit as st
import folium
import numpy as np
import random
import tensorflow as tf
from collections import deque
from geopy.geocoders import Nominatim
from collections import defaultdict
from streamlit_folium import st_folium
import matplotlib.cm as cm
import matplotlib.colors as mcolors


# Fungsi untuk mengambil koordinat lokasi berdasarkan nama tempat (menggunakan geopy)
def get_location_coordinates(location_name):
    geolocator = Nominatim(user_agent="route_optimizer")
    try:
        location = geolocator.geocode(location_name)
        if location:
            return (location.latitude, location.longitude)
        else:
            st.error(f"Location {location_name} not found.")
            return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Fungsi untuk mengambil input lokasi dan kendaraan
def get_vehicle_and_route_input():
    start_location_name = st.text_input("Enter start location:")
    end_location_name = st.text_input("Enter destination location:")
    vehicle_type = st.selectbox("Enter vehicle type (car/motorcycle):", ['car', 'motorcycle'])
    battery_level = st.number_input("Enter battery level (%):", min_value=0, max_value=100, value=80)
    return start_location_name, end_location_name, vehicle_type, battery_level

# Data SPKLU yang dimasukkan manual
def get_spklu_locations():
    spklu_locations = [
        (-7.28906, 112.67560),  # EVCuzz Stasiun Pengisian
        (-7.28100, 112.67850),  # CHARGE+ Stasiun Pengisian
        (-7.28110, 112.74630),  # SPKLU Stasiun Pengisian
        (-7.26000, 112.74850),  # Hyundai Stasiun Pengisian
        (-7.25890, 112.74600)   # CHARGE+ Stasiun Pengisian
    ]
    return spklu_locations

# Fungsi untuk memproses rute dan menghitung jarak
def process_route(state, action, spklu_nodes):
    spklu_location = spklu_nodes[action]  # Lokasi SPKLU berdasarkan aksi yang dipilih
    distance = np.linalg.norm(np.array([state[0][0], state[0][1]]) - np.array([spklu_location[0], spklu_location[1]]))
    return distance

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Replay memory
        self.gamma = 0.95    # Discount factor
        self.epsilon = 1.0  # Exploration-exploitation tradeoff
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # Exploration: Random action
            return random.randrange(self.action_size)
        # Exploitation: Choose best action based on Q-values
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        # Add experience to replay memory
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        # Replay memory to train the model
        if len(self.memory) < batch_size:
            return  # If replay memory is not full enough, don't sample yet
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Inisialisasi DQN agent
state_size = 4  
action_size = 5  
agent = DQNAgent(state_size, action_size)

# Fungsi utama untuk proses optimasi rute
def run_dqn_process():
    start_location_name, end_location_name, vehicle_type, battery_level = get_vehicle_and_route_input()

    # Ambil koordinat untuk lokasi start dan end
    start_location = get_location_coordinates(start_location_name)
    end_location = get_location_coordinates(end_location_name)

    if not start_location or not end_location:
        st.error("Invalid locations.")
        return

    # Ambil lokasi SPKLU
    spklu_nodes = get_spklu_locations()

    # Set state dengan lokasi dan battery level
    state = np.array([start_location[0], start_location[1], battery_level, 0]).reshape(1, state_size)

    # Pilih aksi menggunakan DQN
    action = agent.act(state)
    distance = process_route(state, action, spklu_nodes)

    # Tampilkan hasil
    st.write(f"Action taken by DQN Agent: {action}")
    st.write(f"SPKLU chosen: {spklu_nodes[action]}")
    st.write(f"Distance to chosen SPKLU: {distance} km")

    # Visualisasi peta dengan Folium
    m = folium.Map(location=start_location, zoom_start=13)
    folium.Marker(start_location, popup="Start Location", icon=folium.Icon(color="blue")).add_to(m)
    folium.Marker(end_location, popup="End Location", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(spklu_nodes[action], popup="Selected SPKLU", icon=folium.Icon(color="red")).add_to(m)

    # Tampilkan peta di Streamlit
    st_folium(m, width=700)

# Jalankan proses utama
if __name__ == '__main__':
    run_dqn_process()