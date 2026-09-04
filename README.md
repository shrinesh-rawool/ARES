# 🤖 ARES

### Autonomous Robot Edge Synchronization

> **Autonomous robots. Distributed decisions. Resilient fleets.**

ARES is an **Edge-AI based decentralized fleet coordination system** for Autonomous Mobile Robots (AMRs) operating in smart warehouses.

Unlike traditional centralized fleet management, ARES enables individual robots to make local decisions, communicate directly with nearby peers, resolve spatial conflicts, dynamically reroute around obstacles, and reallocate tasks without depending on a central controller.

---

## 🌟 Why ARES?

Modern warehouses increasingly rely on fleets of AMRs for material handling and parcel transportation. Centralized fleet controllers can introduce problems such as:

* 🌐 **Single points of failure**
* ⏱️ **High communication latency**
* 📶 **Wi-Fi dead zones**
* 🚧 **Deadlocks at narrow intersections**
* 🔄 **Slow recovery from robot failures**

ARES addresses these challenges by moving coordination and intelligence closer to the robots themselves.

---

## 🧠 Core Idea

Each AMR acts as an independent **edge node** with its own:

```text
┌──────────────────────────────┐
│          AMR Node             │
│                              │
│  Local Perception            │
│        ↓                     │
│  Path Planning               │
│        ↓                     │
│  Edge-AI / MARL              │
│        ↓                     │
│  P2P Communication           │
│                              │
└──────────────┬───────────────┘
               │
        P2P Mesh Network
               │
       ┌───────┴───────┐
       │               │
     AMR #2          AMR #3
```

Robots periodically exchange compressed telemetry through a local **P2P mesh / UDP multicast channel**, allowing the fleet to maintain awareness of nearby agents without relying on a central controller.

---

## ✨ Key Features

### 🤖 Decentralized Fleet Coordination

Each robot independently performs perception, planning, decision-making, and communication.

### 🧠 Multi-Agent Reinforcement Learning

ARES uses **MARL** to arbitrate conflicts between robots at intersections and other choke points.

The action space includes:

```text
Advance
Yield
Creep
Halt
Sidestep
```

The agent considers local occupancy information, nearby robot states, and remaining distance to its destination.

### 🗺️ Dynamic Path Planning

ARES uses **Space-Time A*** for local path planning, allowing robots to reason about both spatial and temporal conflicts.

### 📡 Peer-to-Peer Communication

Robots communicate directly using a lightweight P2P networking layer based on:

* ZeroMQ
* UDP Multicast

Telemetry is broadcast at approximately **10–20 Hz**.

### 🚧 Dynamic Obstacle Handling

When a robot detects an unexpected obstruction, it broadcasts a localized costmap update. Affected peers can then trigger local replanning rather than waiting for a centralized controller.

### 🔄 Self-Healing Fleet

If a robot fails during a task, nearby robots detect the missing heartbeat and can autonomously reallocate the unfinished task using a **Contract Net Protocol (CNP)** auction.

### 🚨 Priority Task Handling

High-priority deliveries can trigger other robots to temporarily yield or perform holding maneuvers, clearing the required corridor for priority traffic.

---

## 🏗️ Architecture

```text
                 WAREHOUSE ENVIRONMENT
        ┌─────────────────────────────────┐
        │ Aisles │ Shelves │ Pick/Drop    │
        └────────────────┬────────────────┘
                         │
              Local Sensors & Odometry
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   ┌─────────┐      ┌─────────┐      ┌─────────┐
   │  AMR #1 │      │  AMR #2 │      │  AMR #3 │
   ├─────────┤      ├─────────┤      ├─────────┤
   │ Path    │      │ Path    │      │ Path    │
   │ Engine  │      │ Engine  │      │ Engine  │
   ├─────────┤      ├─────────┤      ├─────────┤
   │ Edge-AI │      │ Edge-AI │      │ Edge-AI │
   │  MARL   │      │  MARL   │      │  MARL   │
   ├─────────┤      ├─────────┤      ├─────────┤
   │ P2P     │      │ P2P     │      │ P2P     │
   │  Comms  │      │  Comms  │      │  Comms  │
   └────┬────┘      └────┬────┘      └────┬────┘
        │                │                │
        └────────────────┼────────────────┘
                         │
                  P2P Mesh / UDP
                         │
                         ▼
              ┌──────────────────────┐
              │ Fleet Dashboard      │
              │                      │
              │ • Positions          │
              │ • Robot Health       │
              │ • KPIs               │
              │ • Fleet State        │
              └──────────────────────┘
```

The architecture is designed around **distributed intelligence**, with each AMR functioning as a self-contained edge node.

---

## 🧪 Evaluation Scenarios

ARES is designed around five benchmark scenarios:

| Scenario                        | Description                                                   |
| ------------------------------- | ------------------------------------------------------------- |
| 🅰️ **Nominal Transport**       | Multiple AMRs execute independent warehouse routes            |
| 🅱️ **Intersection Contention** | Three AMRs negotiate passage through a narrow intersection    |
| 🅲 **Dynamic Obstruction**      | A blocked aisle triggers distributed replanning               |
| 🅳 **Node Failure**             | A failed AMR's task is reallocated to another robot           |
| 🅴 **Emergency Priority**       | A high-priority task causes other robots to temporarily yield |

These scenarios test the system's navigation, coordination, resilience, and task-allocation capabilities.

---

## 📊 Target Performance

ARES targets the following benchmark characteristics:

| Metric                     |                      Target |
| -------------------------- | --------------------------: |
| 🤝 Inter-Robot Collisions  |                       **0** |
| 🚀 Task Throughput         |  **>22% gain** vs. baseline |
| 🔒 Deadlock Resolution     | **0% unresolved deadlocks** |
| ⚡ Local Replanning Latency |                  **<35 ms** |
| 🔄 Failure Recovery        |                    **<2.0** |
| 📡 Communication Payload   |       **<2 KB/s per robot** |

The project specification defines these as target benchmark requirements for the proposed Edge-AI MARL system.

---

## 🛠️ Technology Stack

### Edge & Simulation

* **Python 3.10+**
* **ROS 2 Humble**
* **Gazebo**
* 2D WebGL / Pygame simulator

### AI & Planning

* **PyTorch**
* **Stable-Baselines3**
* MAPPO / DQN
* NumPy
* SciPy
* Space-Time A*

### Networking

* **ZeroMQ**
* **UDP Multicast**
* P2P / brokerless communication

### Dashboard

* **React.js**
* WebSockets
* HTML5 Canvas
* Three.js

The proposed implementation targets low-power ARM platforms such as **Raspberry Pi 4** and **NVIDIA Jetson Nano**.

---


## 🔬 Research Focus

ARES explores the intersection of:

```text
        Edge Computing
              │
              ▼
       Multi-Agent AI
              │
              ▼
      Robot Coordination
              │
              ▼
       P2P Networking
              │
              ▼
     Resilient Logistics
```

The central goal is to investigate whether **local intelligence + peer-to-peer coordination** can provide a more responsive and resilient alternative to centralized AMR fleet management.

---

## 📜 License

This project is currently under development.
License information will be added as the project progresses.

---

<div align="center">

### 🤖 ARES

**Autonomous Robot Edge Synchronization**

*Distributed intelligence for the next generation of smart warehouses.*

</div>
