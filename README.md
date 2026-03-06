# pigeon_simulation

The pigeon_simulation project provides a comprehensive environment for modeling the flight dynamics and swarm behavior of fixed-wing autonomous agents. It includes a high-performance C++ kinematic simulator alongside Python tools for optimizing flocking parameters and visualizing trajectory data.

## How to build the simulator

Install the required dependencies using your package manager.

```bash
# Install dependencies
apt-get update && apt-get install -y cmake build-essential libeigen3-dev libboost-dev
```

Configure and build the project using CMake presets.

```bash
# Build project
cmake --workflow --preset default
```

## How to run the simulator
Execute the compiled binary directly from the command line.

```bash
# Run simulator with default parameters
./install/bin/simulator
```
Pass arguments to set controller parameters, the number of drones, and waypoints. Append individual configurations for each drone and waypoint in sequence.
```bash
# Run with custom parameters
./install/bin/simulator 6.0 12.0 4.0 0.8 1.2 1.5 25.0 0.0 20 5
```

### Output Format
The Python scripts evaluate three fixed-wing control methods: **Consensus Leaderless**, **Fixed Leader**, and **Dynamic Leader**.

Use the optimizer script to fine-tune controller parameters via Covariance Matrix Adaptation Evolution Strategy (CMA-ES).

```bash
# Run optimization script
python3 scripts/optimizer.py
```

Use the runner script to execute the simulation, calculate performance metrics, and generate an animated comparison video.

```bash
# Run simulation and visualization script
python3 scripts/run_sim.py
```