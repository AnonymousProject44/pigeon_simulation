#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <vector>

struct StartingState
{
    Eigen::Vector3d position;
    double heading;
};

struct SimParameters
{
    double dt;
    size_t steps;
    double neighborhood_range;
    std::vector<StartingState> starting_states;
};

inline SimParameters get_parameters(int num_uavs)
{
    std::vector<StartingState> initial_states;
    
    // Create a dynamic grid based on the number of birds in the CSV
    int cols = 4;
    for (int i = 0; i < num_uavs; ++i) {
        int row = i / cols;
        int col = i % cols;
        initial_states.push_back({.position = {row * 2.0, col * 2.0, 10.0}, .heading = 0.0});
    }

    return {
        .dt = 0.02,
        .steps = 25000, 
        .neighborhood_range = 15.0,
        .starting_states = initial_states,
    };
}