#include <format>
#include <iostream>
#include <vector>

#include "controller.hpp"
#include "params.hpp"

struct Agent
{
    KinematicState state;
    KinematicController controller;
};

class Simulator
{
public:
    explicit Simulator(
        SimParameters sim_params, const std::vector<double> &controller_params, 
        const std::vector<Eigen::Vector3d> &start_positions,
        const std::vector<double> &start_headings,
        const std::vector<double> &start_speeds,
        const std::vector<Eigen::Vector3d> &waypoints,
        const std::vector<Eigen::Vector3d> &target_velocities
    )
        : parameters_(std::move(sim_params)), waypoints_(waypoints), target_velocities_(target_velocities), current_wp_idx_(0)
    {
        mode_ = controller_params.size() > 0 ? static_cast<int>(controller_params[0]) : 0;
        
        for (size_t i = 0; i < parameters_.starting_states.size() && i < start_positions.size(); ++i) {
            parameters_.starting_states[i].position = start_positions[i];
            if (i < start_headings.size()) {
                parameters_.starting_states[i].heading = start_headings[i];
            }
        }

        size_t id = 0;
        for (size_t i = 0; i < parameters_.starting_states.size(); ++i)
        {
            KinematicState initial_state;
            initial_state.x = parameters_.starting_states[i].position;
            initial_state.heading = parameters_.starting_states[i].heading;
            
            double init_speed = (i < start_speeds.size()) ? (start_speeds[i] / parameters_.dt) : 5.0;
            
            initial_state.v = Eigen::Vector3d(
                init_speed * std::cos(initial_state.heading), 
                init_speed * std::sin(initial_state.heading), 
                0.0
            );
            
            agents_.push_back({
                .state = initial_state,
                .controller = KinematicController(controller_params, id++),
            });
        }
    }

    void run()
    {
        output_header_();
        
        current_goal_ = waypoints_.empty() ? Eigen::Vector3d(0,0,0) : waypoints_[0];
        current_wp_idx_ = 0;
        int final_wait_steps = 0;
        
        for (size_t step = 0; step < parameters_.steps; ++step)
        {
            if (waypoints_.empty()) break;

            std::vector<KinematicState> all_states;
            for (auto& agent : agents_) {
                all_states.push_back(agent.state);
            }

            Eigen::Vector3d final_target = waypoints_[current_wp_idx_];
            Eigen::Vector3d dir_to_wp = final_target - current_goal_;
            double dist_to_wp = dir_to_wp.norm();

            double target_speed = target_velocities_.empty() ? 5.0 : target_velocities_[current_wp_idx_].norm();
            
            if (target_speed < 0.5) {
                target_speed = 0.5;
            }

            if (dist_to_wp < target_speed * parameters_.dt) {
                current_goal_ = final_target;
                if (current_wp_idx_ < waypoints_.size() - 1) {
                    current_wp_idx_++;
                }
            } else {
                current_goal_ += (dir_to_wp / dist_to_wp) * target_speed * parameters_.dt;
            }

            int arrived_count = 0;
            double min_dist_to_final = 1e9;
            for (const auto& s : all_states) {
                double dist = (s.x - waypoints_.back()).norm();
                if (dist <= 20.0) {
                    arrived_count++;
                }
                if (dist < min_dist_to_final) {
                    min_dist_to_final = dist;
                }
            }

            if (current_wp_idx_ == waypoints_.size() - 1) {
                if (mode_ == 2 && min_dist_to_final <= 20.0) {
                    break;
                }

                if (dist_to_wp < 0.1) {
                    final_wait_steps++;
                    bool almost_all_arrived = (arrived_count >= static_cast<int>(agents_.size() * 0.70));
                    
                    if (almost_all_arrived || final_wait_steps > 800) {
                        break;
                    }
                }
            }

            for (auto& agent : agents_) {
                agent.controller.set_target(current_goal_.x(), current_goal_.y(), current_goal_.z());
            }

            step_simulation_();
            
            if (step % 10 == 0) {
                output_step_();
            }
        }
    }

private:
    void step_simulation_()
    {
        std::vector<KinematicState> all_states;
        for (auto& agent : agents_) {
            all_states.push_back(agent.state);
        }

        std::vector<Eigen::Vector3d> new_velocities;
        for (size_t agent_idx = 0; agent_idx < agents_.size(); ++agent_idx)
        {
            auto command_vel = agents_[agent_idx].controller.step(
                parameters_.dt, agent_idx, all_states
            );
            new_velocities.push_back(command_vel);
        }

        double max_turn_rate = 0.75; 

        for (size_t agent_idx = 0; agent_idx < agents_.size(); ++agent_idx)
        {
            if (mode_ != 2 && current_wp_idx_ == waypoints_.size() - 1 && 
                (agents_[agent_idx].state.x - waypoints_.back()).norm() <= 20.0) {
                agents_[agent_idx].state.v = Eigen::Vector3d(0, 0, 0);
                continue; 
            }

            Eigen::Vector3d desired_vel = new_velocities[agent_idx];
            double desired_speed = desired_vel.norm();
            
            if (desired_speed < 0.1) {
                desired_speed = 0.1; 
            }

            double desired_heading = std::atan2(desired_vel.y(), desired_vel.x());
            
            double current_speed = agents_[agent_idx].state.v.norm();
            double current_heading = 0.0;
            if (current_speed > 0.001) {
                current_heading = std::atan2(agents_[agent_idx].state.v.y(), agents_[agent_idx].state.v.x());
            } else {
                current_heading = desired_heading; 
            }

            double heading_error = desired_heading - current_heading;
            while (heading_error > M_PI) heading_error -= 2.0 * M_PI;
            while (heading_error < -M_PI) heading_error += 2.0 * M_PI;

            double max_turn_step = max_turn_rate * parameters_.dt;
            double actual_turn = std::max(-max_turn_step, std::min(max_turn_step, heading_error));
            
            double new_heading = current_heading + actual_turn;
            
            agents_[agent_idx].state.v = Eigen::Vector3d(
                desired_speed * std::cos(new_heading),
                desired_speed * std::sin(new_heading),
                0.0
            );
            
            agents_[agent_idx].state.x += agents_[agent_idx].state.v * parameters_.dt;
        }
    }

    void output_header_()
    {
        std::cout << agents_.size() << "\n";
    }

    void output_step_()
    {
        for (const auto& agent : agents_) {
            std::cout << agent.state.x.x() << " " << agent.state.x.y() << " " << agent.state.x.z() << " ";
        }
        std::cout << current_goal_.x() << " " << current_goal_.y() << " " << current_goal_.z() << "\n";
    }

    SimParameters parameters_;
    std::vector<Eigen::Vector3d> waypoints_;
    std::vector<Eigen::Vector3d> target_velocities_;
    size_t current_wp_idx_;
    int mode_;
    std::vector<Agent> agents_;
    Eigen::Vector3d current_goal_;
};

void parse_arguments(
    int argc, char **argv, 
    std::vector<double>& controller_params, 
    std::vector<Eigen::Vector3d>& start_positions, 
    std::vector<double>& start_headings, 
    std::vector<double>& start_speeds, 
    std::vector<Eigen::Vector3d>& waypoints,
    std::vector<Eigen::Vector3d>& target_velocities)
{
    for (int i = 1; i <= 8; ++i) { 
        if (argc > i) controller_params.push_back(std::stod(argv[i]));
    }
    
    if (argc > 10) { 
        int num_drones = std::stoi(argv[9]); 
        int num_waypoints = std::stoi(argv[10]); 
        int arg_idx = 11; 
        
        for (int i = 0; i < num_drones; ++i) {
            if (arg_idx + 4 < argc) { 
                double x = std::stod(argv[arg_idx++]);
                double y = std::stod(argv[arg_idx++]);
                double z = std::stod(argv[arg_idx++]);
                double heading = std::stod(argv[arg_idx++]);
                double speed = std::stod(argv[arg_idx++]);
                
                start_positions.push_back(Eigen::Vector3d(x, y, z));
                start_headings.push_back(heading);
                start_speeds.push_back(speed);
            }
        }
        
        for (int i = 0; i < num_waypoints; ++i) {
            if (arg_idx + 5 < argc) {
                double px = std::stod(argv[arg_idx++]);
                double py = std::stod(argv[arg_idx++]);
                double pz = std::stod(argv[arg_idx++]);
                double vx = std::stod(argv[arg_idx++]);
                double vy = std::stod(argv[arg_idx++]);
                double vz = std::stod(argv[arg_idx++]);
                
                waypoints.push_back(Eigen::Vector3d(px, py, pz));
                target_velocities.push_back(Eigen::Vector3d(vx, vy, vz));
            }
        }
    }
}

int main(int argc, char **argv)
try
{
    int num_drones = (argc > 9) ? std::stoi(argv[9]) : 20; 
    auto sim_parameters = get_parameters(num_drones);
    std::vector<double> controller_parameters;
    std::vector<Eigen::Vector3d> start_positions;
    std::vector<double> start_headings; 
    std::vector<double> start_speeds;
    std::vector<Eigen::Vector3d> waypoints;
    std::vector<Eigen::Vector3d> target_velocities;
    
    parse_arguments(argc, argv, controller_parameters, start_positions, start_headings, start_speeds, waypoints, target_velocities);

    auto sim = Simulator(sim_parameters, controller_parameters, start_positions, start_headings, start_speeds, waypoints, target_velocities);
    sim.run();
}
catch (std::exception &e)
{
    std::cerr << std::format("Uncaught error:\n  {}\n", e.what());
    return 1;
}