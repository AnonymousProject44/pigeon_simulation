#pragma once

#include <vector>
#include <Eigen/Dense>

struct KinematicState {
    Eigen::Vector3d x;
    Eigen::Vector3d v;
    double heading;
};

class KinematicController
{
 public:
    explicit KinematicController(const std::vector<double> &parameters, size_t id)
        : id_(id), leader_initialized_(false), current_dynamic_leader_(0)
    {
        mode_ = parameters.size() > 0 ? static_cast<int>(parameters.at(0)) : 0;
        max_speed_ = parameters.size() > 1 ? parameters.at(1) : 6.0;
        rep_radius_ = parameters.size() > 2 ? parameters.at(2) : 12.0;
        rep_weight_ = parameters.size() > 3 ? parameters.at(3) : 4.0;
        param_a_ = parameters.size() > 4 ? parameters.at(4) : 0.8; 
        param_b_ = parameters.size() > 5 ? parameters.at(5) : 1.2; 
        param_c_ = parameters.size() > 6 ? parameters.at(6) : 1.5; 
        interaction_radius_ = parameters.size() > 7 ? parameters.at(7) : 25.0; 
        
        target_x_ = 0.0;
        target_y_ = 0.0;
        target_z_ = 10.0;
    }

    void set_target(double x, double y, double z)
    {
        target_x_ = x;
        target_y_ = y;
        target_z_ = z;
    }

    Eigen::Vector3d step(
    double dt, size_t my_id, const std::vector<KinematicState>& all_states)
    {
        KinematicState my_state = all_states[my_id];
        Eigen::Vector3d pos = my_state.x;
        Eigen::Vector3d target(target_x_, target_y_, target_z_);

        double fov_limit_rad = (316.4 / 2.0) * (M_PI / 180.0);

        Eigen::Vector3d v_repulsion(0, 0, 0);
        Eigen::Vector3d center_of_mass(0, 0, 0);
        Eigen::Vector3d my_v = my_state.v;
        Eigen::Vector3d avg_velocity = my_state.v;
        int visible_neighbors = 0;

        for (size_t i = 0; i < all_states.size(); ++i) {
            if (i == my_id) continue;
            
            Eigen::Vector3d vec_to_neighbor = all_states[i].x - pos;
            double dist = vec_to_neighbor.norm();

            double angle_to_neighbor = std::atan2(vec_to_neighbor.y(), vec_to_neighbor.x());
            double angle_diff = angle_to_neighbor - my_state.heading;
            
            while (angle_diff > M_PI) angle_diff -= 2.0 * M_PI;
            while (angle_diff < -M_PI) angle_diff += 2.0 * M_PI;
            
            if (std::abs(angle_diff) > fov_limit_rad) continue;

            if (mode_ == 0) {
                if (dist < interaction_radius_) {
                    visible_neighbors++;
                    center_of_mass += all_states[i].x;
                    avg_velocity += all_states[i].v;
                }
            } else {
                visible_neighbors++;
                center_of_mass += all_states[i].x;
                avg_velocity += all_states[i].v;
            }

            if (dist > 0 && dist < rep_radius_) {
                v_repulsion -= (vec_to_neighbor / dist) * (rep_radius_ - dist);
            }
        }

        if (v_repulsion.norm() > 0.0001) {
            v_repulsion.normalize();
        }

        Eigen::Vector3d desired_vel(0, 0, 0);
        size_t active_leader_id = 0;

        if (mode_ == 0) {
            Eigen::Vector3d v_cohesion(0, 0, 0);
            Eigen::Vector3d v_alignment(0, 0, 0);
            
            if (visible_neighbors > 0) {
                center_of_mass /= visible_neighbors;
                v_cohesion = center_of_mass - pos;
                if (v_cohesion.norm() > 0.0001) v_cohesion.normalize();
                
                avg_velocity /= (visible_neighbors + 1);
                v_alignment = avg_velocity;
                if (v_alignment.norm() > 0.0001) v_alignment.normalize();
            } else {
                v_alignment = my_state.v;
                if (v_alignment.norm() > 0.0001) v_alignment.normalize();
            }
            
            Eigen::Vector3d v_target = target - pos;
            if (v_target.norm() > 0.0001) v_target.normalize();

            Eigen::Vector3d blend = v_cohesion * param_a_ + v_target * param_b_ + v_repulsion * rep_weight_ + v_alignment * param_c_;
            if (blend.norm() > 0.0001) blend.normalize();
            
            desired_vel = blend * max_speed_;
        }
        else if (mode_ == 1) {
            active_leader_id = 0;
            if (my_id == active_leader_id) {
                
                Eigen::Vector3d v_target = target - pos;
                if (v_target.norm() > 0.0001) v_target.normalize();

                Eigen::Vector3d blend = v_target * param_c_;
                if (blend.norm() > 0.0001) blend.normalize();
                
                desired_vel = blend * (max_speed_ - 0.5);
            } else {
                Eigen::Vector3d v_leader(0, 0, 0);
                
                Eigen::Vector3d vec_to_leader = all_states[active_leader_id].x - pos;
                double angle_to_leader = std::atan2(vec_to_leader.y(), vec_to_leader.x());
                double angle_diff = angle_to_leader - my_state.heading;
                
                while (angle_diff > M_PI) angle_diff -= 2.0 * M_PI;
                while (angle_diff < -M_PI) angle_diff += 2.0 * M_PI;
                
                if (std::abs(angle_diff) <= fov_limit_rad) {
                    v_leader = vec_to_leader;
                    if (v_leader.norm() > 0.0001) v_leader.normalize();
                }
                
                Eigen::Vector3d v_alignment = all_states[active_leader_id].v;
                if (v_alignment.norm() > 0.0001) v_alignment.normalize();
                
                Eigen::Vector3d blend = v_leader * param_b_ + v_repulsion * rep_weight_ + v_alignment * param_a_;
                if (blend.norm() > 0.0001) blend.normalize();
                
                desired_vel = blend * max_speed_;
            }
        }
        else if (mode_ == 2) {
            if (!leader_initialized_) {
                double min_dist = (all_states[0].x - target).norm();
                current_dynamic_leader_ = 0;
                for (size_t i = 1; i < all_states.size(); ++i) {
                    double d = (all_states[i].x - target).norm();
                    if (d < min_dist) {
                        min_dist = d;
                        current_dynamic_leader_ = i;
                    }
                }
                leader_initialized_ = true;
            } else {
                size_t candidate_leader = current_dynamic_leader_;
                double current_dist = (all_states[current_dynamic_leader_].x - target).norm();
                double min_dist = current_dist;

                for (size_t i = 0; i < all_states.size(); ++i) {
                    double d = (all_states[i].x - target).norm();
                    if (d < min_dist) {
                        min_dist = d;
                        candidate_leader = i;
                    }
                }

                if ((current_dist - min_dist) > 2.0) {
                    current_dynamic_leader_ = candidate_leader;
                }
            }

            active_leader_id = current_dynamic_leader_;

            if (my_id == active_leader_id) {
                
                Eigen::Vector3d v_target = target - pos;
                if (v_target.norm() > 0.0001) v_target.normalize();

                Eigen::Vector3d blend = v_target * param_c_;
                if (blend.norm() > 0.0001) blend.normalize();
                
                desired_vel = blend * (max_speed_ - 0.5);
            } else {
                Eigen::Vector3d v_leader(0, 0, 0);
                
                Eigen::Vector3d vec_to_leader = all_states[active_leader_id].x - pos;
                double angle_to_leader = std::atan2(vec_to_leader.y(), vec_to_leader.x());
                double angle_diff = angle_to_leader - my_state.heading;
                
                while (angle_diff > M_PI) angle_diff -= 2.0 * M_PI;
                while (angle_diff < -M_PI) angle_diff += 2.0 * M_PI;
                
                if (std::abs(angle_diff) <= fov_limit_rad) {
                    v_leader = vec_to_leader;
                    if (v_leader.norm() > 0.0001) v_leader.normalize();
                }
                
                Eigen::Vector3d v_alignment = all_states[active_leader_id].v;
                if (v_alignment.norm() > 0.0001) v_alignment.normalize();
                
                Eigen::Vector3d blend = v_leader * param_b_ + v_repulsion * rep_weight_ + v_alignment * param_a_;
                if (blend.norm() > 0.0001) blend.normalize();
                
                desired_vel = blend * max_speed_;
            }
        }

        return desired_vel;
    }

 private:
    size_t id_;
    int mode_;
    double rep_radius_;
    double rep_weight_;
    double param_a_;
    double param_b_;
    double param_c_;
    double interaction_radius_;
    double target_x_;
    double target_y_;
    double target_z_;
    double max_speed_;
    
    bool leader_initialized_;
    size_t current_dynamic_leader_;
};