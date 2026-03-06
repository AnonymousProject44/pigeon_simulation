import pathlib
import subprocess
import json
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import cma

from multiprocessing import Pool, cpu_count

PATH_TO_EXECUTABLE = (
    pathlib.Path(__file__).parent / ".." / "install" / "bin" / "simulator"
)

clip_offsets = {
    "006": (0.0, -50.0),
    "011": (0.0, 50.0),
    "015": (-150.0, -100.0),
    "021": (-75.0, -100.0)
}

ALIGNMENT_OFFSET = 30.0

clips_preloaded = {}

def get_data_from_csv(filename, sampling_divisor=5):
    df = pd.read_csv(filename)
    bird_ids = sorted(df['bird_id'].unique())
    num_birds = len(bird_ids)
    
    mean_data = df.groupby('step')[['px', 'py', 'vx', 'vy']].mean().reset_index()
    mean_data = mean_data.sort_values('step')
    
    indices = np.linspace(0, len(mean_data) - 1, int(len(mean_data)/sampling_divisor), dtype=int)
    sampled_mean_data = mean_data.iloc[indices]
    
    waypoints = sampled_mean_data[['px', 'py']].values
    
    dt = 1.0 / 60.0
    computed_velocities = []
    for i in range(len(waypoints) - 1):
        step_diff = sampled_mean_data['step'].iloc[i+1] - sampled_mean_data['step'].iloc[i]
        time_sec = step_diff * dt
        if time_sec > 0:
            v_avg = (waypoints[i+1] - waypoints[i]) / time_sec
        else:
            v_avg = np.array([0.0, 0.0])
        computed_velocities.append(v_avg)
        
    if len(computed_velocities) > 0:
        computed_velocities.append(computed_velocities[-1])
    else:
        computed_velocities.append(np.array([0.0, 0.0]))
        
    mean_velocities = np.array(computed_velocities)
    
    unique_steps = sorted(df['step'].unique())
    first_step = unique_steps[0]
    
    velocity_step_index = 50 if len(unique_steps) > 50 else len(unique_steps) - 1
    velocity_step = unique_steps[velocity_step_index]
    
    start_positions = []
    start_velocities = []
    bird_trajectories = []
    
    for b_id in bird_ids:
        b_data = df[df['bird_id'] == b_id].sort_values('step')
        bird_trajectories.append(b_data[['px', 'py']].values)
        
        f_row = b_data[b_data['step'] == first_step]
        if not f_row.empty:
            start_positions.append([f_row['px'].values[0], f_row['py'].values[0]])
        else:
            start_positions.append(b_data[['px', 'py']].values[0])
            
        v_row = b_data[b_data['step'] == velocity_step]
        if not v_row.empty:
            start_velocities.append([v_row['vx'].values[0], v_row['vy'].values[0]])
        else:
            start_velocities.append([b_data['vx'].values[0], b_data['vy'].values[0]])
            
    return waypoints, mean_velocities, num_birds, np.array(start_positions), np.array(start_velocities), bird_trajectories

def parse_output(data: str):
    trajectories = []
    agents_count = -1
    for idx, line in enumerate(data.splitlines()):
        if idx == 0:
            agents_count = int(line.strip())
            for _ in range(agents_count):
                trajectories.append([])
            continue
        values = list(map(float, line.strip().split()))
        
        if len(values) < agents_count + agents_count + agents_count + 3:
            continue
        for agent_idx in range(agents_count):
            trajectories[agent_idx].append((values[agent_idx + agent_idx + agent_idx], values[agent_idx + agent_idx + agent_idx + 1], values[agent_idx + agent_idx + agent_idx + 2]))
    return trajectories

def run_simulator(controller_parameters, base_environment):
    cmd = [str(PATH_TO_EXECUTABLE)]
    for val in controller_parameters:
        cmd.append(str(val))
    for val in base_environment:
        cmd.append(str(val))
        
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        return None
    return parse_output(result.stdout.decode())

def calculate_fitness(trajectories, waypoints):
    traj_array = np.array(trajectories)
    num_steps = traj_array.shape[1]
    
    com_sim = np.mean(traj_array[:, :, :2], axis=0)
    ref_path = np.array(waypoints)[:, :2]
    
    P = com_sim[:, np.newaxis, :]
    A = ref_path[:-1][np.newaxis, :, :]
    B = ref_path[1:][np.newaxis, :, :]
    
    AB = B - A
    AP = P - A
    
    dot_AP_AB = np.sum(AP * AB, axis=2)
    dot_AB_AB = np.sum(AB * AB, axis=2)
    
    t = dot_AP_AB / np.where(dot_AB_AB == 0, 1e-8, dot_AB_AB)
    t = np.clip(t, 0.0, 1.0)
    
    C = A + t[:, :, np.newaxis] * AB
    dists = np.linalg.norm(P - C, axis=2)
    
    min_dists = np.min(dists, axis=1)
    com_tracking_cost = np.mean(min_dists)
    
    cohesion_cost = np.mean(np.linalg.norm(traj_array[:, :, :2] - com_sim[np.newaxis, :, :], axis=2))
    
    path_diffs = np.diff(traj_array[:, :, :2], axis=1)
    vel_diffs = np.diff(path_diffs, axis=1)
    jitter_cost = np.mean(np.sum(np.linalg.norm(vel_diffs, axis=2), axis=1))
    
    collision_cost = 0.0
    collision_threshold = 1.0 
    
    for step in range(num_steps):
        step_positions = traj_array[:, step, :2]
        diffs = step_positions[:, np.newaxis, :] - step_positions[np.newaxis, :, :]
        dists_col = np.linalg.norm(diffs, axis=-1)
        np.fill_diagonal(dists_col, np.inf) 
        
        num_collisions = np.sum(dists_col < collision_threshold) / 2.0
        collision_cost += num_collisions * 1000.0 

    time_cost = num_steps * 0.1

    total_cost = 15.0 * com_tracking_cost + 5.0 * jitter_cost + 2.0 * cohesion_cost + time_cost + collision_cost
    
    return total_cost

def objective_function(params, mode, clips_data):
    if mode == 0:
        max_speed, rep_radius, rep_weight, param_a, param_b, param_c, interaction_radius = params
        if max_speed < 3.0 or rep_radius < 1.0 or rep_weight < 0.0 or param_a < 0.0 or param_b < 0.0 or param_c < 0.0 or interaction_radius < 1.0:
            return 100000.0
        controller_weights = [mode, max_speed, rep_radius, rep_weight, param_a, param_b, param_c, interaction_radius]
    else:
        max_speed, rep_radius, rep_weight, param_a, param_b, param_c = params
        if max_speed < 3.0 or rep_radius < 1.0 or rep_weight < 0.0 or param_a < 0.0 or param_b < 0.0 or param_c < 0.0:
            return 100000.0
        controller_weights = [mode, max_speed, rep_radius, rep_weight, param_a, param_b, param_c, 25.0]

    total_fitness = 0.0
    
    for clip_id, clip_offset in clips_data.items():
        offset_x, offset_y = clip_offset
        
        wp_orig, vel_orig, drone_count, sp_orig, sv_orig, bt_orig = clips_preloaded[clip_id]
        
        waypoints = wp_orig.copy()
        target_velocities = vel_orig.copy()
        start_positions = sp_orig.copy()
        start_velocities = sv_orig.copy()
        
        shifted_positions = start_positions.copy()
        shifted_positions[:, 0] += offset_x
        shifted_positions[:, 1] += offset_y
        
        current_candidate = np.argmin([np.linalg.norm(p - waypoints[0][:2]) for p in shifted_positions])
        
        for _ in range(20):
            leader_v = start_velocities[current_candidate]
            leader_heading = np.arctan2(leader_v[1], leader_v[0])
            heading_dir = np.array([np.cos(leader_heading), np.sin(leader_heading)])
            
            target_leader_pos = waypoints[0][:2] - ALIGNMENT_OFFSET * heading_dir
            alignment_vec = target_leader_pos - shifted_positions[current_candidate]
            
            test_shifted = shifted_positions + alignment_vec
            new_closest = np.argmin([np.linalg.norm(p - waypoints[0][:2]) for p in test_shifted])
            
            if new_closest == current_candidate:
                shifted_positions = test_shifted
                closest_idx = current_candidate
                break
            current_candidate = new_closest
        else:
            shifted_positions = test_shifted
            closest_idx = current_candidate
            
        shifted_positions[[0, closest_idx]] = shifted_positions[[closest_idx, 0]]
        start_velocities[[0, closest_idx]] = start_velocities[[closest_idx, 0]]
        
        extension_start_idx = len(waypoints)
        
        if len(waypoints) >= 2:
            unit_dir = (waypoints[-1][:2] - waypoints[-2][:2])
            unit_dir = unit_dir / np.linalg.norm(unit_dir)
            ext_dist = 50.0
            for i in range(1, 11):
                new_end = waypoints[extension_start_idx-1].copy()
                new_end[:2] = new_end[:2] + unit_dir * (ext_dist * i / 10.0)
                waypoints = np.vstack([waypoints, new_end])
                target_velocities = np.vstack([target_velocities, target_velocities[extension_start_idx-1]])
        
        flat_starts = []
        for p, v in zip(shifted_positions, start_velocities):
            start_x = p[0]
            start_y = p[1]
            
            heading = np.arctan2(v[1], v[0])
            v_norm = np.linalg.norm(v)
            
            flat_starts.extend([start_x, start_y, 10.0, heading, v_norm])
            
        flat_waypoints = []
        for wp, t_vel in zip(waypoints, target_velocities):
            flat_waypoints.extend([wp[0], wp[1], 10.0, t_vel[0], t_vel[1], 0.0])
            
        base_environment = [drone_count, len(waypoints)] + flat_starts + flat_waypoints
        
        trajectories = run_simulator(controller_weights, base_environment)
        
        if trajectories is None or len(trajectories) == 0 or len(trajectories[0]) == 0:
            return 100000.0 
            
        traj_array = np.array(trajectories)
        
        orange_dot = waypoints[extension_start_idx - 1][:2]
        wp_prev = waypoints[extension_start_idx - 2][:2]
        finish_dir = orange_dot - wp_prev
        finish_dir = finish_dir / np.linalg.norm(finish_dir)
        
        com_per_frame = np.mean(traj_array[:, :, :2], axis=0)
        com_proj = np.sum((com_per_frame - orange_dot) * finish_dir, axis=1)
        dist_com = np.linalg.norm(com_per_frame - orange_dot, axis=1)
        com_passed = (com_proj > 0) & (dist_com < 50.0)
        
        agent_proj = np.sum((traj_array[:, :, :2] - orange_dot) * finish_dir, axis=2)
        dist_agents = np.linalg.norm(traj_array[:, :, :2] - orange_dot, axis=2)
        agent_passed = (agent_proj > 0) & (dist_agents < 50.0)
        any_agent_passed = np.any(agent_passed, axis=0)
        
        if np.any(com_passed) or np.any(any_agent_passed):
            passed_frames = np.where(com_passed | any_agent_passed)[0]
            cutoff_frame = passed_frames[0]
        else:
            cutoff_frame = np.argmin(dist_com)
        
        truncated_trajectories = traj_array[:, :cutoff_frame + 1, :].tolist()
        original_waypoints = waypoints[:extension_start_idx]
            
        fitness = calculate_fitness(truncated_trajectories, original_waypoints)
        total_fitness += fitness
        
    return total_fitness / len(clips_data)

if __name__ == "__main__":
    
    for clip_id in clip_offsets:
        csv_file = f'../config/reference_trajectory_{clip_id}.csv'
        clips_preloaded[clip_id] = get_data_from_csv(csv_file)

    print("Select which mode to optimize:")
    print("0 - All modes")
    print("1 - Consensus Leaderless")
    print("2 - Fixed Leader")
    print("3 - Dynamic Leader")
    
    try:
        choice = int(input("Enter your choice: ").strip())
    except ValueError:
        choice = 0

    if choice == 1:
        modes = [0]
        mode_names = ["Consensus Leaderless"]
        json_filename = "optimized_parameters_consensus.json"
    elif choice == 2:
        modes = [1]
        mode_names = ["Fixed Leader"]
        json_filename = "optimized_parameters_fixed.json"
    elif choice == 3:
        modes = [2]
        mode_names = ["Dynamic Leader"]
        json_filename = "optimized_parameters_dynamic.json"
    else:
        modes = [0, 1, 2]
        mode_names = ["Consensus Leaderless", "Fixed Leader", "Dynamic Leader"]
        json_filename = "optimized_parameters.json"
    
    all_best_params = []
    all_best_errors = []
    
    initial_std = 1.5
    
    for target_mode, mode_name in zip(modes, mode_names):
        print(f"\nStarting CMA-ES optimization for {mode_name}")
        
        if target_mode == 0:
            current_initial_params = [6.0, 12.0, 4.0, 0.8, 1.2, 1.5, 25.0]
        else:
            current_initial_params = [6.0, 12.0, 4.0, 0.8, 1.2, 1.5]
            
        es = cma.CMAEvolutionStrategy(current_initial_params, initial_std)
        generation = 0
        
        while not es.stop():
            solutions = es.ask()
            
            with Pool(cpu_count()) as pool:
                fitness_values = pool.starmap(
                    objective_function,
                    [(sol, target_mode, clip_offsets) for sol in solutions]
                )
            es.tell(solutions, fitness_values)
            es.disp()
            
            if generation > 0 and generation % 20 == 0:
                current_best = es.result.xbest
                print("Current best parameters at generation", generation)
                print("max_speed:", current_best[0])
                print("rep_radius:", current_best[1])
                print("rep_weight:", current_best[2])
                print("param_a:", current_best[3])
                print("param_b:", current_best[4])
                print("param_c:", current_best[5])
                if target_mode == 0:
                    print("interaction_radius:", current_best[6])
                
            generation += 1
            
        print(f"Optimization finished for {mode_name}.")
        all_best_params.append(es.result.xbest)
        all_best_errors.append(es.result.fbest)
        
    print("\nOptimization complete. Final Best Parameters:")
    
    output_path = pathlib.Path(__file__).parent / ".." / "config" / json_filename
    results_dict = {}
    
    if output_path.exists():
        with open(output_path, "r") as json_file:
            try:
                results_dict = json.load(json_file)
            except json.JSONDecodeError:
                pass
    
    for target_mode, mode_name, best_params, best_error in zip(modes, mode_names, all_best_params, all_best_errors):
        print(f"\n{mode_name}:")
        print("max_speed:", best_params[0])
        print("rep_radius:", best_params[1])
        print("rep_weight:", best_params[2])
        print("param_a:", best_params[3])
        print("param_b:", best_params[4])
        print("param_c:", best_params[5])
        if target_mode == 0:
            print("interaction_radius:", best_params[6])
        print("final_error:", best_error)
        
        dict_key = f"Option_{target_mode + 1}_{mode_name.replace(' ', '_')}"
        results_dict[dict_key] = {
            "max_speed": float(best_params[0]),
            "rep_radius": float(best_params[1]),
            "rep_weight": float(best_params[2]),
            "param_a": float(best_params[3]),
            "param_b": float(best_params[4]),
            "param_c": float(best_params[5]),
            "final_error": float(best_error)
        }
        if target_mode == 0:
            results_dict[dict_key]["interaction_radius"] = float(best_params[6])
        
    with open(output_path, "w") as json_file:
        json.dump(results_dict, json_file, indent=4)
        
    print(f"\nAll final parameters and errors have been successfully saved to {output_path}")