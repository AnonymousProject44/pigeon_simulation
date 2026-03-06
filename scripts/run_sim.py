import pathlib
import subprocess
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Wedge
import numpy as np
import pandas as pd

clip = "015"

clip_offsets = {
    "006": (0.0, -50.0),
    "011": (0.0, 50.0),
    "015": (-150.0, -100.0),
    "021": (-75.0, -100.0)
}

offset = 30

OPTIMIZATION_MODE = 1
optimization_name = "_wrt_real" if OPTIMIZATION_MODE == 2 else ""

if clip in clip_offsets:
    OFFSET_X, OFFSET_Y = clip_offsets[clip]
else:
    OFFSET_X, OFFSET_Y = 0.0, 0.0
    
PATH_TO_EXECUTABLE = (
    pathlib.Path(__file__).parent / ".." / "install" / "bin" / "simulator"
)

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
        start_positions.append([f_row['px'].values[0], f_row['py'].values[0]] if not f_row.empty else b_data[['px', 'py']].values[0])
            
        v_row = b_data[b_data['step'] == velocity_step]
        if not v_row.empty:
            start_velocities.append([v_row['vx'].values[0], v_row['vy'].values[0]])
        else:
            start_velocities.append([b_data['vx'].values[0], b_data['vy'].values[0]])
                
    return waypoints, mean_velocities, num_birds, np.array(start_positions), np.array(start_velocities), bird_trajectories

def parse_output(data: str):
    trajectories = []
    target_trajectory = []
    agents_count = -1
    for i, line in enumerate(data.splitlines()):
        if i == 0:
            agents_count = int(line.strip())
            for _ in range(agents_count):
                trajectories.append([])
            continue
        values = list(map(float, line.strip().split()))
        if len(values) < 3 * agents_count + 3:
            continue
        for j in range(agents_count):
            trajectories[j].append((values[3 * j], values[3 * j + 1], values[3 * j + 2]))
        target_trajectory.append((values[-3], values[-2], values[-1]))
    return trajectories, target_trajectory

def calculate_metrics(trajectories, original_waypoints):
    traj_array = np.array(trajectories)
    
    num_frames = traj_array.shape[1]
    
    com_sim = np.mean(traj_array[:, :, :2], axis=0)
    ref_path = np.array(original_waypoints)[:, :2]
    
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
    avg_path_distance = np.mean(min_dists)
    
    dispersion = np.mean(np.linalg.norm(traj_array[:, :, :2] - com_sim[np.newaxis, :, :], axis=2))
    
    return avg_path_distance, num_frames, dispersion

def run_simulator(controller_parameters: list[float]):
    cmd = [str(PATH_TO_EXECUTABLE)]
    for val in controller_parameters:
        cmd.append(str(val))
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode())
    return parse_output(result.stdout.decode())

def animate_trajectories(res_cons, target_cons, res_fixed, target_fixed, res_dyn, target_dyn, waypoints, initial_headings, extension_start_idx):
    max_frames = max(len(res_cons[0]), len(res_fixed[0]), len(res_dyn[0]))
    num_agents = len(res_cons)
    
    for res in [res_cons, res_fixed, res_dyn]:
        for agent_traj in res:
            while len(agent_traj) < max_frames:
                agent_traj.append(agent_traj[-1])
                
    while len(target_cons) < max_frames: target_cons.append(target_cons[-1])
    while len(target_fixed) < max_frames: target_fixed.append(target_fixed[-1])
    while len(target_dyn) < max_frames: target_dyn.append(target_dyn[-1])
    
    orange_dot = waypoints[extension_start_idx - 1][:2]
    wp_prev = waypoints[extension_start_idx - 2][:2]
    finish_dir = orange_dot - wp_prev
    finish_dir = finish_dir / np.linalg.norm(finish_dir)
    
    def calculate_cutoff(res_array):
        res_np = np.array(res_array)[:, :, :2]
        com_per_frame = np.mean(res_np, axis=0)
        
        com_proj = np.sum((com_per_frame - orange_dot) * finish_dir, axis=1)
        dist_com = np.linalg.norm(com_per_frame - orange_dot, axis=1)
        com_passed = (com_proj > 0) & (dist_com < 50.0)
        
        agent_proj = np.sum((res_np - orange_dot) * finish_dir, axis=2)
        dist_agents = np.linalg.norm(res_np - orange_dot, axis=2)
        agent_passed = (agent_proj > 0) & (dist_agents < 50.0)
        any_agent_passed = np.any(agent_passed, axis=0)
        
        if np.any(com_passed) or np.any(any_agent_passed):
            passed_frames = np.where(com_passed | any_agent_passed)[0]
            return passed_frames[0]
        else:
            return np.argmin(dist_com)
        
    cutoff_c = calculate_cutoff(res_cons)
    cutoff_f = calculate_cutoff(res_fixed)
    cutoff_d = calculate_cutoff(res_dyn)
    
    original_waypoints = waypoints[:extension_start_idx]
    traj_cons = np.array(res_cons)[:, :cutoff_c + 1, :].tolist()
    pd_1, t_1, disp_1 = calculate_metrics(traj_cons, original_waypoints)
    
    traj_fixed = np.array(res_fixed)[:, :cutoff_f + 1, :].tolist()
    pd_2, t_2, disp_2 = calculate_metrics(traj_fixed, original_waypoints)
    
    traj_dyn = np.array(res_dyn)[:, :cutoff_d + 1, :].tolist()
    pd_3, t_3, disp_3 = calculate_metrics(traj_dyn, original_waypoints)
    
    real_traj_array = np.array(bird_trajectories) # Shape: [agents, frames, 2]
    real_com = np.mean(real_traj_array, axis=0)
    real_dispersion = np.mean(np.linalg.norm(real_traj_array - real_com[np.newaxis, :, :], axis=2))

    print(f"Real Flock Dispersion: {real_dispersion:.2f}") # New output
    print(f"\nMetrics for Clip {clip}:")
    print(f"Mode 1: Dist. to path: {pd_1:.2f}, Time (frames): {t_1}, Dispersion: {disp_1:.2f}")
    print(f"Mode 2: Dist. to path: {pd_2:.2f}, Time (frames): {t_2}, Dispersion: {disp_2:.2f}")
    print(f"Mode 3: Dist. to path: {pd_3:.2f}, Time (frames): {t_3}, Dispersion: {disp_3:.2f}\n")

    print("Simulations completed. Generating animation...")

    max_frames = max(cutoff_c, cutoff_f, cutoff_d) + 1
    
    all_x, all_y = [], []
    for res in [res_cons, res_fixed, res_dyn]:
        for agent_traj in res:
            all_x.extend([p[0] for p in agent_traj[:max_frames]])
            all_y.extend([p[1] for p in agent_traj[:max_frames]])
    all_x.extend(waypoints[:, 0]); all_y.extend(waypoints[:, 1])
    
    min_x, max_x = np.min(all_x), np.max(all_x)
    min_y, max_y = np.min(all_y), np.max(all_y)
    pad = 15.0
                
    fig, (ax_cons, ax_fixed, ax_dyn) = plt.subplots(1, 3, figsize=(20, 6))
    
    scat_cons = ax_cons.scatter([], [], c='blue', s=20, zorder=4)
    scat_fixed = ax_fixed.scatter([], [], c='blue', s=20, zorder=4)
    leader_fixed = ax_fixed.scatter([], [], c='green', s=80, zorder=5)
    scat_dyn = ax_dyn.scatter([], [], c='blue', s=20, zorder=4)
    leader_dyn = ax_dyn.scatter([], [], c='green', s=80, zorder=5)
    
    target_scat_cons = ax_cons.scatter([], [], c='red', s=120, marker='X', zorder=6)
    target_scat_fixed = ax_fixed.scatter([], [], c='red', s=120, marker='X', zorder=6)
    target_scat_dyn = ax_dyn.scatter([], [], c='red', s=120, marker='X', zorder=6)
    line_dyn, = ax_dyn.plot([], [], 'r--', alpha=0.5)

    com_scat_cons = ax_cons.scatter([], [], c='magenta', s=150, marker='*', zorder=7)
    com_scat_fixed = ax_fixed.scatter([], [], c='magenta', s=150, marker='*', zorder=7)
    com_scat_dyn = ax_dyn.scatter([], [], c='magenta', s=150, marker='*', zorder=7)

    init_zeros = np.zeros(num_agents)
    quiver_c = ax_cons.quiver(init_zeros, init_zeros, init_zeros, init_zeros, color='black', scale=25, width=0.005, zorder=3)
    quiver_f = ax_fixed.quiver(init_zeros, init_zeros, init_zeros, init_zeros, color='black', scale=25, width=0.005, zorder=3)
    quiver_d = ax_dyn.quiver(init_zeros, init_zeros, init_zeros, init_zeros, color='black', scale=25, width=0.005, zorder=3)

    fov_angle, half_fov, fov_radius = 316.4, 316.4/2.0, 6.0
    wedges_c = [Wedge((0,0), fov_radius, 0, 0, color='blue', alpha=0.1, zorder=1) for _ in range(num_agents)]
    wedges_f = [Wedge((0,0), fov_radius, 0, 0, color='blue', alpha=0.1, zorder=1) for _ in range(num_agents)]
    wedges_d = [Wedge((0,0), fov_radius, 0, 0, color='blue', alpha=0.1, zorder=1) for _ in range(num_agents)]

    for w in wedges_c: ax_cons.add_patch(w)
    for w in wedges_f: ax_fixed.add_patch(w)
    for w in wedges_d: ax_dyn.add_patch(w)
    
    for ax, title in zip([ax_cons, ax_fixed, ax_dyn], ['Consensus', 'Fixed Leader', 'Dynamic Leader']):
        ax.set_xlim(min_x - pad, max_x + pad)
        ax.set_ylim(min_y - pad, max_y + pad)
        ax.set_title(title)
        ax.set_aspect('equal', adjustable='box') 
        ax.scatter(waypoints[:extension_start_idx, 0], waypoints[:extension_start_idx, 1], c='gray', s=60, marker='o', alpha=0.5, zorder=2)
        ax.scatter(waypoints[extension_start_idx:, 0], waypoints[extension_start_idx:, 1], c='gray', s=20, marker='o', alpha=0.3, zorder=2)
        ax.scatter(orange_dot[0], orange_dot[1], c='orange', s=180, marker='o', zorder=8, edgecolors='black')
        
    h_c, h_f, h_d = initial_headings.copy(), initial_headings.copy(), initial_headings.copy()
    active_dyn_leader = 0

    def update_headings(pos, prev_pos, curr_h):
        for i in range(len(pos)):
            dx, dy = pos[i][0] - prev_pos[i][0], pos[i][1] - prev_pos[i][1]
            if np.hypot(dx, dy) > 0.05: curr_h[i] = np.arctan2(dy, dx)
        return curr_h

    def update_wedges(wedges, positions, curr_h):
        for i, w in enumerate(wedges):
            w.set_center(positions[i]); deg = np.degrees(curr_h[i])
            w.set_theta1(deg - half_fov); w.set_theta2(deg + half_fov)
        
    def update(frame):
        nonlocal active_dyn_leader, h_c, h_f, h_d
        
        f_c = min(frame, cutoff_c)
        p_c = np.array([res_cons[i][f_c][:2] for i in range(num_agents)])
        h_c = update_headings(p_c, np.array([res_cons[i][max(0, f_c-1)][:2] for i in range(num_agents)]), h_c)
        scat_cons.set_offsets(p_c); quiver_c.set_offsets(p_c); quiver_c.set_UVC(np.cos(h_c), np.sin(h_c))
        update_wedges(wedges_c, p_c, h_c); com_scat_cons.set_offsets([np.mean(p_c, axis=0)])
        target_scat_cons.set_offsets([target_cons[f_c][:2]])
        
        f_f = min(frame, cutoff_f)
        p_f = np.array([res_fixed[i][f_f][:2] for i in range(num_agents)])
        h_f = update_headings(p_f, np.array([res_fixed[i][max(0, f_f-1)][:2] for i in range(num_agents)]), h_f)
        scat_fixed.set_offsets(p_f); leader_fixed.set_offsets(p_f[0]); quiver_f.set_offsets(p_f)
        quiver_f.set_UVC(np.cos(h_f), np.sin(h_f))
        update_wedges(wedges_f, p_f, h_f); com_scat_fixed.set_offsets([np.mean(p_f, axis=0)])
        wedges_f[0].set_color('green'); wedges_f[0].set_alpha(0.2)
        target_scat_fixed.set_offsets([target_fixed[f_f][:2]])
        
        f_d = min(frame, cutoff_d)
        p_d = np.array([res_dyn[i][f_d][:2] for i in range(num_agents)])
        h_d = update_headings(p_d, np.array([res_dyn[i][max(0, f_d-1)][:2] for i in range(num_agents)]), h_d)
        
        if f_d == frame:
            dists = [np.linalg.norm(np.array(p) - target_dyn[f_d][:2]) for p in p_d]
            cand = np.argmin(dists)
            if (dists[active_dyn_leader] - dists[cand]) > 2.0:
                wedges_d[active_dyn_leader].set_color('blue'); wedges_d[active_dyn_leader].set_alpha(0.1)
                active_dyn_leader = cand
                
        scat_dyn.set_offsets(p_d); leader_dyn.set_offsets(p_d[active_dyn_leader]); quiver_d.set_offsets(p_d)
        quiver_d.set_UVC(np.cos(h_d), np.sin(h_d))
        update_wedges(wedges_d, p_d, h_d); com_scat_dyn.set_offsets([np.mean(p_d, axis=0)])
        wedges_d[active_dyn_leader].set_color('green'); wedges_d[active_dyn_leader].set_alpha(0.2)
        line_dyn.set_data([p_d[active_dyn_leader][0], target_dyn[f_d][0]], [p_d[active_dyn_leader][1], target_dyn[f_d][1]])
        target_scat_dyn.set_offsets([target_dyn[f_d][:2]])
        
        return (scat_cons, target_scat_cons, com_scat_cons, scat_fixed, leader_fixed, target_scat_fixed, com_scat_fixed, scat_dyn, leader_dyn, target_scat_dyn, line_dyn, com_scat_dyn, quiver_c, quiver_f, quiver_d) + tuple(wedges_c) + tuple(wedges_f) + tuple(wedges_d)
        
    skip_step = 1
    animation_frames = list(range(0, max_frames, skip_step))
    adjusted_fps = max(10.0 * (5.0 / skip_step), len(animation_frames) / 5.0)
    ani = animation.FuncAnimation(fig, update, frames=animation_frames, interval=int(1000 / adjusted_fps), blit=True)
    ani.save(f'swarm_waypoints_{clip}.mp4', writer='ffmpeg', fps=adjusted_fps)
    print("Animation saved successfully.")

if __name__ == "__main__":
    csv_path = f'../config/reference_trajectory_{clip}.csv'
    waypoints, velocities, drone_count, start_pos, start_vel, bird_trajectories = get_data_from_csv(csv_path)

    shifted_init = start_pos.copy()
    shifted_init[:, 0] += OFFSET_X
    shifted_init[:, 1] += OFFSET_Y

    current_candidate = np.argmin([np.linalg.norm(p - waypoints[0][:2]) for p in shifted_init])
    
    for _ in range(20):
        leader_v = start_vel[current_candidate]
        leader_heading = np.arctan2(leader_v[1], leader_v[0])
        heading_dir = np.array([np.cos(leader_heading), np.sin(leader_heading)])
        
        target_leader_pos = waypoints[0][:2] - offset * heading_dir
        alignment_vec = target_leader_pos - shifted_init[current_candidate]
        
        test_shifted = shifted_init + alignment_vec
        new_closest = np.argmin([np.linalg.norm(p - waypoints[0][:2]) for p in test_shifted])
        
        if new_closest == current_candidate:
            shifted_init = test_shifted
            closest_idx = current_candidate
            break
        current_candidate = new_closest
    else:
        shifted_init = test_shifted
        closest_idx = current_candidate

    shifted_init[[0, closest_idx]] = shifted_init[[closest_idx, 0]]
    start_vel[[0, closest_idx]] = start_vel[[closest_idx, 0]]
    bird_trajectories[0], bird_trajectories[closest_idx] = bird_trajectories[closest_idx], bird_trajectories[0]

    extension_start_idx = len(waypoints)

    if len(waypoints) >= 2:
        unit_dir = (waypoints[-1][:2] - waypoints[-2][:2])
        unit_dir = unit_dir / np.linalg.norm(unit_dir)
        ext_dist = 50.0
        for i in range(1, 11):
            new_end = waypoints[extension_start_idx-1].copy()
            new_end[:2] = new_end[:2] + unit_dir * (ext_dist * i / 10.0)
            waypoints = np.vstack([waypoints, new_end])
            velocities = np.vstack([velocities, velocities[extension_start_idx-1]])
    
    flat_starts, initial_headings = [], []
    for p, v in zip(shifted_init, start_vel):
        sx, sy = p[0], p[1] 
        h, vn = np.arctan2(v[1], v[0]), np.linalg.norm(v)
        flat_starts.extend([sx, sy, 10.0, h, vn]); initial_headings.append(h)
        
    flat_wps = []
    for wp, tv in zip(waypoints, velocities):
        flat_wps.extend([wp[0], wp[1], 10.0, tv[0], tv[1], 0.0])
        
    base_env = [drone_count, len(waypoints)] + flat_starts + flat_wps

    j_path = pathlib.Path(__file__).parent / ".." / "config" / f"optimized_parameters{optimization_name}.json"
    
    def extract_params(key, mode_idx):
        base_params = [6.5, 9.2, 5.9, 1.0, 3.4, 3.1]
        if j_path.exists():
            with open(j_path, "r") as f:
                s = json.load(f)
                if key in s: 
                    o = s[key]
                    base_params = [o.get("max_speed", 6.0), o.get("rep_radius", 12.0), o.get("rep_weight", 4.0), o.get("param_a", 0.8), o.get("param_b", 1.2), o.get("param_c", 1.5)]
                    
                    if mode_idx == 0:
                        base_params.append(o.get("interaction_radius", 25.0))
                    else:
                        base_params.append(25.0)
                    return base_params
                    
        base_params.append(25.0)
        return base_params

    w_c = extract_params("Option_1_Consensus_Leaderless", 0)
    w_f = extract_params("Option_2_Fixed_Leader", 1)
    w_d = extract_params("Option_3_Dynamic_Leader", 2)

    res_cons, target_cons = run_simulator([0] + w_c + base_env)
    res_fixed, target_fixed = run_simulator([1] + w_f + base_env)
    res_dyn, target_dyn = run_simulator([2] + w_d + base_env)
    
    animate_trajectories(res_cons, target_cons, res_fixed, target_fixed, res_dyn, target_dyn, waypoints, np.array(initial_headings), extension_start_idx)