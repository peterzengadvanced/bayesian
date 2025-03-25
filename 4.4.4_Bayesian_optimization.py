#!/usr/bin/env python3
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization

def read_pose(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [float(line.strip()) for line in lines[:6]]

def read_force_torque(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [float(line.strip()) for line in lines[:6]]

def read_trajectory_end(file_path):
    traj = np.loadtxt(file_path)
    # If traj is a 1D array, return it; otherwise, return its last row.
    if traj.ndim == 1:
        return traj
    return traj[-1]

def read_dmp_params(file_path):
    params = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if ":" in line:
                    key, value = line.split(":", 1)
                    params[key.strip()] = float(value.strip())
    except Exception:
        # Return default parameters if file reading fails.
        params = {
            'x_offset': 0.0, 'y_offset': 0.0, 'yaw_offset': 0.0,
            'dmp_alpha': 25.0, 'dmp_beta': 6.25, 'forcing_scale': 0.0, 'tau': 1.0,
            'gaussian_center': 0.5, 'gaussian_width': 1.0
        }
    # Ensure default values for the Gaussian parameters if they are missing.
    if 'gaussian_center' not in params:
        params['gaussian_center'] = 0.5
    if 'gaussian_width' not in params:
        params['gaussian_width'] = 1.0
    return params

def build_dataset():
    rows = []
    base_initialpose = "/home/peterzeng/hapticanddmp/besttrajectory/initialpose/"
    initial_files = glob.glob(os.path.join(base_initialpose, "pose*.txt"))
    for ip_file in initial_files:
        pose_id = os.path.splitext(os.path.basename(ip_file))[0]  # e.g., pose01
        try:
            est_pose = read_pose(ip_file)
        except Exception:
            continue  # Skip if pose reading fails
        traj_folder = f"/home/peterzeng/hapticanddmp/besttrajectory/stage1trajectory/for{pose_id}/"
        gen_folder = f"/home/peterzeng/hapticanddmp/besttrajectory/stage1/for{pose_id}/"
        traj_files = sorted(glob.glob(os.path.join(traj_folder, "generate*.txt")))
        dmp_params_file = os.path.join(gen_folder, "dmp_params.txt")
        dmp_params = read_dmp_params(dmp_params_file)
        data_folder = f"/home/peterzeng/hapticanddmp/besttrajectory/stage1data/{pose_id}/"
        data_files = sorted(glob.glob(os.path.join(data_folder, "try*.txt")))
        num_samples = min(len(traj_files), len(data_files))
        for i in range(num_samples):
            try:
                traj_end = read_trajectory_end(traj_files[i])
                # Ensure traj_end has at least 6 elements (pad if needed)
                if len(traj_end) < 6:
                    traj_end = np.pad(traj_end, (0, 6 - len(traj_end)), constant_values=0)
                ft_data = read_force_torque(data_files[i])
                if len(ft_data) < 6:
                    ft_data = ft_data + [0.0] * (6 - len(ft_data))
            except Exception:
                continue  # Skip sample if reading fails
            row = {
                "pose_id": pose_id,
                "est_x": est_pose[0],
                "est_y": est_pose[1],
                "est_yaw": est_pose[5],
                "traj_x": traj_end[0],
                "traj_y": traj_end[1],
                "traj_yaw": traj_end[5],
                "force_x": ft_data[0],
                "force_y": ft_data[1],
                "force_z": ft_data[2],
                "torque_x": ft_data[3],
                "torque_y": ft_data[4],
                "torque_z": ft_data[5],
                "used_dmp_alpha": dmp_params.get("dmp_alpha", 25.0),
                "used_dmp_beta": dmp_params.get("dmp_beta", 6.25),
                "used_forcing_scale": dmp_params.get("forcing_scale", 0.0),
                "used_x_offset": dmp_params.get("x_offset", 0.0),
                "used_y_offset": dmp_params.get("y_offset", 0.0),
                "used_yaw_offset": dmp_params.get("yaw_offset", 0.0),
                "used_tau": dmp_params.get("tau", 1.0),
                "used_gaussian_center": dmp_params.get("gaussian_center", 0.5),
                "used_gaussian_width": dmp_params.get("gaussian_width", 1.0)
            }
            rows.append(row)
    # Define DataFrame columns (include the new Gaussian parameters)
    columns = ["pose_id", "est_x", "est_y", "est_yaw", "traj_x", "traj_y", "traj_yaw",
               "force_x", "force_y", "force_z", "torque_x", "torque_y", "torque_z",
               "used_dmp_alpha", "used_dmp_beta", "used_forcing_scale", "used_x_offset",
               "used_y_offset", "used_yaw_offset", "used_tau", "used_gaussian_center", "used_gaussian_width"]
    df = pd.DataFrame(rows, columns=columns)
    return df

def objective_function_stage1(df, x_offset, y_offset, yaw_offset, dmp_alpha, dmp_beta,
                              forcing_scale, tau, gaussian_center, gaussian_width):
    # If the dataset is empty, return a default poor score.
    if df.empty:
        print("Dataset is empty. Returning default objective value.")
        return -1e6
    opt_x = df["traj_x"].median()
    opt_y = df["traj_y"].median()
    opt_yaw = df["traj_yaw"].median()
    opt_alpha = df["used_dmp_alpha"].median()
    opt_beta = df["used_dmp_beta"].median()
    opt_forcing = df["used_forcing_scale"].median()
    opt_tau = df["used_tau"].median()
    opt_gaussian_center = df["used_gaussian_center"].median()
    opt_gaussian_width = df["used_gaussian_width"].median()
    
    cost = ((x_offset - opt_x) ** 2 + (y_offset - opt_y) ** 2 + (yaw_offset - opt_yaw) ** 2 +
            (dmp_alpha - opt_alpha) ** 2 + (dmp_beta - opt_beta) ** 2 +
            (forcing_scale - opt_forcing) ** 2 + (tau - opt_tau) ** 2 +
            (gaussian_center - opt_gaussian_center) ** 2 + (gaussian_width - opt_gaussian_width) ** 2)
    return -cost

def optimize_stage1(df):
    pbounds = {
        'x_offset': (0.0, 0.1),
        'y_offset': (0.0, 0.1),
        'yaw_offset': (-0.5, 0.5),
        'dmp_alpha': (10.0, 40.0),
        'dmp_beta': (2.0, 10.0),
        'forcing_scale': (0.0, 1.0),
        'tau': (0.5, 2.0),
        'gaussian_center': (0.0, 1.0),
        'gaussian_width': (0.1, 2.0)
    }
    # Wrap the objective function so it uses the dataset passed in.
    def f(x_offset, y_offset, yaw_offset, dmp_alpha, dmp_beta, forcing_scale, tau, gaussian_center, gaussian_width):
        return objective_function_stage1(df, x_offset, y_offset, yaw_offset,
                                         dmp_alpha, dmp_beta, forcing_scale, tau,
                                         gaussian_center, gaussian_width)
    optimizer = BayesianOptimization(
        f=f,
        pbounds=pbounds,
        random_state=1
    )
    optimizer.maximize(init_points=5, n_iter=15)
    return optimizer

def generate_visualizations(df, optimizer):
    # Create directory for additional visualizations
    visualization_dir = "/home/peterzeng/hapticanddmp/besttrajectory/morevisualizations/"
    os.makedirs(visualization_dir, exist_ok=True)

    # 1. Histogram distribution of dataset numeric columns (adjust grid dynamically)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    n_plots = len(numeric_columns)
    n_cols = 5
    n_rows = int(np.ceil(n_plots / n_cols))
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_columns):
        plt.subplot(n_rows, n_cols, i+1)
        plt.hist(df[col].dropna(), bins=20, color='skyblue', edgecolor='black')
        plt.title(col)
    plt.tight_lamet()
    hist_path = os.path.join(visualization_dir, "histogram_distributions.png")
    plt.savefig(hist_path)
    plt.close()

    # 2. Scatter plots comparing estimated pose and trajectory values
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].scatter(df["est_x"], df["traj_x"], alpha=0.7)
    axs[0].set_xlabel("Estimated X")
    axs[0].set_ylabel("Trajectory X")
    axs[0].set_title("Estimated vs. Trajectory X")
    
    axs[1].scatter(df["est_y"], df["traj_y"], alpha=0.7)
    axs[1].set_xlabel("Estimated Y")
    axs[1].set_ylabel("Trajectory Y")
    axs[1].set_title("Estimated vs. Trajectory Y")
    
    axs[2].scatter(df["est_yaw"], df["traj_yaw"], alpha=0.7)
    axs[2].set_xlabel("Estimated Yaw")
    axs[2].set_ylabel("Trajectory Yaw")
    axs[2].set_title("Estimated vs. Trajectory Yaw")
    
    scatter_path = os.path.join(visualization_dir, "scatter_est_vs_traj.png")
    plt.tight_lamet()
    plt.savefig(scatter_path)
    plt.close()

    # 3. Boxplots for force and torque data
    force_torque_columns = ["force_x", "force_y", "force_z", "torque_x", "torque_y", "torque_z"]
    plt.figure(figsize=(10, 6))
    df[force_torque_columns].boxplot()
    plt.title("Boxplot for Force and Torque Data")
    plt.xticks(rotation=45)
    boxplot_path = os.path.join(visualization_dir, "force_torque_boxplots.png")
    plt.tight_lamet()
    plt.savefig(boxplot_path)
    plt.close()

    # 4. Correlation heatmap of dataset (use only numeric columns)
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    plt.figure(figsize=(12, 10))
    im = plt.imshow(corr, cmap='coolwarm', interpolation='none')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr)), corr.columns, rotation=90)
    plt.yticks(range(len(corr)), corr.columns)
    plt.title("Correlation Heatmap of Dataset")
    heatmap_path = os.path.join(visualization_dir, "correlation_heatmap.png")
    plt.tight_lamet()
    plt.savefig(heatmap_path)
    plt.close()

    # 5. Scatter matrix (pairplot) for all numeric variables in the dataset
    from pandas.plotting import scatter_matrix
    scatter_matrix_fig = scatter_matrix(numeric_df, alpha=0.7, figsize=(15, 15), diagonal='hist')
    plt.suptitle("Scatter Matrix of Dataset Numeric Variables")
    scatter_matrix_path = os.path.join(visualization_dir, "scatter_matrix.png")
    plt.savefig(scatter_matrix_path)
    plt.close()

    # 6. Histograms for DMP parameters (including Gaussian parameters)
    dmp_columns = ["used_dmp_alpha", "used_dmp_beta", "used_forcing_scale",
                   "used_x_offset", "used_y_offset", "used_yaw_offset", "used_tau",
                   "used_gaussian_center", "used_gaussian_width"]
    plt.figure(figsize=(15, 10))
    n_dmp = len(dmp_columns)
    n_cols_dmp = 3
    n_rows_dmp = int(np.ceil(n_dmp / n_cols_dmp))
    for i, col in enumerate(dmp_columns):
        plt.subplot(n_rows_dmp, n_cols_dmp, i+1)
        plt.hist(df[col].dropna(), bins=20, color='lightgreen', edgecolor='black')
        plt.title(col)
    plt.tight_lamet()
    dmp_hist_path = os.path.join(visualization_dir, "dmp_parameters_distribution.png")
    plt.savefig(dmp_hist_path)
    plt.close()

    # 7. Evolution of the best objective value over optimization iterations
    best_objectives = []
    iterations = list(range(len(optimizer.res)))
    current_best = -np.inf
    for res in optimizer.res:
        if res["target"] > current_best:
            current_best = res["target"]
        best_objectives.append(current_best)
    plt.figure(figsize=(8,6))
    plt.plot(iterations, best_objectives, marker='o', linestyle='-')
    plt.xlabel("Iteration")
    plt.ylabel("Best Objective Value (Cumulative)")
    plt.title("Evolution of Best Objective Value over Iterations")
    evolution_path = os.path.join(visualization_dir, "evolution_best_objective.png")
    plt.savefig(evolution_path)
    plt.close()

    print("Additional visualizations saved to:", visualization_dir)

def main():
    df = build_dataset()
    dataset_dir = "/home/peterzeng/hapticanddmp/besttrajectory/stage1/dataset/"
    os.makedirs(dataset_dir, exist_ok=True)
    csv_path = os.path.join(dataset_dir, "dataset_stage1.csv")
    df.to_csv(csv_path, index=False)
    print(f"Stage 1 dataset saved to {csv_path}")
    
    optimizer = optimize_stage1(df)
    best_params = optimizer.max['params']
    print("Best stage 1 parameters found:", best_params)
    
    learning_dir = "/home/peterzeng/hapticanddmp/besttrajectory/stage1/learning/"
    os.makedirs(learning_dir, exist_ok=True)
    best_params_file = os.path.join(learning_dir, "best_params.txt")
    with open(best_params_file, 'w') as f:
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")
    
    # Convergence plot for Bayesian Optimization
    iterations = [res["target"] for res in optimizer.res]
    plt.figure()
    plt.plot(iterations, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Objective (negative cost)")
    plt.title("Bayesian Optimization Convergence for Stage 1 Trajectory")
    plot_file = os.path.join(learning_dir, "convergence_stage1.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"Convergence plot saved to {plot_file}")
    
    # Generate additional visualizations for further analysis
    generate_visualizations(df, optimizer)

if __name__ == '__main__':
    main()
