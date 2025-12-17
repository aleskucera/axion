import csv
import numpy as np
import matplotlib.pyplot as plt
import os

def write_state_to_csv(csv_filename, step, states):
    """
    Write state vector to CSV file.
    
    Args:
        csv_filename: Path to the CSV file
        step: Current simulation step
        states: State tensor of shape (num_envs, state_dim)
    """
    # On first step (step 0), open in write mode to create/overwrite file and write header
    # On subsequent steps, open in append mode to add data rows
    mode = 'w' if step == 0 else 'a'
    with open(csv_filename, mode, newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write header only on first step
        if step == 0:
            csv_writer.writerow(['step', 'theta_1', 'theta_2', 'dot_theta_1', 'dot_theta_2'])
        
        # Write state data
        csv_writer.writerow([
            step,
            states[0, 0].item(),  # theta_1
            states[0, 1].item(),  # theta_2
            states[0, 2].item(),  # dot_theta_1
            states[0, 3].item()   # dot_theta_2
        ])

def plot_states_from_csv(csv_filename, output_filename=None, format='pdf'):
    """
    Read CSV file produced by write_state_to_csv and plot the 4 state values.
    
    Args:
        csv_filename: Path to the CSV file to read
        output_filename: Path for the output plot file. If None, uses csv_filename with extension replaced
        format: Output format ('pdf', 'png', or 'jpg'). Default is 'pdf'
    
    Returns:
        Path to the saved plot file
    """
    # Read CSV file
    steps = []
    theta_1 = []
    theta_2 = []
    dot_theta_1 = []
    dot_theta_2 = []
    
    with open(csv_filename, 'r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            steps.append(int(row['step']))
            theta_1.append(float(row['theta_1']))
            theta_2.append(float(row['theta_2']))
            dot_theta_1.append(float(row['dot_theta_1']))
            dot_theta_2.append(float(row['dot_theta_2']))
    
    # Convert to numpy arrays
    steps = np.array(steps)
    theta_1 = np.array(theta_1)
    theta_2 = np.array(theta_2)
    dot_theta_1 = np.array(dot_theta_1)
    dot_theta_2 = np.array(dot_theta_2)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps, theta_1, label='θ₁ (theta_1)', linewidth=1.5)
    plt.plot(steps, theta_2, label='θ₂ (theta_2)', linewidth=1.5)
    plt.plot(steps, dot_theta_1, label='θ̇₁ (dot_theta_1)', linewidth=1.5)
    plt.plot(steps, dot_theta_2, label='θ̇₂ (dot_theta_2)', linewidth=1.5)
    
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('State Value', fontsize=12)
    plt.title('Pendulum State Evolution', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Determine output filename
    if output_filename is None:
        base_name = os.path.splitext(csv_filename)[0]
        output_filename = f"{base_name}_plot.{format}"
    
    # Save plot
    plt.savefig(output_filename, format=format, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_filename