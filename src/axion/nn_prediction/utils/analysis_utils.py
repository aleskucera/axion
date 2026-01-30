import csv
import numpy as np
import matplotlib.pyplot as plt
import os

def write_contacts_to_csv(csv_filename, step, contacts):
    """
    Write contact data passed into the predictor to CSV file.

    Args:
        csv_filename: Path to the CSV file
        step: Current simulation step
        contacts: Dictionary with keys contact_normals, contact_depths, contact_thicknesses,
                  contact_points_0, contact_points_1 (tensors of shape (num_envs, ...))
    """
    mode = 'w' if step == 0 else 'a'
    with open(csv_filename, mode, newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        env_idx = 0
        row_data = [step]
        header = ['step']

        if step == 0:
            for key in ['contact_depths', 'contact_normals', 'contact_thicknesses',
                        'contact_points_0', 'contact_points_1']:
                if key in contacts:
                    dim = contacts[key].shape[-1]
                    if key == 'contact_depths':
                        for i in range(dim):
                            header.append(f'contact_depth_{i}')
                    elif key == 'contact_normals':
                        for i in range(dim):
                            header.append(f'contact_normal_{i}')
                    elif key == 'contact_thicknesses':
                        for i in range(dim):
                            header.append(f'contact_thickness_{i}')
                    elif key == 'contact_points_0':
                        for i in range(dim):
                            header.append(f'contact_point_0_{i}')
                    elif key == 'contact_points_1':
                        for i in range(dim):
                            header.append(f'contact_point_1_{i}')
            csv_writer.writerow(header)

        for key in ['contact_depths', 'contact_normals', 'contact_thicknesses',
                    'contact_points_0', 'contact_points_1']:
            if key in contacts:
                data = contacts[key][env_idx, :].cpu().numpy()
                row_data.extend(data.tolist())
        csv_writer.writerow(row_data)


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

def write_model_inputs_to_csv(csv_filename, step, model_inputs):
    """
    Write neural model inputs to CSV file.
    
    Args:
        csv_filename: Path to the CSV file
        step: Current simulation step
        model_inputs: Dictionary containing model inputs with tensors of shape (num_envs, T, dim)
                     Keys: root_body_q, states, states_embedding, joint_acts, gravity_dir,
                           contact_masks, contact_normals, contact_depths, contact_thicknesses,
                           contact_points_0, contact_points_1
    """
    # Extract data for first environment (env 0) and last timestep (T-1) if T > 1, else timestep 0
    # All inputs have shape (num_envs, T, dim)
    env_idx = 0
    
    # On first step (step 0), open in write mode to create/overwrite file and write header
    # On subsequent steps, open in append mode to add data rows
    mode = 'w' if step == 0 else 'a'
    with open(csv_filename, mode, newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Extract and flatten all inputs
        row_data = [step]
        header = ['step']
        
        if step == 0:
            # Build header on first step
            # root_body_q: (num_envs, T, 7)
            if 'root_body_q' in model_inputs:
                for i in range(7):
                    header.append(f'root_body_q_{i}')
            
            # states: (num_envs, T, state_dim)
            if 'states' in model_inputs:
                state_dim = model_inputs['states'].shape[-1]
                for i in range(state_dim):
                    header.append(f'state_{i}')
            
            # states_embedding: (num_envs, T, state_embedding_dim)
            if 'states_embedding' in model_inputs:
                embedding_dim = model_inputs['states_embedding'].shape[-1]
                for i in range(embedding_dim):
                    header.append(f'states_embedding_{i}')
            
            # joint_acts: (num_envs, T, joint_act_dim)
            if 'joint_acts' in model_inputs:
                joint_act_dim = model_inputs['joint_acts'].shape[-1]
                for i in range(joint_act_dim):
                    header.append(f'joint_act_{i}')
            
            # gravity_dir: (num_envs, T, 3)
            if 'gravity_dir' in model_inputs:
                for i in range(3):
                    header.append(f'gravity_dir_{i}')
            
            # contact_masks: (num_envs, T, num_contacts_per_env)
            if 'contact_masks' in model_inputs:
                num_contacts = model_inputs['contact_masks'].shape[-1]
                for i in range(num_contacts):
                    header.append(f'contact_mask_{i}')
            
            # contact_normals: (num_envs, T, num_contacts_per_env * 3)
            if 'contact_normals' in model_inputs:
                contact_normals_dim = model_inputs['contact_normals'].shape[-1]
                for i in range(contact_normals_dim):
                    header.append(f'contact_normal_{i}')
            
            # contact_depths: (num_envs, T, num_contacts_per_env)
            if 'contact_depths' in model_inputs:
                num_contacts = model_inputs['contact_depths'].shape[-1]
                for i in range(num_contacts):
                    header.append(f'contact_depth_{i}')
            
            # contact_thicknesses: (num_envs, T, num_contacts_per_env)
            if 'contact_thicknesses' in model_inputs:
                num_contacts = model_inputs['contact_thicknesses'].shape[-1]
                for i in range(num_contacts):
                    header.append(f'contact_thickness_{i}')
            
            # contact_points_0: (num_envs, T, num_contacts_per_env * 3)
            if 'contact_points_0' in model_inputs:
                contact_points_0_dim = model_inputs['contact_points_0'].shape[-1]
                for i in range(contact_points_0_dim):
                    header.append(f'contact_point_0_{i}')
            
            # contact_points_1: (num_envs, T, num_contacts_per_env * 3)
            if 'contact_points_1' in model_inputs:
                contact_points_1_dim = model_inputs['contact_points_1'].shape[-1]
                for i in range(contact_points_1_dim):
                    header.append(f'contact_point_1_{i}')
            
            csv_writer.writerow(header)
        
        # Extract data for env 0, last timestep (or timestep 0 if T == 1)
        # root_body_q: (num_envs, T, 7)
        if 'root_body_q' in model_inputs:
            T = model_inputs['root_body_q'].shape[1]
            timestep_idx = T - 1 if T > 1 else 0
            data = model_inputs['root_body_q'][env_idx, timestep_idx, :].cpu().numpy()
            row_data.extend(data.tolist())
        
        # states: (num_envs, T, state_dim)
        if 'states' in model_inputs:
            T = model_inputs['states'].shape[1]
            timestep_idx = T - 1 if T > 1 else 0
            data = model_inputs['states'][env_idx, timestep_idx, :].cpu().numpy()
            row_data.extend(data.tolist())
        
        # states_embedding: (num_envs, T, state_embedding_dim)
        if 'states_embedding' in model_inputs:
            T = model_inputs['states_embedding'].shape[1]
            timestep_idx = T - 1 if T > 1 else 0
            data = model_inputs['states_embedding'][env_idx, timestep_idx, :].cpu().numpy()
            row_data.extend(data.tolist())
        
        # joint_acts: (num_envs, T, joint_act_dim)
        if 'joint_acts' in model_inputs:
            T = model_inputs['joint_acts'].shape[1]
            timestep_idx = T - 1 if T > 1 else 0
            data = model_inputs['joint_acts'][env_idx, timestep_idx, :].cpu().numpy()
            row_data.extend(data.tolist())
        
        # gravity_dir: (num_envs, T, 3)
        if 'gravity_dir' in model_inputs:
            T = model_inputs['gravity_dir'].shape[1]
            timestep_idx = T - 1 if T > 1 else 0
            data = model_inputs['gravity_dir'][env_idx, timestep_idx, :].cpu().numpy()
            row_data.extend(data.tolist())
        
        # contact_masks: (num_envs, T, num_contacts_per_env)
        if 'contact_masks' in model_inputs:
            T = model_inputs['contact_masks'].shape[1]
            timestep_idx = T - 1 if T > 1 else 0
            data = model_inputs['contact_masks'][env_idx, timestep_idx, :].cpu().numpy()
            # Convert boolean to float for CSV
            row_data.extend(data.astype(float).tolist())
        
        # contact_normals: (num_envs, T, num_contacts_per_env * 3)
        if 'contact_normals' in model_inputs:
            T = model_inputs['contact_normals'].shape[1]
            timestep_idx = T - 1 if T > 1 else 0
            data = model_inputs['contact_normals'][env_idx, timestep_idx, :].cpu().numpy()
            row_data.extend(data.tolist())
        
        # contact_depths: (num_envs, T, num_contacts_per_env)
        if 'contact_depths' in model_inputs:
            T = model_inputs['contact_depths'].shape[1]
            timestep_idx = T - 1 if T > 1 else 0
            data = model_inputs['contact_depths'][env_idx, timestep_idx, :].cpu().numpy()
            row_data.extend(data.tolist())
        
        # contact_thicknesses: (num_envs, T, num_contacts_per_env)
        if 'contact_thicknesses' in model_inputs:
            T = model_inputs['contact_thicknesses'].shape[1]
            timestep_idx = T - 1 if T > 1 else 0
            data = model_inputs['contact_thicknesses'][env_idx, timestep_idx, :].cpu().numpy()
            row_data.extend(data.tolist())
        
        # contact_points_0: (num_envs, T, num_contacts_per_env * 3)
        if 'contact_points_0' in model_inputs:
            T = model_inputs['contact_points_0'].shape[1]
            timestep_idx = T - 1 if T > 1 else 0
            data = model_inputs['contact_points_0'][env_idx, timestep_idx, :].cpu().numpy()
            row_data.extend(data.tolist())
        
        # contact_points_1: (num_envs, T, num_contacts_per_env * 3)
        if 'contact_points_1' in model_inputs:
            T = model_inputs['contact_points_1'].shape[1]
            timestep_idx = T - 1 if T > 1 else 0
            data = model_inputs['contact_points_1'][env_idx, timestep_idx, :].cpu().numpy()
            row_data.extend(data.tolist())
        
        # Write data row
        csv_writer.writerow(row_data)


def write_model_inputs_contacts_to_csv(csv_filename, step, model_inputs):
    """
    Write only contact-related model inputs to CSV file.

    Args:
        csv_filename: Path to the CSV file
        step: Current simulation step
        model_inputs: Dictionary containing model inputs with tensors of shape (num_envs, T, dim).
                     Only contact_* keys are written: contact_masks, contact_normals, contact_depths,
                     contact_thicknesses, contact_points_0, contact_points_1
    """
    env_idx = 0
    contact_keys = [
        'contact_masks', 'contact_normals', 'contact_depths',
        'contact_thicknesses', 'contact_points_0', 'contact_points_1'
    ]
    mode = 'w' if step == 0 else 'a'
    with open(csv_filename, mode, newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        row_data = [step]
        header = ['step']

        if step == 0:
            for key in contact_keys:
                if key not in model_inputs:
                    continue
                dim = model_inputs[key].shape[-1]
                if key == 'contact_masks':
                    for i in range(dim):
                        header.append(f'contact_mask_{i}')
                elif key == 'contact_normals':
                    for i in range(dim):
                        header.append(f'contact_normal_{i}')
                elif key == 'contact_depths':
                    for i in range(dim):
                        header.append(f'contact_depth_{i}')
                elif key == 'contact_thicknesses':
                    for i in range(dim):
                        header.append(f'contact_thickness_{i}')
                elif key == 'contact_points_0':
                    for i in range(dim):
                        header.append(f'contact_point_0_{i}')
                elif key == 'contact_points_1':
                    for i in range(dim):
                        header.append(f'contact_point_1_{i}')
            csv_writer.writerow(header)

        for key in contact_keys:
            if key not in model_inputs:
                continue
            T = model_inputs[key].shape[1]
            timestep_idx = T - 1 if T > 1 else 0
            data = model_inputs[key][env_idx, timestep_idx, :].cpu().numpy()
            if key == 'contact_masks':
                row_data.extend(data.astype(float).tolist())
            else:
                row_data.extend(data.tolist())
        csv_writer.writerow(row_data)


def plot_model_input_from_csv(csv_filename, keyword, num_steps, output_filename=None, format='pdf'):
    """
    Plot a specific model input from CSV file over time.
    
    Args:
        csv_filename: Path to the CSV file containing model inputs
        keyword: Keyword to filter columns (e.g., "gravity_dir", "states", "root_body_q")
                 Will plot all columns that start with this keyword followed by underscore
        num_steps: Number of steps to display on x-axis (will show first num_steps from the data)
        output_filename: Path for the output plot file. If None, uses csv_filename with keyword suffix
        format: Output format ('pdf', 'png', or 'jpg'). Default is 'pdf'
    
    Returns:
        Path to the saved plot file
    """
    # Read CSV file
    steps = []
    data_dict = {}
    
    with open(csv_filename, 'r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        
        # Get all column names that match the keyword pattern
        # Pattern: keyword_0, keyword_1, etc.
        matching_columns = []
        for col_name in csv_reader.fieldnames:
            if col_name == 'step':
                continue
            if col_name.startswith(keyword + '_'):
                matching_columns.append(col_name)
        
        if not matching_columns:
            raise ValueError(f"No columns found matching keyword '{keyword}'. "
                           f"Available columns: {[c for c in csv_reader.fieldnames if c != 'step']}")
        
        # Sort columns by their index suffix for consistent ordering
        def get_index(col_name):
            try:
                return int(col_name.split('_')[-1])
            except ValueError:
                return 0
        
        matching_columns.sort(key=get_index)
        
        # Initialize data lists
        for col in matching_columns:
            data_dict[col] = []
        
        # Read data
        for row in csv_reader:
            steps.append(int(row['step']))
            for col in matching_columns:
                data_dict[col].append(float(row[col]))
    
    # Convert to numpy arrays
    steps = np.array(steps)
    for col in matching_columns:
        data_dict[col] = np.array(data_dict[col])
    
    # Limit data to num_steps
    if num_steps > len(steps):
        num_steps = len(steps)
        print(f"Warning: Requested {num_steps} steps but only {len(steps)} available. Showing all steps.")
    
    steps = steps[:num_steps]
    for col in matching_columns:
        data_dict[col] = data_dict[col][:num_steps]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot each element
    for col in matching_columns:
        # Extract element index from column name (e.g., "gravity_dir_1" -> 1)
        element_idx = col.split('_')[-1]
        label = f'{keyword}[{element_idx}]'
        plt.plot(steps, data_dict[col], label=label, linewidth=1.5)
    
    plt.xlabel('Step', fontsize=12)
    plt.xlim(0, num_steps - 1)
    plt.ylabel(f'{keyword} Value', fontsize=12)
    plt.title(f'{keyword} Evolution Over Time', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Determine output filename
    if output_filename is None:
        base_name = os.path.splitext(csv_filename)[0]
        output_filename = f"{base_name}_{keyword}_plot.{format}"
    
    # Save plot
    plt.savefig(output_filename, format=format, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {output_filename}")
    return output_filename