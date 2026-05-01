import numpy as np

def generate_dictionary_matrix_H(x1, x2, x3, memory_depth_M):
    """
    Generates the dictionary matrix H for a tri-band DPD.
    
    Args:
        x1, x2, x3: Complex numpy arrays of shape (N,) representing the baseband signals.
        memory_depth_M: Integer representing the number of past samples to include (memory).
        
    Returns:
        H: Complex numpy array of shape (N - M, N_features * M).
    """
    N = len(x1)
    
    # Calculate magnitude squared (hardware efficient as discussed previously)
    # Using |u|^2 instead of |u| avoids square root approximations in RTL IF POSSIBLE.
    # NO WE STILL NEED TO INSPECT |u| and |u|^3 features, but we can keep the squared magnitude to evaluate the relevancy.
    mag1 = np.abs(x1)
    mag2 = np.abs(x2)
    mag3 = np.abs(x3)
    mag_sq1 = np.abs(x1)**2
    mag_sq2 = np.abs(x2)**2
    mag_sq3 = np.abs(x3)**2
    mag_cube1 = np.abs(x1)**3
    mag_cube2 = np.abs(x2)**3
    mag_cube3 = np.abs(x3)**3

    # Initialize a list to hold the feature vectors for the current time step
    instantaneous_features = []
    
    # 1. Intra-band features (Classical polynomial equivalent)
    instantaneous_features.append(x1)
    instantaneous_features.append(x1 * mag_sq1)
    instantaneous_features.append(x2)
    instantaneous_features.append(x2 * mag_sq2)
    instantaneous_features.append(x3)
    instantaneous_features.append(x3 * mag_sq3)
    instantaneous_features.append(mag1)
    instantaneous_features.append(mag2)
    instantaneous_features.append(mag3)
    instantaneous_features.append(mag_cube1)
    instantaneous_features.append(mag_cube2)
    instantaneous_features.append(mag_cube3)
    
    # 2. Cross-band envelope features
    instantaneous_features.append(x1 * mag_sq2)
    instantaneous_features.append(x1 * mag_sq3)
    instantaneous_features.append(x2 * mag_sq1)
    instantaneous_features.append(x2 * mag_sq3)
    instantaneous_features.append(x3 * mag_sq1)
    instantaneous_features.append(x3 * mag_sq2)
    
    # 3. IMD3 Phase-Coherent Cross-terms (from Jaraut)
    instantaneous_features.append((x1**2) * np.conj(x2))
    instantaneous_features.append((x2**2) * np.conj(x1))
    instantaneous_features.append((x2**2) * np.conj(x3))
    instantaneous_features.append((x3**2) * np.conj(x2))
    instantaneous_features.append((x1**3) * np.conj(x3))
    instantaneous_features.append((x3**3) * np.conj(x1))

    # 4. Tri-band IMD feature
    instantaneous_features.append(x1 * x2 * np.conj(x3))
    instantaneous_features.append(np.conj(x1) * (x2**2) * np.conj(x3))
    
    # instantaneous_features is a list of arrays. Stack them into shape (N, P)
    # where P is the number of instantaneous features.
    F = np.column_stack(instantaneous_features)
    P = F.shape[1]
    
    # Construct the final matrix H by incorporating temporal memory (M)
    # We must truncate the first M samples to avoid negative indexing.
    valid_samples = N - memory_depth_M
    H = np.zeros((valid_samples, P * memory_depth_M), dtype=complex)
    
    for m in range(memory_depth_M):
        # Shift the features back in time by m
        shifted_F = F[memory_depth_M - m : N - m, :]
        # Place into the corresponding columns of H
        start_col = m * P
        end_col = (m + 1) * P
        H[:, start_col:end_col] = shifted_F
        
    return H

# --- Execution Example ---
# Assuming you have 10,000 samples per band separated by your DDC script
N_samples = 10000
x1_data = np.random.randn(N_samples) + 1j * np.random.randn(N_samples)
x2_data = np.random.randn(N_samples) + 1j * np.random.randn(N_samples)
x3_data = np.random.randn(N_samples) + 1j * np.random.randn(N_samples)

M = 4 # Memory depth as defined in Jaraut

# Construct the numerical dictionary matrix H
H_matrix = generate_dictionary_matrix_H(x1_data, x2_data, x3_data, M)

print(f"Shape of Dictionary Matrix H: {H_matrix.shape}")