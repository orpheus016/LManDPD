import sympy as sp

# 1. Define real symbolic variables for the I and Q components of the 3 bands
I1, Q1, I2, Q2, I3, Q3 = sp.symbols('I1 Q1 I2 Q2 I3 Q3', real=True)

# 2. Construct the complex baseband signals
u1 = I1 + sp.I * Q1
u2 = I2 + sp.I * Q2
u3 = I3 + sp.I * Q3

# 3. Define the complex conjugates
u1_conj = I1 - sp.I * Q1
u2_conj = I2 - sp.I * Q2
u3_conj = I3 - sp.I * Q3

# 4. Define the squared magnitudes (Hardware-efficient DSP mapping)
mag_sq1 = I1**2 + Q1**2
mag_sq2 = I2**2 + Q2**2
mag_sq3 = I3**2 + Q3**2

# Note: If exact magnitude (square root) is strictly required, use sp.sqrt(mag_sq1).
# However, this will not expand into simple multiplier paths for FPGA.
mag1 = sp.sqrt(mag_sq1)

# 5. Define the dictionary of features based on Jaraut's equations
# Add the rest of your target features to this dictionary
features = {
    "u1 * |u1|^2 (Intra-band)??": u1 * mag_sq1,
    "u1 * |u2|^2 (Cross-band)??": u1 * mag_sq2,
    "u1^2 * u2^* (IMD3 Cross-term)": (u1**2) * u2_conj,
    "u1 * u2 * u3^* (Tri-band IMD)": u1 * u2 * u3_conj,
    "u1 * |u1| (Exact Magnitude)??": u1 * mag1,
    "u1^3 * u3^* (Higher-order IMD)": (u1**3) * u3_conj,
    "u1^* u2^2 u3^* (Complex Nonlinearity)": u1_conj * (u2**2) * u3_conj
}

# 6. Iterate, expand, and separate into Real (I) and Imaginary (Q) hardware paths
for name, expr in features.items():
    # Expand the algebraic expression completely
    expanded_expr = sp.expand(expr)
    
    # Isolate the real and imaginary components
    real_part = sp.re(expanded_expr)
    imag_part = sp.im(expanded_expr)
    
    print(f"--- Feature: {name} ---")
    print(f"I_out (Real Data Path)      = {real_part}")
    print(f"Q_out (Imaginary Data Path) = {imag_part}\n")