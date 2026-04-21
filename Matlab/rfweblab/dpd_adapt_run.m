%% DPD Orchestration for RFWebLab
% Configuration Parameters
fs_weblab = 200e6;       % WebLab strict sampling rate
fs_bb = 30.72e6;         % Baseband sampling rate for standard 20 MHz NR
N_samples = 100000;      % Number of samples for extraction (max 1,000,000)

%% 1. Baseband Signal Generation: Dual-Band OFDM Scenario
% IMS SDC compliance: 5G NR-like OFDM, 30 kHz SCS.
N_fft = 1024;            
N_RB = 51;               
N_sc = N_RB * 12;        
cp_len = 72;             
M = 64;                  

fc1 = -15e6;             
fc2 = 15e6;              

samples_per_sym = N_fft + cp_len;
bb_samples_needed = ceil(N_samples * (fs_bb / fs_weblab));
N_symbols = ceil(bb_samples_needed / samples_per_sym);

bb1 = zeros(samples_per_sym * N_symbols, 1);
bb2 = zeros(samples_per_sym * N_symbols, 1);

for k = 1:N_symbols
    data1 = qammod(randi([0 M-1], N_sc, 1), M, 'UnitAveragePower', true);
    data2 = qammod(randi([0 M-1], N_sc, 1), M, 'UnitAveragePower', true);
    
    grid1 = zeros(N_fft, 1);
    grid2 = zeros(N_fft, 1);
    
    sc_idx = [ (N_fft - N_sc/2 + 1) : N_fft, 2 : (N_sc/2 + 1) ];
    grid1(sc_idx) = data1;
    grid2(sc_idx) = data2;
    
    tx_time1 = ifft(grid1) * sqrt(N_fft);
    tx_time2 = ifft(grid2) * sqrt(N_fft);
    
    tx_sym1 = [tx_time1(end-cp_len+1:end); tx_time1];
    tx_sym2 = [tx_time2(end-cp_len+1:end); tx_time2];
    
    idx_start = (k-1)*samples_per_sym + 1;
    idx_end = k*samples_per_sym;
    bb1(idx_start:idx_end) = tx_sym1;
    bb2(idx_start:idx_end) = tx_sym2;
end

[P, Q] = rat(fs_weblab / fs_bb);
bb1_resamp = resample(bb1, P, Q);
bb2_resamp = resample(bb2, P, Q);

% Digital Frequency Shift (Pre-frequency shift technique)
len = length(bb1_resamp);
t = (0:len-1)' / fs_weblab;

% Generate isolated pre-frequency shifted baseband signals: x'_b(n)
x1_prime = bb1_resamp .* exp(1i * 2 * pi * fc1 * t);
x2_prime = bb2_resamp .* exp(1i * 2 * pi * fc2 * t);

x1_prime = x1_prime(1:N_samples);
x2_prime = x2_prime(1:N_samples);

% Combine signals strictly to drive the physical PA measurement
x_orig = x1_prime + x2_prime;

% Dynamic Power Scaling
peak_val = max(abs(x_orig));
x_orig = x_orig / peak_val;

% Scale isolated signals by the exact same scalar to maintain amplitude ratios
x1_prime = x1_prime / peak_val;
x2_prime = x2_prime / peak_val;

%% 2. Baseline PA Measurement (Without DPD)
disp('Executing Pass 1: Baseline Measurement...');
rmsin_base = calculate_safe_rmsin(x_orig);
fprintf('Pass 1 PAPR: %.2f dB | RMSin: %.2f dBm\n', calculate_papr(x_orig), rmsin_base);
[y_base, RMSout_base, ~, ~] = RFWebLab_PA_meas_v1_2(x_orig, rmsin_base);

if isempty(y_base)
    error('Measurement failed. Check power limits or network connection.');
end

%% 3. Synchronization (Time Alignment) & Initialization
% Derive bulk delay using the composite signal to preserve inter-band phase
[c, lags] = xcorr(y_base, x_orig);
[~, max_idx] = max(abs(c));
delay = lags(max_idx);

if delay > 0
    y_aligned = y_base(delay+1:end);
    x_aligned = x_orig(1:end-delay);
    x1_prime_aligned = x1_prime(1:end-delay);
    x2_prime_aligned = x2_prime(1:end-delay);
elseif delay < 0
    y_aligned = y_base(1:end+delay);
    x_aligned = x_orig(-delay+1:end);
    x1_prime_aligned = x1_prime(-delay+1:end);
    x2_prime_aligned = x2_prime(-delay+1:end);
else
    y_aligned = y_base;
    x_aligned = x_orig;
    x1_prime_aligned = x1_prime;
    x2_prime_aligned = x2_prime;
end

min_len = min(length(x_aligned), length(y_aligned));
y_aligned = y_aligned(1:min_len);
x_aligned = x_aligned(1:min_len);
x1_prime_aligned = x1_prime_aligned(1:min_len);
x2_prime_aligned = x2_prime_aligned(1:min_len);

% Gain and Phase Normalization for Extraction
G_lin = y_aligned \ x_aligned;
y_norm = y_aligned * G_lin;

%% ========================================================================
%  --- DPD ALGORITHM DROP-IN ZONE ---
%  Replace the section below with your custom algorithm.
%  For NN-DPD implementations (e.g., GAN-based architectures or 
%  systolic array verifications), export x_aligned and y_norm to PyTorch 
%  or pass them into your MATLAB neural network inference function here.
% ========================================================================

% MB-BAPS DPD Extraction: Basis-Function Search
% Configuration for Search
%% MB-BAPS DPD Extraction: Band 1
R = 24; M_max = 2; B = 2;

% 1. Downconvert, Filter, and RE-UPCONVERT to isolate the IF signals
y1_bb = y_norm .* exp(-1i * 2 * pi * fc1 * t(1:length(y_norm)));
y2_bb = y_norm .* exp(-1i * 2 * pi * fc2 * t(1:length(y_norm)));

d_filt_ext = designfilt('lowpassfir', 'PassbandFrequency', 10e6, ...
    'StopbandFrequency', 15e6, 'SampleRate', fs_weblab);

y1_bb_filt = filtfilt(d_filt_ext, y1_bb);
y2_bb_filt = filtfilt(d_filt_ext, y2_bb);

% Re-upconvert to IF to satisfy the pre-frequency shift requirement
y1_prime_shifted = y1_bb_filt .* exp(1i * 2 * pi * fc1 * t(1:length(y_norm)));
y2_prime_shifted = y2_bb_filt .* exp(1i * 2 * pi * fc2 * t(1:length(y_norm)));

% Initialize Codebook H with frequency-shifted Outputs
H = [y1_prime_shifted, y2_prime_shifted];
target = x1_prime_aligned; % Target is at fc1

% Preallocate
H = [H, zeros(length(target), R-B)];
op_log = zeros(R, 5);
lambda = 0.1;

for r = (B+1):R
    num_existing = r - 1;
    min_E = inf;
    best_phi = zeros(length(target), 1);
    best_op = [0 0 0 0 0];
    
    % Type I Operations
    for i1 = 1:num_existing
        for m = 1:M_max
            phi_g = [zeros(m, 1); H(1:end-m, i1)];
            H_test = [H(:, 1:num_existing), phi_g];
            
            theta = (H_test' * H_test + lambda * eye(size(H_test,2))) \ (H_test' * target);
            E = norm(H_test * theta - target);
            
            if E < min_E
                min_E = E; best_phi = phi_g; best_op = [1, i1, 0, 0, m];
            end
        end
    end
    
    % Type II Operations
    for j1 = 1:num_existing
        for j2 = 1:j1
            for j3 = 1:num_existing
                phi_g = H(:, j1) .* H(:, j2) .* conj(H(:, j3));
                H_test = [H(:, 1:num_existing), phi_g];
                
                theta = (H_test' * H_test + lambda * eye(size(H_test,2))) \ (H_test' * target);
                E = norm(H_test * theta - target);
                
                if E < min_E
                    min_E = E; best_phi = phi_g; best_op = [2, j1, j2, j3, 0];
                end
            end
        end
    end
    
    H(:, r) = best_phi;
    op_log(r, :) = best_op;
end

w_dpd_b1 = (H' * H + lambda * eye(size(H,2))) \ (H' * target);

%% Predistortion Generation: Band 1
H_pd = [x1_prime, x2_prime]; 
H_pd = [H_pd, zeros(length(x1_prime), R-B)];

for r = (B+1):R
    op = op_log(r, :);
    if op(1) == 1
        m = op(5);
        H_pd(:, r) = [zeros(m, 1); H_pd(1:end-m, op(2))];
    elseif op(1) == 2
        H_pd(:, r) = H_pd(:, op(2)) .* H_pd(:, op(3)) .* conj(H_pd(:, op(4)));
    end
end

x_pd_b1 = H_pd * w_dpd_b1;

%% MB-BAPS DPD Extraction: Band 2
% 1. Downconvert, Filter, and RE-UPCONVERT to isolate Band 2 IF signals
% Band 2 is centered at fc2 (e.g., +15 MHz)
y1_bb_b2 = y_norm .* exp(-1i * 2 * pi * fc1 * t(1:length(y_norm)));
y2_bb_b2 = y_norm .* exp(-1i * 2 * pi * fc2 * t(1:length(y_norm)));

% Use the same isolation filter as Band 1
y1_bb_filt_b2 = filtfilt(d_filt_ext, y1_bb_b2);
y2_bb_filt_b2 = filtfilt(d_filt_ext, y2_bb_b2);

% Re-upconvert to original IF locations for the H-matrix [cite: 167]
y1_prime_b2 = y1_bb_filt_b2 .* exp(1i * 2 * pi * fc1 * t(1:length(y_norm)));
y2_prime_b2 = y2_bb_filt_b2 .* exp(1i * 2 * pi * fc2 * t(1:length(y_norm)));

% Initialize Codebook H for Band 2 [cite: 156]
H_b2 = [y1_prime_b2, y2_prime_b2];
target_b2 = x2_prime_aligned; % Target is the Band 2 input at fc2

% Preallocate and log [cite: 531]
H_b2 = [H_b2, zeros(length(target_b2), R-B)];
op_log_b2 = zeros(R, 5); 
lambda = 0.1; % Tikhonov regularization [cite: 252]

for r = (B+1):R
    num_existing = r - 1;
    min_E = inf;
    best_phi = zeros(length(target_b2), 1);
    best_op = [0 0 0 0 0];
    
    % Type I Operations: Memory Delay [cite: 125]
    for i1 = 1:num_existing
        for m = 1:M_max
            phi_g = [zeros(m, 1); H_b2(1:end-m, i1)];
            H_test = [H_b2(:, 1:num_existing), phi_g];
            
            % Solve via Ridge Regression
            theta = (H_test' * H_test + lambda * eye(size(H_test,2))) \ (H_test' * target_b2);
            E = norm(H_test * theta - target_b2);
            
            if E < min_E
                min_E = E; best_phi = phi_g; best_op = [1, i1, 0, 0, m];
            end
        end
    end
    
    % Type II Operations: 3rd-order Nonlinearity 
    for j1 = 1:num_existing
        for j2 = 1:j1
            for j3 = 1:num_existing
                % Basis: phi_i * phi_l * conj(phi_k) 
                phi_g = H_b2(:, j1) .* H_b2(:, j2) .* conj(H_b2(:, j3));
                H_test = [H_b2(:, 1:num_existing), phi_g];
                
                theta = (H_test' * H_test + lambda * eye(size(H_test,2))) \ (H_test' * target_b2);
                E = norm(H_test * theta - target_b2);
                
                if E < min_E
                    min_E = E; best_phi = phi_g; best_op = [2, j1, j2, j3, 0];
                end
            end
        end
    end
    
    H_b2(:, r) = best_phi;
    op_log_b2(r, :) = best_op;
end

% Final model coefficients for Band 2 [cite: 252]
w_dpd_b2 = (H_b2' * H_b2 + lambda * eye(size(H_b2,2))) \ (H_b2' * target_b2);

%% Predistortion Generation: Band 2
% Reconstruct using the original ideal inputs [cite: 170]
H_pd_b2 = [x1_prime, x2_prime];
H_pd_b2 = [H_pd_b2, zeros(length(x1_prime), R-B)];

for r = (B+1):R
    op = op_log_b2(r, :);
    if op(1) == 1
        m = op(5);
        H_pd_b2(:, r) = [zeros(m, 1); H_pd_b2(1:end-m, op(2))];
    elseif op(1) == 2
        H_pd_b2(:, r) = H_pd_b2(:, op(2)) .* H_pd_b2(:, op(3)) .* conj(H_pd_b2(:, op(4)));
    end
end

x_pd_b2 = H_pd_b2 * w_dpd_b2;

%% Final Composite Upconversion
% The predistorted signal generated by DPD is ALREADY at IF.
% Do NOT multiply by the carrier exponential again.
x_pd = x_pd_b1 + x_pd_b2;

% Apply amplitude hard-clipping while strictly preserving the phase array
clip_idx = abs(x_pd) > max_allowable_peak;
x_pd(clip_idx) = max_allowable_peak .* exp(1i * angle(x_pd(clip_idx)));

%% ========================================================================
%  --- END DPD ALGORITHM DROP-IN ZONE ---
% ========================================================================

%% 4. Power Normalization for WebLab Hardware Limits
% The predistorted signal inherently has peak expansion. 
% The WebLab enforces a strict maximum peak generator power of -8 dBm and 
% limits input RMS power based on PAPR.
PAPR_pd = 20*log10(max(abs(x_pd))*sqrt(length(x_pd))/norm(x_pd));

if PAPR_pd > 20
    error('Predistorted signal PAPR exceeds 20 dB limit. Apply crest factor reduction.');
end

% Normalize x_pd as required by the WebLab API
x_pd_norm = x_pd / max(abs(x_pd));

%% 5. Validation PA Measurement (With DPD)
disp('Executing Pass 2: DPD Validation Measurement...');
rmsin_pd = calculate_safe_rmsin(x_pd_norm);
fprintf('Pass 2 PAPR: %.2f dB | RMSin: %.2f dBm\n', calculate_papr(x_pd_norm), rmsin_pd);
[y_pd, RMSout_pd, ~, ~] = RFWebLab_PA_meas_v1_2(x_pd_norm, rmsin_pd);

if isempty(y_pd)
    error('DPD validation measurement failed.');
end

%% 6. Post-Measurement Synchronization and Evaluation
% Align DPD output to original target
[c_pd, lags_pd] = xcorr(y_pd, x_orig);
[~, max_idx_pd] = max(abs(c_pd));
delay_pd = lags_pd(max_idx_pd);

if delay_pd > 0
    y_pd_aligned = y_pd(delay_pd+1:end);
    x_orig_eval = x_orig(1:end-delay_pd);
else
    y_pd_aligned = y_pd(1:end+delay_pd);
    x_orig_eval = x_orig(-delay_pd+1:end);
end

min_len_eval = min(length(x_orig_eval), length(y_pd_aligned));
x_orig_eval = x_orig_eval(1:min_len_eval);
y_pd_aligned = y_pd_aligned(1:min_len_eval);

% Normalize gain for NMSE calculation using the linear core
y_pd_aligned_norm = y_pd_aligned * (y_pd_aligned \ x_orig_eval);

% Compute Composite Normalized Mean Square Error (NMSE)
NMSE_base = 10*log10(sum(abs(x_aligned - y_norm).^2) / sum(abs(x_aligned).^2));
NMSE_pd = 10*log10(sum(abs(x_orig_eval - y_pd_aligned_norm).^2) / sum(abs(x_orig_eval).^2));

fprintf('\n--- Global Metrics ---\n');
fprintf('Composite Baseline NMSE: %.2f dB\n', NMSE_base);
fprintf('Composite DPD NMSE:      %.2f dB\n', NMSE_pd);

%% 7. Power Spectral Density (PSD)
nfft = 4096;
window = hamming(1024);
noverlap = 512;

[Pxx_base, F] = pwelch(y_aligned, window, noverlap, nfft, fs_weblab, 'centered');
[Pxx_pd, ~] = pwelch(y_pd_aligned, window, noverlap, nfft, fs_weblab, 'centered');
[Pxx_in, ~] = pwelch(x_aligned, window, noverlap, nfft, fs_weblab, 'centered');

figure;
plot(F/1e6, 10*log10(Pxx_in), 'k', 'DisplayName', 'Original Input'); hold on;
plot(F/1e6, 10*log10(Pxx_base), 'b', 'DisplayName', 'Without DPD');
plot(F/1e6, 10*log10(Pxx_pd), 'r', 'DisplayName', 'With DPD');
xlabel('Frequency (MHz)');
ylabel('Power Spectral Density (dB/Hz)');
title('Dual-Band Spectrum Output');
legend;
grid on;

%% 8. Adjacent Channel Leakage Ratio (ACLR)
% Target specific frequencies for Dual-Band
bw_meas = 18e6; % 18 MHz integration bandwidth for a 20 MHz carrier

% Band 1 (-15 MHz) and lower IM3 (-45 MHz)
idx_main1 = find(F >= fc1 - bw_meas/2 & F <= fc1 + bw_meas/2);
idx_im3_lower = find(F >= (2*fc1 - fc2) - bw_meas/2 & F <= (2*fc1 - fc2) + bw_meas/2);

% Band 2 (+15 MHz) and upper IM3 (+45 MHz)
idx_main2 = find(F >= fc2 - bw_meas/2 & F <= fc2 + bw_meas/2);
idx_im3_upper = find(F >= (2*fc2 - fc1) - bw_meas/2 & F <= (2*fc2 - fc1) + bw_meas/2);

% Integrate power
P_main1_base = sum(Pxx_base(idx_main1));
P_im3_lower_base = sum(Pxx_base(idx_im3_lower));
P_main2_base = sum(Pxx_base(idx_main2));
P_im3_upper_base = sum(Pxx_base(idx_im3_upper));

P_main1_pd = sum(Pxx_pd(idx_main1));
P_im3_lower_pd = sum(Pxx_pd(idx_im3_lower));
P_main2_pd = sum(Pxx_pd(idx_main2));
P_im3_upper_pd = sum(Pxx_pd(idx_im3_upper));

fprintf('\n--- Spectral Metrics ---\n');
fprintf('Baseline IM3 (Lower / Upper): %.2f dBc / %.2f dBc\n', ...
    10*log10(P_im3_lower_base / P_main1_base), 10*log10(P_im3_upper_base / P_main2_base));
fprintf('DPD IM3      (Lower / Upper): %.2f dBc / %.2f dBc\n', ...
    10*log10(P_im3_lower_pd / P_main1_pd), 10*log10(P_im3_upper_pd / P_main2_pd));

%% 9. Time-Domain EVM Calculation (Parseval's Theorem)
% Generate strict local time vectors to prevent slicing length mismatches
t_eval_base = (0:length(y_norm)-1)' / fs_weblab;
t_eval_pd   = (0:length(y_pd_aligned_norm)-1)' / fs_weblab;
t_eval_orig = (0:length(x_orig_eval)-1)' / fs_weblab;

% 1. Downconvert Band 1 (Desired Signal) to DC independently
y_b1_base  = y_norm .* exp(-1i * 2 * pi * fc1 * t_eval_base);
y_b1_pd    = y_pd_aligned_norm .* exp(-1i * 2 * pi * fc1 * t_eval_pd);
x_b1_ideal = x_orig_eval .* exp(-1i * 2 * pi * fc1 * t_eval_orig);

% 2. Isolate Band 1 with a strict Low-Pass Filter
% Cutoff at 10 MHz isolates the 20 MHz bandwidth carrier
d_filt = designfilt('lowpassfir', 'PassbandFrequency', 9e6, ...
    'StopbandFrequency', 12e6, 'SampleRate', fs_weblab);

y_b1_base_filt  = filtfilt(d_filt, y_b1_base);
y_b1_pd_filt    = filtfilt(d_filt, y_b1_pd);
x_b1_ideal_filt = filtfilt(d_filt, x_b1_ideal);

% 3. Normalize individual band magnitude to reference
y_b1_base_filt = y_b1_base_filt * (y_b1_base_filt \ x_b1_ideal_filt);
y_b1_pd_filt   = y_b1_pd_filt * (y_b1_pd_filt \ x_b1_ideal_filt);

% 4. Compute RMS EVM
EVM_base = 100 * sqrt(mean(abs(y_b1_base_filt - x_b1_ideal_filt).^2) / mean(abs(x_b1_ideal_filt).^2));
EVM_pd   = 100 * sqrt(mean(abs(y_b1_pd_filt - x_b1_ideal_filt).^2) / mean(abs(x_b1_ideal_filt).^2));

fprintf('\n--- Modulation Metrics ---\n');
fprintf('Band 1 Baseline EVM: %.2f%%\n', EVM_base);
fprintf('Band 1 DPD EVM:      %.2f%%\n', EVM_pd);

%% 10. Single-Band AM/AM Characteristics
% Plotting the composite dual-band signal causes severe dispersion. 
% Plotting the isolated Band 1 provides a clearer assessment of compression recovery.

figure;
plot(abs(x_b1_ideal_filt), abs(y_b1_base_filt), '.', 'DisplayName', 'Without DPD'); hold on;
plot(abs(x_b1_ideal_filt), abs(y_b1_pd_filt), '.', 'DisplayName', 'With DPD');
plot([0 max(abs(x_b1_ideal_filt))], [0 max(abs(x_b1_ideal_filt))], 'k--', 'DisplayName', 'Ideal Linear');
xlabel('Input Magnitude |x_1(n)|');
ylabel('Output Magnitude |y_1(n)|');
title('Isolated Band 1 AM/AM Characteristics');
legend;
grid on;

%% 11. Single-Band AM/PM Characteristics
% Extracts the phase deviation induced by the PA's AM/PM distortion.
% Relies on the filtered, downconverted Band 1 signals from Section 9.

phase_err_base = unwrap(angle(y_b1_base_filt)) - unwrap(angle(x_b1_ideal_filt));
phase_err_pd   = unwrap(angle(y_b1_pd_filt)) - unwrap(angle(x_b1_ideal_filt));

% Zero-center the phase error based on the linear (small signal) region
linear_region = abs(x_b1_ideal_filt) < (0.1 * max(abs(x_b1_ideal_filt)));
phase_err_base = phase_err_base - mean(phase_err_base(linear_region));
phase_err_pd   = phase_err_pd - mean(phase_err_pd(linear_region));

figure;
plot(abs(x_b1_ideal_filt), rad2deg(phase_err_base), '.', 'MarkerSize', 2, 'DisplayName', 'Without DPD'); hold on;
plot(abs(x_b1_ideal_filt), rad2deg(phase_err_pd), '.', 'MarkerSize', 2, 'DisplayName', 'With DPD');
plot([0 max(abs(x_b1_ideal_filt))], [0 0], 'k--', 'LineWidth', 1.5, 'DisplayName', 'Ideal Linear');
xlabel('Input Magnitude |x_1(n)|');
ylabel('Phase Deviation (Degrees)');
title('Isolated Band 1 AM/PM Characteristics');
legend;
grid on;

%% 12. OFDM Demodulation & IQ Constellation
% Time-domain OFDM signals form a Gaussian distribution. To view the 64-QAM 
% constellation, the signal must be explicitly demodulated.

% 1. Resample downconverted Band 1 back to baseband rate (30.72 MHz)
[P_demod, Q_demod] = rat(fs_bb / fs_weblab);
rx_bb_base = resample(y_b1_base, P_demod, Q_demod);
rx_bb_pd   = resample(y_b1_pd, P_demod, Q_demod);

% Calculate integer number of recoverable OFDM symbols
samples_per_sym = N_fft + cp_len;
n_full_syms_base = floor(length(rx_bb_base) / samples_per_sym);
n_full_syms_pd   = floor(length(rx_bb_pd) / samples_per_sym);
n_syms = min(n_full_syms_base, n_full_syms_pd);

% 2. Reshape into [Samples x Symbols] matrix
rx_matrix_base = reshape(rx_bb_base(1:n_syms*samples_per_sym), samples_per_sym, n_syms);
rx_matrix_pd   = reshape(rx_bb_pd(1:n_syms*samples_per_sym), samples_per_sym, n_syms);

% 3. Strip Cyclic Prefix
rx_matrix_base = rx_matrix_base(cp_len+1:end, :);
rx_matrix_pd   = rx_matrix_pd(cp_len+1:end, :);

% 4. FFT to return to frequency domain
rx_grid_base = fft(rx_matrix_base) / sqrt(N_fft);
rx_grid_pd   = fft(rx_matrix_pd) / sqrt(N_fft);

% 5. Extract active subcarriers
sc_idx = [ (N_fft - N_sc/2 + 1) : N_fft, 2 : (N_sc/2 + 1) ];
syms_base = rx_grid_base(sc_idx, :);
syms_pd   = rx_grid_pd(sc_idx, :);

% Flatten matrices to vectors for scatter plotting
syms_base = syms_base(:);
syms_pd   = syms_pd(:);

% Normalize constellations to unit average power for display
syms_base = syms_base / sqrt(mean(abs(syms_base).^2));
syms_pd   = syms_pd / sqrt(mean(abs(syms_pd).^2));

figure;
subplot(1,2,1);
plot(real(syms_base), imag(syms_base), 'b.', 'MarkerSize', 2);
title('Band 1 Constellation (Without DPD)');
axis([-1.5 1.5 -1.5 1.5]); axis square; grid on;

subplot(1,2,2);
plot(real(syms_pd), imag(syms_pd), 'r.', 'MarkerSize', 2);
title('Band 1 Constellation (With DPD)');
axis([-1.5 1.5 -1.5 1.5]); axis square; grid on;

%% --- HELPER FUNCTIONS (Placed at bottom for syntax compliance) ---
function papr = calculate_papr(x)
    %
    papr = 20*log10(max(abs(x)) * sqrt(length(x)) / norm(x));
end

function rms_safe = calculate_safe_rmsin(x_vec)
    % 1. Calculate PAPR exactly as the server does
    %
    papr = 20*log10(max(abs(x_vec)) * sqrt(length(x_vec)) / norm(x_vec));
    
    if papr > 20
        error('Signal PAPR (%.2f dB) exceeds hardware limit of 20 dB.', papr);
    end
    
    % 2. Evaluate WebLab constraints
    %
    limit1 = -8 - papr;          % Peak power limit
    limit2 = -8 - papr * 0.77;   % Protection curve limit
    
    % 3. Apply a 0.5 dB safety margin for numerical stability
    rms_safe = min(limit1, limit2) - 0.5;
end
