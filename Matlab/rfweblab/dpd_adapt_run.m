% DPD Orchestration for RFWebLab
% Configuration Parameters
fs = 200e6;                 % Sampling rate (200 MHz)
bw = 160e6;                 % Max allowable bandwidth (160 MHz)
target_RMSin = -25;         % Target input RMS power in dBm (max -16.47 for step 1)
N_samples = 100000;         % Number of samples (max 1,000,000)

%% 1. Baseband Signal Generation: Dual-Band Scenario
% Hardware constraints: fs = 200 MHz, max bandwidth = 160 MHz [-80 to 80 MHz].

fs = 200e6;          % 200 MHz sampling rate
N_symbols = 20000;   % Number of symbols per band
M = 64;              % 64-QAM

% Band 1: Desired Signal
bw1 = 20e6;          % 20 MHz bandwidth
fc1 = -30e6;         % Carrier shifted to -30 MHz

% Band 2: Interferer Signal
bw2 = 20e6;          % 20 MHz bandwidth
fc2 = 30e6;          % Carrier shifted to +30 MHz

% Oversampling factor
sps = floor(fs / bw1); 

% Generate random QAM symbols
data1 = randi([0 M-1], N_symbols, 1);
data2 = randi([0 M-1], N_symbols, 1);
sym1 = qammod(data1, M, 'UnitAveragePower', true);
sym2 = qammod(data2, M, 'UnitAveragePower', true);

% Pulse Shaping (Root Raised Cosine)
beta = 0.22; % Standard roll-off factor
span = 10;   % Filter span in symbols
rrcFilter = rcosdesign(beta, span, sps, 'sqrt');

% Upsample and apply RRC filter
bb1 = upfirdn(sym1, rrcFilter, sps);
bb2 = upfirdn(sym2, rrcFilter, sps);

% Truncate to ensure equal length matrices
len = min(length(bb1), length(bb2));
bb1 = bb1(1:len);
bb2 = bb2(1:len);

% Digital Frequency Shift (Upconversion to digital IF)
t = (0:len-1)' / fs;
signal1 = bb1 .* exp(1i * 2 * pi * fc1 * t);
signal2 = bb2 .* exp(1i * 2 * pi * fc2 * t);

% Combine signals
% The interferer power ratio can be adjusted here.
power_ratio_interferer = 1; % 0 dB ratio
x_orig = signal1 + sqrt(power_ratio_interferer) * signal2;

% Normalize peak to 1 for WebLab constraints
x_orig = x_orig / max(abs(x_orig));

%% 2. Baseline PA Measurement (Without DPD)
disp('Executing Pass 1: Baseline Measurement...');
[y_base, RMSout_base, Idc_base, Vdc_base] = RFWebLab_PA_meas_v1_2(x_orig, target_RMSin);

if isempty(y_base)
    error('Measurement failed. Check power limits or network connection.');
end

%% 3. Synchronization (Time Alignment)
% Critical Step: The WebLab starts sampling randomly. The PA output (y_base)
% must be time-aligned with the input (x_orig) before model extraction.
[c, lags] = xcorr(y_base, x_orig);
[~, max_idx] = max(abs(c));
delay = lags(max_idx);

if delay > 0
    y_aligned = y_base(delay+1:end);
    x_aligned = x_orig(1:end-delay);
elseif delay < 0
    y_aligned = y_base(1:end+delay);
    x_aligned = x_orig(-delay+1:end);
else
    y_aligned = y_base;
    x_aligned = x_orig;
end

% Truncate to ensure equal lengths for matrix operations
min_len = min(length(x_aligned), length(y_aligned));
x_aligned = x_aligned(1:min_len);
y_aligned = y_aligned(1:min_len);

% Gain and Phase Normalization for Extraction
G_lin = y_aligned \ x_aligned; % Linear gain estimation
y_norm = y_aligned * G_lin;

%% ========================================================================
%  --- DPD ALGORITHM DROP-IN ZONE ---
%  Replace the section below with your custom algorithm.
%  For NN-DPD implementations (e.g., GAN-based architectures or 
%  systolic array verifications), export x_aligned and y_norm to PyTorch 
%  or pass them into your MATLAB neural network inference function here.
% ========================================================================

% Step A: Extraction (Learning the inverse PA behavior)
% Target: Find a function f() such that x_aligned = f(y_norm)
% [EXAMPLE: Memory Polynomial Extraction]
M = 3; % Memory depth
K = 5; % Non-linear order (must be odd)
Phi = zeros(length(y_norm), M * ((K+1)/2)); % Pre-allocate basis matrix

col = 1;
for m = 0:M-1
    for k = 1:2:K
        shifted_y = [zeros(m,1); y_norm(1:end-m)];
        Phi(:, col) = shifted_y .* abs(shifted_y).^(k-1);
        col = col + 1;
    end
end
% Solve for coefficients using Least Squares
w_dpd = Phi \ x_aligned; 

% Step B: Predistortion (Applying the inverse to the original signal)
% Target: Create x_pd = f(x_orig)
Phi_pd = zeros(length(x_orig), M * ((K+1)/2));
col = 1;
for m = 0:M-1
    for k = 1:2:K
        shifted_x = [zeros(m,1); x_orig(1:end-m)];
        Phi_pd(:, col) = shifted_x .* abs(shifted_x).^(k-1);
        col = col + 1;
    end
end
x_pd = Phi_pd * w_dpd;

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
[y_pd, RMSout_pd, Idc_pd, Vdc_pd] = RFWebLab_PA_meas_v1_2(x_pd_norm, target_RMSin);

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

% Normalize gain for NMSE calculation
y_pd_aligned_norm = y_pd_aligned * (y_pd_aligned \ x_orig_eval);

% Compute Normalized Mean Square Error (NMSE)
NMSE_base = 10*log10(sum(abs(x_aligned - y_norm).^2) / sum(abs(x_aligned).^2));
NMSE_pd = 10*log10(sum(abs(x_orig_eval - y_pd_aligned_norm).^2) / sum(abs(x_orig_eval).^2));

fprintf('Baseline NMSE: %.2f dB\n', NMSE_base);
fprintf('DPD NMSE: %.2f dB\n', NMSE_pd);

% Calculate AM/AM characteristic plot data
figure;
plot(abs(x_aligned), abs(y_norm), '.', 'DisplayName', 'Without DPD'); hold on;
plot(abs(x_orig_eval), abs(y_pd_aligned_norm), '.', 'DisplayName', 'With DPD');
plot([0 1], [0 1], 'k--', 'DisplayName', 'Ideal Linear');
xlabel('Input Magnitude |x(n)|');
ylabel('Output Magnitude |y(n)|');
title('AM/AM Characteristics');
legend;
grid on;

%% 7. AM-PM Characteristics
% Calculates the phase deviation between input and output dependent on input amplitude.
phase_error_base = unwrap(angle(y_norm)) - unwrap(angle(x_aligned));
phase_error_pd = unwrap(angle(y_pd_aligned_norm)) - unwrap(angle(x_orig_eval));

% Center phase around 0 for small signals
phase_error_base = phase_error_base - mean(phase_error_base(abs(x_aligned) < 0.1));
phase_error_pd = phase_error_pd - mean(phase_error_pd(abs(x_orig_eval) < 0.1));

figure;
plot(abs(x_aligned), rad2deg(phase_error_base), '.', 'DisplayName', 'Without DPD'); hold on;
plot(abs(x_orig_eval), rad2deg(phase_error_pd), '.', 'DisplayName', 'With DPD');
plot([0 1], [0 0], 'k--', 'DisplayName', 'Ideal Linear');
xlabel('Input Magnitude |x(n)|');
ylabel('Phase Deviation (Degrees)');
title('AM/PM Characteristics');
legend;
grid on;

%% 8. Power Spectral Density (PSD)
% Uses Welch's method for spectral estimation.
nfft = 4096;
window = hamming(1024);
noverlap = 512;

[Pxx_base, F] = pwelch(y_aligned, window, noverlap, nfft, fs, 'centered');
[Pxx_pd, ~] = pwelch(y_pd_aligned, window, noverlap, nfft, fs, 'centered');
[Pxx_in, ~] = pwelch(x_aligned, window, noverlap, nfft, fs, 'centered');

figure;
plot(F/1e6, 10*log10(Pxx_in), 'k', 'DisplayName', 'Original Input'); hold on;
plot(F/1e6, 10*log10(Pxx_base), 'b', 'DisplayName', 'Without DPD');
plot(F/1e6, 10*log10(Pxx_pd), 'r', 'DisplayName', 'With DPD');
xlabel('Frequency (MHz)');
ylabel('Power Spectral Density (dB/Hz)');
title('Spectrum Output');
legend;
grid on;

%% 9. Adjacent Channel Leakage Ratio (ACLR)
% Assumes a signal bandwidth of 20 MHz. Adjust based on your actual signal.
ch_bw = 20e6; 
adj_offset = 20e6; 

% Integration indices
main_idx = find(F >= -ch_bw/2 & F <= ch_bw/2);
adj_lower_idx = find(F >= -adj_offset - ch_bw/2 & F <= -adj_offset + ch_bw/2);
adj_upper_idx = find(F >= adj_offset - ch_bw/2 & F <= adj_offset + ch_bw/2);

% Integrate power
P_main_base = sum(Pxx_base(main_idx));
P_adj_lower_base = sum(Pxx_base(adj_lower_idx));
P_adj_upper_base = sum(Pxx_base(adj_upper_idx));

P_main_pd = sum(Pxx_pd(main_idx));
P_adj_lower_pd = sum(Pxx_pd(adj_lower_idx));
P_adj_upper_pd = sum(Pxx_pd(adj_upper_idx));

ACLR_lower_base = 10*log10(P_adj_lower_base / P_main_base);
ACLR_upper_base = 10*log10(P_adj_upper_base / P_main_base);
ACLR_lower_pd = 10*log10(P_adj_lower_pd / P_main_pd);
ACLR_upper_pd = 10*log10(P_adj_upper_pd / P_main_pd);

fprintf('Baseline ACLR: Lower -%.2f dBc, Upper -%.2f dBc\n', ACLR_lower_base, ACLR_upper_base);
fprintf('DPD ACLR:      Lower -%.2f dBc, Upper -%.2f dBc\n', ACLR_lower_pd, ACLR_upper_pd);

%% 10. IQ Constellation & EVM (REQUIRES QAM SOURCE SIGNAL)
% The following will only produce a scatter plot of noise unless x_orig is modified.
% Assuming matched filtering and downsampling to 1 sample/symbol has been performed:

figure;
subplot(1,2,1);
plot(real(y_norm), imag(y_norm), 'b.', 'MarkerSize', 1);
title('Constellation Without DPD');
axis square; grid on;

subplot(1,2,2);
plot(real(y_pd_aligned_norm), imag(y_pd_aligned_norm), 'r.', 'MarkerSize', 1);
title('Constellation With DPD');
axis square; grid on;

%% 11. Receiver Chain and EVM Calculation
% The competition requires EVM for the desired signal[cite: 46].
% We assume Band 1 (fc1) is the desired signal[cite: 9].

% Recovery Parameters
% Filter delay for RRC is (span * sps) / 2. We must compensate for this.
filterDelay = (span * sps) / 2;

% --- Process Band 1 (Desired) ---
% 1. Digital Downconversion
rx_bb1_base = y_norm .* exp(-1i * 2 * pi * fc1 * t(1:length(y_norm)));
rx_bb1_pd   = y_pd_aligned_norm .* exp(-1i * 2 * pi * fc1 * t(1:length(y_pd_aligned_norm)));

% 2. Matched Filtering (using the same RRC filter)
rx_filt1_base = filter(rrcFilter, 1, rx_bb1_base);
rx_filt1_pd   = filter(rrcFilter, 1, rx_bb1_pd);

% 3. Symbol Sampling and Delay Compensation
% Start sampling after the filter transient and account for pulse shaping delay.
sample_offset = span * sps + 1; 
y_base_symbols = rx_filt1_base(sample_offset:sps:end);
y_pd_symbols   = rx_filt1_pd(sample_offset:sps:end);

% 4. Ideal Reference Recovery
% We use the original symbols (sym1) as the 'ideal' reference[cite: 18].
% Ensure length alignment due to filter truncation.
n_syms = min([length(y_base_symbols), length(y_pd_symbols), length(sym1)]);
y_base_symbols = y_base_symbols(1:n_syms);
y_pd_symbols   = y_pd_symbols(1:n_syms);
x_ideal_symbols = sym1(1:n_syms);

% 5. Magnitude Normalization
% Standard EVM requires the received symbols to be scaled to the reference power.
y_base_symbols = y_base_symbols * (y_base_symbols \ x_ideal_symbols);
y_pd_symbols   = y_pd_symbols * (y_pd_symbols \ x_ideal_symbols);

% 6. EVM Calculation
% RMS EVM Formula: sqrt(mean(error_mag^2) / mean(ref_mag^2))
EVM_base = 100 * sqrt(mean(abs(y_base_symbols - x_ideal_symbols).^2) / mean(abs(x_ideal_symbols).^2));
EVM_pd   = 100 * sqrt(mean(abs(y_pd_symbols - x_ideal_symbols).^2) / mean(abs(x_ideal_symbols).^2));

fprintf('Desired Signal (Band 1) Baseline EVM: %.2f%%\n', EVM_base);
fprintf('Desired Signal (Band 1) DPD EVM:      %.2f%%\n', EVM_pd);

% Optional: Repeat for Band 2 (Interferer) to check isolation[cite: 11].