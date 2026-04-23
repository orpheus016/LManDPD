function meta = acquire_pa_dataset_rfweblab(varargin)
% Acquire PA dataset from RFWebLab and export in OpenDPD split-CSV format.
%
% Output dataset directory layout:
%   datasets/<dataset_name>/
%     train_input.csv, train_output.csv
%     val_input.csv,   val_output.csv
%     test_input.csv,  test_output.csv
%     spec.json
%     acquisition_log.csv
%
% Example:
%   acquire_pa_dataset_rfweblab('dataset_name', 'RFWebLab_200MHz', ...
%       'n_captures', 8, 'samples_per_capture', 120000);

cfg = parse_inputs(varargin{:});

if cfg.random_seed >= 0
    rng(cfg.random_seed);
end

dataset_dir = resolve_dataset_dir(cfg.output_dir, cfg.dataset_name);
if ~exist(dataset_dir, 'dir')
    mkdir(dataset_dir);
end

fprintf('Dataset directory: %s\n', dataset_dir);
fprintf('Starting RFWebLab acquisition (%d captures) ...\n', cfg.n_captures);

x_all = complex([], []);
y_all = complex([], []);

log_capture = zeros(cfg.n_captures, 11);
% Columns:
% [capture_idx success attempts req_samples aligned_samples papr_db rmsin_dbm rmsout_dbm idc_a vdc_v delay_samp]

for k = 1:cfg.n_captures
    x_req = generate_ofdm_excitation(cfg.samples_per_capture, cfg);
    x_req = normalize_like_weblab(x_req);

    papr_db = calculate_papr_db(x_req);
    rmsin_dbm = calculate_safe_rmsin_dbm(papr_db, cfg.rmsin_margin_db);

    success = false;
    attempts = 0;
    y_meas = [];
    rmsout_dbm = NaN;
    idc_a = NaN;
    vdc_v = NaN;

    while ~success && attempts < cfg.max_retries
        attempts = attempts + 1;
        fprintf('Capture %d/%d | attempt %d | PAPR %.2f dB | RMSin %.2f dBm\n', ...
            k, cfg.n_captures, attempts, papr_db, rmsin_dbm);

        [y_try, rmsout_try, idc_try, vdc_try] = RFWebLab_PA_meas_v1_2(x_req, rmsin_dbm);
        if ~isempty(y_try)
            y_meas = y_try(:);
            rmsout_dbm = rmsout_try;
            idc_a = idc_try;
            vdc_v = vdc_try;
            success = true;
        else
            warning('Capture %d attempt %d failed (empty response).', k, attempts);
            pause(cfg.retry_pause_seconds);
        end
    end

    aligned_len = 0;
    delay_samp = NaN;

    if success
        [x_aligned, y_aligned, delay_samp] = align_iq_by_xcorr(x_req, y_meas);
        aligned_len = numel(x_aligned);

        if aligned_len < cfg.min_aligned_samples
            warning('Capture %d discarded: aligned length %d < min %d.', ...
                k, aligned_len, cfg.min_aligned_samples);
            success = false;
        else
            x_all = [x_all; x_aligned]; %#ok<AGROW>
            y_all = [y_all; y_aligned]; %#ok<AGROW>
        end
    end

    log_capture(k, :) = [k, success, attempts, cfg.samples_per_capture, aligned_len, ...
        papr_db, rmsin_dbm, rmsout_dbm, idc_a, vdc_v, delay_samp];
end

ok_rows = log_capture(:, 2) > 0;
if ~any(ok_rows)
    error('All captures failed. No dataset was created.');
end

n_total = numel(x_all);
if n_total < cfg.min_total_samples
    error('Too few total aligned samples (%d). Increase n_captures or samples_per_capture.', n_total);
end

fprintf('Acquisition complete. Total aligned samples: %d\n', n_total);

[idx_train_end, idx_val_end] = split_indices(n_total, cfg.train_ratio, cfg.val_ratio, cfg.test_ratio);

x_train = x_all(1:idx_train_end);
y_train = y_all(1:idx_train_end);

x_val = x_all(idx_train_end + 1:idx_val_end);
y_val = y_all(idx_train_end + 1:idx_val_end);

x_test = x_all(idx_val_end + 1:end);
y_test = y_all(idx_val_end + 1:end);

write_iq_csv(fullfile(dataset_dir, 'train_input.csv'), x_train);
write_iq_csv(fullfile(dataset_dir, 'train_output.csv'), y_train);
write_iq_csv(fullfile(dataset_dir, 'val_input.csv'), x_val);
write_iq_csv(fullfile(dataset_dir, 'val_output.csv'), y_val);
write_iq_csv(fullfile(dataset_dir, 'test_input.csv'), x_test);
write_iq_csv(fullfile(dataset_dir, 'test_output.csv'), y_test);

write_spec_json(dataset_dir, cfg);
write_capture_log(dataset_dir, log_capture);

if cfg.enable_plots
    show_measurement_diagnostics(x_all, y_all, cfg, dataset_dir);
end

meta = struct();
meta.dataset_dir = dataset_dir;
meta.total_samples = n_total;
meta.train_samples = numel(x_train);
meta.val_samples = numel(x_val);
meta.test_samples = numel(x_test);
meta.successful_captures = sum(ok_rows);

fprintf('Saved dataset to: %s\n', dataset_dir);
fprintf('Train/Val/Test samples: %d / %d / %d\n', ...
    meta.train_samples, meta.val_samples, meta.test_samples);
end

function cfg = parse_inputs(varargin)
cfg = struct();

% Configure dataset dir here
cfg.dataset_name = 'RFWebLab_PA_200MHz';
cfg.output_dir = 'D:\James\Homework\ITB\TKTE\Matlab\datasets';
cfg.n_captures = 6;
cfg.samples_per_capture = 120000;
cfg.min_aligned_samples = 10000;
cfg.min_total_samples = 50000;

cfg.fs_hz = 200e6;
cfg.fs_bb_hz = 30.72e6;
cfg.bb_nfft = 1024;
cfg.bb_n_rb = 51;
cfg.cp_len = 72;
cfg.cp_len_first = 80;
cfg.mod_order = 64;
cfg.fc1_hz = -60e6;
cfg.fc2_hz = 0;
cfg.fc3_hz = 60e6;
cfg.nr_scs_hz = 30e3;
cfg.nr_symbols_per_slot = 14;
cfg.nr_dmrs_symbols = [4, 11];
cfg.nr_guard_sc_each_side = 6;
cfg.nr_data_occupancy = 0.9;
cfg.target_papr_db = 12;
cfg.max_cfr_iters = 6;

cfg.rmsin_margin_db = 0.75;
cfg.max_retries = 3;
cfg.retry_pause_seconds = 2;

cfg.train_ratio = 0.6;
cfg.val_ratio = 0.2;
cfg.test_ratio = 0.2;

cfg.n_sub_ch = 3;
cfg.nperseg = 2560;
cfg.random_seed = 0;
cfg.enable_plots = true;
cfg.plot_max_samples = 300000;

if mod(numel(varargin), 2) ~= 0
    error('Arguments must be name/value pairs.');
end

for i = 1:2:numel(varargin)
    key = varargin{i};
    value = varargin{i + 1};
    if ~ischar(key) && ~isstring(key)
        error('Argument names must be strings.');
    end
    key = char(lower(string(key)));
    if ~isfield(cfg, key)
        error('Unknown parameter: %s', key);
    end
    cfg.(key) = value;
end

validateattributes(cfg.n_captures, {'numeric'}, {'scalar', 'integer', '>=', 1});
validateattributes(cfg.samples_per_capture, {'numeric'}, {'scalar', 'integer', '>=', 1000, '<=', 1e6});
validateattributes(cfg.fs_hz, {'numeric'}, {'scalar', 'positive'});
validateattributes(cfg.fs_bb_hz, {'numeric'}, {'scalar', 'positive', '<=', cfg.fs_hz});
validateattributes(cfg.bb_nfft, {'numeric'}, {'scalar', 'integer', '>=', 128});
validateattributes(cfg.bb_n_rb, {'numeric'}, {'scalar', 'integer', '>=', 1});
validateattributes(cfg.cp_len, {'numeric'}, {'scalar', 'integer', '>=', 1});
validateattributes(cfg.cp_len_first, {'numeric'}, {'scalar', 'integer', '>=', cfg.cp_len});
validateattributes(cfg.mod_order, {'numeric'}, {'scalar', 'integer', '>=', 4});
validateattributes(cfg.nr_scs_hz, {'numeric'}, {'scalar', 'positive'});
validateattributes(cfg.nr_symbols_per_slot, {'numeric'}, {'scalar', 'integer', '>=', 1});
validateattributes(cfg.nr_dmrs_symbols, {'numeric'}, {'vector', 'integer', '>=', 1});
validateattributes(cfg.nr_guard_sc_each_side, {'numeric'}, {'scalar', 'integer', '>=', 0});
validateattributes(cfg.nr_data_occupancy, {'numeric'}, {'scalar', '>', 0, '<=', 1});
validateattributes(cfg.max_retries, {'numeric'}, {'scalar', 'integer', '>=', 1});
validateattributes(cfg.train_ratio, {'numeric'}, {'scalar', '>', 0, '<', 1});
validateattributes(cfg.val_ratio, {'numeric'}, {'scalar', '>', 0, '<', 1});
validateattributes(cfg.test_ratio, {'numeric'}, {'scalar', '>', 0, '<', 1});
validateattributes(cfg.plot_max_samples, {'numeric'}, {'scalar', 'integer', '>=', 5000});

if ischar(cfg.enable_plots) || isstring(cfg.enable_plots)
    s = lower(strtrim(char(string(cfg.enable_plots))));
    if any(strcmp(s, {'true', '1', 'yes', 'y'}))
        cfg.enable_plots = true;
    elseif any(strcmp(s, {'false', '0', 'no', 'n'}))
        cfg.enable_plots = false;
    else
        error('enable_plots string value must be true/false/1/0/yes/no.');
    end
elseif isnumeric(cfg.enable_plots)
    cfg.enable_plots = logical(cfg.enable_plots);
end
validateattributes(cfg.enable_plots, {'logical'}, {'scalar'});

ratio_sum = cfg.train_ratio + cfg.val_ratio + cfg.test_ratio;
if abs(ratio_sum - 1.0) > 1e-9
    error('train_ratio + val_ratio + test_ratio must equal 1.0');
end

if mod(cfg.samples_per_capture, 2) ~= 0
    cfg.samples_per_capture = cfg.samples_per_capture - 1;
end

if cfg.fs_hz ~= 200e6
    error('RFWebLab requires fs_hz to be exactly 200e6.');
end

if cfg.bb_n_rb * 12 >= cfg.bb_nfft
    error('bb_n_rb is too large for bb_nfft. Require bb_n_rb*12 < bb_nfft.');
end

if any(cfg.nr_dmrs_symbols > cfg.nr_symbols_per_slot)
    error('nr_dmrs_symbols must be within 1..nr_symbols_per_slot.');
end

if numel(unique(cfg.nr_dmrs_symbols)) ~= numel(cfg.nr_dmrs_symbols)
    error('nr_dmrs_symbols must not contain duplicates.');
end

validate_nr_numerology(cfg);
validate_rfweblab_bw(cfg);
end

function dataset_dir = resolve_dataset_dir(output_dir, dataset_name)
if ~isempty(output_dir)
    dataset_dir = fullfile(output_dir, dataset_name);
    return;
end

this_file = mfilename('fullpath');
rfweblab_dir = fileparts(this_file);
matlab_dir = fileparts(rfweblab_dir);
repo_root = fileparts(matlab_dir);
dataset_dir = fullfile(repo_root, 'datasets', dataset_name);
end

function x = generate_ofdm_excitation(num_samples, cfg)
% Generate a practical NR FR1-like multiband waveform:
% - Slot-based OFDM (14 symbols/slot)
% - Normal CP with first-symbol extension
% - PDSCH-like random data with DMRS-like pilot symbols
% - Three component carriers shifted to IF and summed
fs_weblab = cfg.fs_hz;
fs_bb = cfg.fs_bb_hz;
N_fft = cfg.bb_nfft;
N_RB = cfg.bb_n_rb;
N_sc = N_RB * 12;
cp_len = cfg.cp_len;
cp_len_first = cfg.cp_len_first;
symbols_per_slot = cfg.nr_symbols_per_slot;

avg_cp = cp_len + (cp_len_first - cp_len) / symbols_per_slot;
samples_per_sym_avg = N_fft + avg_cp;
bb_samples_needed = ceil(num_samples * (fs_bb / fs_weblab));
N_symbols = ceil(bb_samples_needed / samples_per_sym_avg);
N_slots = ceil(N_symbols / symbols_per_slot);

bb1 = complex([], []);
bb2 = complex([], []);
bb3 = complex([], []);

for slot_idx = 1:N_slots
    slot_td_1 = nr_slot_waveform(cfg, N_fft, N_sc, cp_len, cp_len_first);
    slot_td_2 = nr_slot_waveform(cfg, N_fft, N_sc, cp_len, cp_len_first);
    slot_td_3 = nr_slot_waveform(cfg, N_fft, N_sc, cp_len, cp_len_first);

    bb1 = [bb1; slot_td_1]; %#ok<AGROW>
    bb2 = [bb2; slot_td_2]; %#ok<AGROW>
    bb3 = [bb3; slot_td_3]; %#ok<AGROW>
end

if numel(bb1) < bb_samples_needed || numel(bb2) < bb_samples_needed || numel(bb3) < bb_samples_needed
    error('Generated baseband waveform is shorter than requested.');
end

bb1 = bb1(1:bb_samples_needed);
bb2 = bb2(1:bb_samples_needed);
bb3 = bb3(1:bb_samples_needed);

[P, Q] = rat(fs_weblab / fs_bb);
bb1_resamp = resample(bb1, P, Q);
bb2_resamp = resample(bb2, P, Q);
bb3_resamp = resample(bb3, P, Q);

len = min([numel(bb1_resamp), numel(bb2_resamp), numel(bb3_resamp)]);
t = (0:len - 1)' / fs_weblab;

x1_prime = bb1_resamp(1:len) .* exp(1i * 2 * pi * cfg.fc1_hz * t);
x2_prime = bb2_resamp(1:len) .* exp(1i * 2 * pi * cfg.fc2_hz * t);
x3_prime = bb3_resamp(1:len) .* exp(1i * 2 * pi * cfg.fc3_hz * t);

x = x1_prime + x2_prime + x3_prime;
x = x(1:min(num_samples, numel(x)));
x = apply_cfr(x, cfg.target_papr_db, cfg.max_cfr_iters);
if numel(x) < num_samples
    x = [x; complex(zeros(num_samples - numel(x), 1), 0)];
end
end

function td_slot = nr_slot_waveform(cfg, N_fft, N_sc, cp_len, cp_len_first)
symbols_per_slot = cfg.nr_symbols_per_slot;
dmrs_symbols = cfg.nr_dmrs_symbols;
guard_sc = cfg.nr_guard_sc_each_side;
data_occ = cfg.nr_data_occupancy;

td_slot = complex([], []);

half_sc = N_sc / 2;
if mod(half_sc, 1) ~= 0
    error('N_sc must be even for NR subcarrier mapping.');
end

usable_half = half_sc - guard_sc;
if usable_half < 1
    error('nr_guard_sc_each_side too large for configured RB allocation.');
end

neg_bins_full = (N_fft - half_sc + 1):N_fft;
pos_bins_full = 2:(half_sc + 1);

neg_bins_data = (N_fft - usable_half + 1):N_fft;
pos_bins_data = 2:(usable_half + 1);

for sym = 1:symbols_per_slot
    grid = complex(zeros(N_fft, 1), 0);

    if any(dmrs_symbols == sym)
        % DMRS-like sparse QPSK comb for channel tracking behavior.
        dmrs_neg = qpsk_symbols(ceil(numel(neg_bins_full) / 2));
        dmrs_pos = qpsk_symbols(ceil(numel(pos_bins_full) / 2));
        grid(neg_bins_full(1:2:end)) = dmrs_neg(1:numel(neg_bins_full(1:2:end)));
        grid(pos_bins_full(1:2:end)) = dmrs_pos(1:numel(pos_bins_full(1:2:end)));

        % Fill remaining RE with lower occupancy payload to mimic puncturing.
        grid = map_payload_with_occupancy(grid, neg_bins_data, pos_bins_data, cfg.mod_order, data_occ * 0.5);
    else
        grid = map_payload_with_occupancy(grid, neg_bins_data, pos_bins_data, cfg.mod_order, data_occ);
    end

    td = ifft(grid) * sqrt(N_fft);
    if sym == 1
        cp = td(end - cp_len_first + 1:end);
    else
        cp = td(end - cp_len + 1:end);
    end
    td_slot = [td_slot; cp; td]; %#ok<AGROW>
end
end

function grid = map_payload_with_occupancy(grid, neg_bins, pos_bins, mod_order, occ)
num_neg = numel(neg_bins);
num_pos = numel(pos_bins);

mask_neg = rand(num_neg, 1) <= occ;
mask_pos = rand(num_pos, 1) <= occ;

if ~any(mask_neg)
    mask_neg(randi(num_neg)) = true;
end
if ~any(mask_pos)
    mask_pos(randi(num_pos)) = true;
end

data_neg = qam_symbols(sum(mask_neg), mod_order);
data_pos = qam_symbols(sum(mask_pos), mod_order);

grid(neg_bins(mask_neg)) = data_neg;
grid(pos_bins(mask_pos)) = data_pos;
end

function s = qam_symbols(num_symbols, mod_order)
sqrt_m = round(sqrt(mod_order));
if sqrt_m * sqrt_m ~= mod_order
    error('mod_order must be a square number (e.g., 4,16,64,256).');
end

levels = -(sqrt_m - 1):2:(sqrt_m - 1);
i_idx = randi([1, sqrt_m], num_symbols, 1);
q_idx = randi([1, sqrt_m], num_symbols, 1);

s = levels(i_idx) + 1i * levels(q_idx);
s = s ./ sqrt(mean(abs(s).^2));
end

function s = qpsk_symbols(num_symbols)
levels = [-1, 1];
i_idx = randi([1, 2], num_symbols, 1);
q_idx = randi([1, 2], num_symbols, 1);
s = levels(i_idx) + 1i * levels(q_idx);
s = s ./ sqrt(2);
end

function x_out = apply_cfr(x_in, target_papr_db, max_iters)
x_out = x_in(:);
if isempty(x_out)
    return;
end

for i = 1:max_iters
    papr_db = calculate_papr_db(x_out);
    if papr_db <= target_papr_db
        break;
    end

    rms_val = sqrt(mean(abs(x_out).^2));
    clip_amp = rms_val * 10^(target_papr_db / 20);
    idx = abs(x_out) > clip_amp;
    x_out(idx) = clip_amp .* exp(1i * angle(x_out(idx)));
end

x_out = normalize_like_weblab(x_out);
end

function x = normalize_peak(x)
x = x(:);
peak_val = max(abs(x));
if peak_val > 0
    x = x / peak_val;
end
end

function x = normalize_like_weblab(x)
% RFWebLab normalizes by max peak across real/imag components.
x = x(:);
peak_real = max(abs(real(x)));
peak_imag = max(abs(imag(x)));
peak = max([peak_real, peak_imag]);
if peak > 0
    x = x / peak;
end
end

function papr_db = calculate_papr_db(x)
x = x(:);
papr_db = 20 * log10(max(abs(x)) * sqrt(numel(x)) / norm(x));
end

function rmsin_dbm = calculate_safe_rmsin_dbm(papr_db, margin_db)
if papr_db > 20
    error('PAPR %.2f dB exceeds RFWebLab limit (20 dB).', papr_db);
end

limit_peak = -8 - papr_db;
limit_protect = -8 - 0.77 * papr_db;
% Keep margin to reduce intermittent power-check rejections.
rmsin_dbm = min(limit_peak, limit_protect) - margin_db;
end

function validate_rfweblab_bw(cfg)
N_sc = cfg.bb_n_rb * 12;
bw_sub = (N_sc / cfg.bb_nfft) * cfg.fs_bb_hz;

if mod(N_sc, 2) ~= 0
    error('N_sc must be even. Check bb_n_rb.');
end

edge1 = abs(cfg.fc1_hz) + bw_sub / 2;
edge2 = abs(cfg.fc2_hz) + bw_sub / 2;
edge3 = abs(cfg.fc3_hz) + bw_sub / 2;
max_edge = max([edge1, edge2, edge3]);

if max_edge > 80e6
    error(['Configured carriers exceed RFWebLab usable band [-80,80] MHz. ', ...
        'Max edge = %.3f MHz'], max_edge / 1e6);
end

if bw_sub >= 80e6
    error('Per-carrier occupied bandwidth must be < 80 MHz for RFWebLab use.');
end

if numel(unique([cfg.fc1_hz, cfg.fc2_hz, cfg.fc3_hz])) < 3
    warning('At least two carrier centers are identical; spectrum will have fewer than 3 distinct bands.');
end
end

function validate_nr_numerology(cfg)
scs_from_fs = cfg.fs_bb_hz / cfg.bb_nfft;
if abs(scs_from_fs - cfg.nr_scs_hz) > 1
    error(['Numerology mismatch: fs_bb_hz/bb_nfft = %.3f Hz but nr_scs_hz = %.3f Hz. ', ...
        'For FR1 30 kHz at 30.72 MHz, use bb_nfft=1024 and nr_scs_hz=30000.'], scs_from_fs, cfg.nr_scs_hz);
end

allowed_scs = [15e3, 30e3, 60e3];
if ~any(abs(cfg.nr_scs_hz - allowed_scs) < 1)
    error('nr_scs_hz must be one of [15e3, 30e3, 60e3] for FR1 profile in this script.');
end
end

function [x_aligned, y_aligned, delay_samp] = align_iq_by_xcorr(x_ref, y_meas)
x_ref = x_ref(:);
y_meas = y_meas(:);

[c, lags] = xcorr(y_meas, x_ref);
[~, idx] = max(abs(c));
delay_samp = lags(idx);

if delay_samp > 0
    y_cut = y_meas(delay_samp + 1:end);
    x_cut = x_ref(1:end - delay_samp);
elseif delay_samp < 0
    y_cut = y_meas(1:end + delay_samp);
    x_cut = x_ref(1 - delay_samp:end);
else
    y_cut = y_meas;
    x_cut = x_ref;
end

min_len = min(numel(x_cut), numel(y_cut));
x_aligned = x_cut(1:min_len);
y_aligned = y_cut(1:min_len);
end

function [idx_train_end, idx_val_end] = split_indices(n_total, train_ratio, val_ratio, test_ratio)
ratio_sum = train_ratio + val_ratio + test_ratio;
if abs(ratio_sum - 1.0) > 1e-9
    error('Split ratios must sum to 1.0');
end

idx_train_end = floor(n_total * train_ratio);
idx_val_end = idx_train_end + floor(n_total * val_ratio);

idx_train_end = max(idx_train_end, 1);
idx_val_end = max(idx_val_end, idx_train_end + 1);
idx_val_end = min(idx_val_end, n_total - 1);
end

function show_measurement_diagnostics(x_all, y_all, cfg, dataset_dir)
% Plot RF-style diagnostics from aggregated aligned captures.
N = min([numel(x_all), numel(y_all), cfg.plot_max_samples]);
if N < 5000
    warning('Skipping plots: not enough samples (%d).', N);
    return;
end

x = x_all(1:N);
y = y_all(1:N);

valid_xy = isfinite(real(x)) & isfinite(imag(x)) & isfinite(real(y)) & isfinite(imag(y));
x = x(valid_xy);
y = y(valid_xy);
if numel(x) < 5000 || numel(y) < 5000
    warning('Skipping plots: insufficient finite samples after filtering.');
    return;
end

fs = cfg.fs_hz;
main_bw = cfg.bb_n_rb * 12 * cfg.nr_scs_hz;

nfft_psd = 4096;
window = hamming(1024);
noverlap = 512;
[Pxx_in, F] = pwelch(x, window, noverlap, nfft_psd, fs, 'centered');
[Pxx_out, ~] = pwelch(y, window, noverlap, nfft_psd, fs, 'centered');

figure('Name', 'RFWebLab Spectrum (Input vs Output)', 'Color', 'w');
plot(F/1e6, 10*log10(Pxx_in + eps), 'k', 'DisplayName', 'Input'); hold on;
plot(F/1e6, 10*log10(Pxx_out + eps), 'b', 'DisplayName', 'PA Output');
xlabel('Frequency (MHz)');
ylabel('Power Spectral Density (dB/Hz)');
title('Spectrum Analyzer View');
legend('Location', 'best');
grid on;
drawnow;

[aclr_l1, aclr_u2] = estimate_im3_aclr(F, Pxx_out, cfg.fc1_hz, cfg.fc2_hz, main_bw);
fprintf('\n--- Spectrum Analyzer Metrics ---\n');
fprintf('PA Output IM3 (Lower around 2f1-f2): %.2f dBc\n', aclr_l1);
fprintf('PA Output IM3 (Upper around 2f2-f1): %.2f dBc\n', aclr_u2);

[x_b1, y_b1] = isolate_band1_baseband(x, y, cfg);
valid_b1 = isfinite(real(x_b1)) & isfinite(imag(x_b1)) & isfinite(real(y_b1)) & isfinite(imag(y_b1));
x_b1 = x_b1(valid_b1);
y_b1 = y_b1(valid_b1);
if isempty(x_b1) || isempty(y_b1)
    warning('Skipping constellation and AM-AM/AM-PM: no finite baseband samples.');
    fprintf('Diagnostics generated for dataset: %s\n', dataset_dir);
    return;
end

[syms_in, syms_out] = extract_constellation_symbols(x_b1, y_b1, cfg);
if ~isempty(syms_in) && ~isempty(syms_out)
    figure('Name', 'Band-1 IQ Constellation', 'Color', 'w');
    subplot(1,2,1);
    plot(real(syms_in), imag(syms_in), 'k.', 'MarkerSize', 2);
    axis square; grid on;
    axis([-1.6 1.6 -1.6 1.6]);
    title('Input Constellation (Band 1)');
    xlabel('I'); ylabel('Q');

    subplot(1,2,2);
    plot(real(syms_out), imag(syms_out), 'b.', 'MarkerSize', 2);
    axis square; grid on;
    axis([-1.6 1.6 -1.6 1.6]);
    title('PA Output Constellation (Band 1)');
    xlabel('I'); ylabel('Q');
    drawnow;
end

[x_ref, y_eq, phase_deg] = compute_amam_ampm(x_b1, y_b1);
if isempty(x_ref) || isempty(y_eq) || isempty(phase_deg)
    warning('Skipping AM-AM/AM-PM: insufficient valid samples.');
    fprintf('Diagnostics generated for dataset: %s\n', dataset_dir);
    return;
end

figure('Name', 'Band-1 AM-AM and AM-PM', 'Color', 'w');
subplot(1,2,1);
plot(abs(x_ref), abs(y_eq), '.', 'MarkerSize', 2, 'DisplayName', 'Measured'); hold on;
mx = max(abs(x_ref));
plot([0 mx], [0 mx], 'k--', 'LineWidth', 1.2, 'DisplayName', 'Ideal Linear');
grid on;
xlabel('|x|');
ylabel('|y_{eq}|');
title('AM-AM (Band 1)');
legend('Location', 'best');

subplot(1,2,2);
plot(abs(x_ref), phase_deg, '.', 'MarkerSize', 2, 'DisplayName', 'Measured'); hold on;
plot([0 mx], [0 0], 'k--', 'LineWidth', 1.2, 'DisplayName', 'Ideal Linear');
grid on;
xlabel('|x|');
ylabel('\Delta\phi (deg)');
title('AM-PM (Band 1, Corrected)');
legend('Location', 'best');
drawnow;

fprintf('Diagnostics generated for dataset: %s\n', dataset_dir);
end

function [aclr_lower, aclr_upper] = estimate_im3_aclr(F, Pxx, fc1, fc2, bw)
bw_meas = 0.9 * bw;

idx_main1 = F >= (fc1 - bw_meas/2) & F <= (fc1 + bw_meas/2);
idx_main2 = F >= (fc2 - bw_meas/2) & F <= (fc2 + bw_meas/2);
idx_im3_l = F >= ((2*fc1 - fc2) - bw_meas/2) & F <= ((2*fc1 - fc2) + bw_meas/2);
idx_im3_u = F >= ((2*fc2 - fc1) - bw_meas/2) & F <= ((2*fc2 - fc1) + bw_meas/2);

P_main1 = sum(Pxx(idx_main1));
P_main2 = sum(Pxx(idx_main2));
P_im3_l = sum(Pxx(idx_im3_l));
P_im3_u = sum(Pxx(idx_im3_u));

aclr_lower = 10 * log10((P_im3_l + eps) / (P_main1 + eps));
aclr_upper = 10 * log10((P_im3_u + eps) / (P_main2 + eps));
end

function [x_b1, y_b1] = isolate_band1_baseband(x, y, cfg)
fs = cfg.fs_hz;
t = (0:numel(x)-1)' / fs;

x_dc = x .* exp(-1i * 2 * pi * cfg.fc1_hz * t);
y_dc = y .* exp(-1i * 2 * pi * cfg.fc1_hz * t);

main_bw = cfg.bb_n_rb * 12 * cfg.nr_scs_hz;
f_pass = 0.45 * main_bw;
f_stop = 0.65 * main_bw;

d = designfilt('lowpassfir', ...
    'PassbandFrequency', f_pass, ...
    'StopbandFrequency', f_stop, ...
    'SampleRate', fs);

x_lp = filtfilt(d, x_dc);
y_lp = filtfilt(d, y_dc);

[P, Q] = rat(cfg.fs_bb_hz / fs);
x_b1 = resample(x_lp, P, Q);
y_b1 = resample(y_lp, P, Q);

M = min(numel(x_b1), numel(y_b1));
x_b1 = x_b1(1:M);
y_b1 = y_b1(1:M);
end

function [syms_in, syms_out] = extract_constellation_symbols(x_bb, y_bb, cfg)
N_fft = cfg.bb_nfft;
symbols_per_slot = cfg.nr_symbols_per_slot;
cp0 = cfg.cp_len_first;
cpn = cfg.cp_len;
N_sc = cfg.bb_n_rb * 12;
g = cfg.nr_guard_sc_each_side;

half = N_sc / 2;
neg_bins = (N_fft - half + 1):N_fft;
pos_bins = 2:(half + 1);

if g * 2 >= numel(neg_bins) || g * 2 >= numel(pos_bins)
    error('nr_guard_sc_each_side is too large for the selected RB allocation.');
end

neg_bins = neg_bins((g+1):(end-g));
pos_bins = pos_bins((g+1):(end-g));

idx = 1;
sym_in = complex([], []);
sym_out = complex([], []);
sym_idx = 0;

while true
    for s = 1:symbols_per_slot
        if s == 1
            cp = cp0;
        else
            cp = cpn;
        end
        stop_idx = idx + cp + N_fft - 1;
        if stop_idx > numel(x_bb) || stop_idx > numel(y_bb)
            syms_in = sym_in;
            syms_out = sym_out;
            if ~isempty(syms_in)
                syms_in = syms_in / sqrt(mean(abs(syms_in).^2) + eps);
            end
            if ~isempty(syms_out)
                syms_out = syms_out / sqrt(mean(abs(syms_out).^2) + eps);
            end
            return;
        end

        x_td = x_bb(idx + cp : idx + cp + N_fft - 1);
        y_td = y_bb(idx + cp : idx + cp + N_fft - 1);

        X = fft(x_td) / sqrt(N_fft);
        Y = fft(y_td) / sqrt(N_fft);

        sym_idx = sym_idx + 1;
        sym_slot_pos = mod(sym_idx - 1, symbols_per_slot) + 1;
        if ~any(cfg.nr_dmrs_symbols == sym_slot_pos)
            sym_in = [sym_in; X(neg_bins); X(pos_bins)]; %#ok<AGROW>
            sym_out = [sym_out; Y(neg_bins); Y(pos_bins)]; %#ok<AGROW>
        end

        idx = idx + cp + N_fft;
    end
end
end

function [x_ref, y_eq, phase_deg] = compute_amam_ampm(x_bb, y_bb)
M = min(numel(x_bb), numel(y_bb));
x_ref = x_bb(1:M);
y_ref = y_bb(1:M);

g_lin = (x_ref' * y_ref) / (x_ref' * x_ref + eps);
y_eq = y_ref / (g_lin + eps);

phase_rad = angle(y_eq .* conj(x_ref));

amp = abs(x_ref);
lin_idx = amp > (0.1 * max(amp)) & amp < (0.3 * max(amp));
if any(lin_idx)
    phase_rad = phase_rad - mean(phase_rad(lin_idx));
end

phase_deg = rad2deg(phase_rad);
end

function write_iq_csv(path_csv, z)
z = z(:);
tbl = table(real(z), imag(z), 'VariableNames', {'I', 'Q'});
writetable(tbl, path_csv);
end

function write_capture_log(dataset_dir, log_capture)
tbl = array2table(log_capture, 'VariableNames', {
    'capture_idx', 'success', 'attempts', 'requested_samples', 'aligned_samples', ...
    'papr_db', 'rmsin_dbm', 'rmsout_dbm', 'idc_a', 'vdc_v', 'delay_samples'});

writetable(tbl, fullfile(dataset_dir, 'acquisition_log.csv'));
end

function write_spec_json(dataset_dir, cfg)
spec = struct();
spec.dataset_format = 'split_csv';
spec.split_ratios = struct('train', cfg.train_ratio, 'val', cfg.val_ratio, 'test', cfg.test_ratio);
spec.input_signal_fs = cfg.fs_hz;
spec.bw_main_ch = (cfg.bb_n_rb * 12 / cfg.bb_nfft) * cfg.fs_bb_hz;
spec.bw_sub_ch = spec.bw_main_ch;
spec.n_sub_ch = cfg.n_sub_ch;
spec.nperseg = cfg.nperseg;

acq = struct();
acq.source = 'RFWebLab_PA_meas_v1_2';
acq.n_captures = cfg.n_captures;
acq.samples_per_capture = cfg.samples_per_capture;
acq.max_retries = cfg.max_retries;
acq.rmsin_margin_db = cfg.rmsin_margin_db;
acq.signal_type = 'OFDM';
acq.signal_profile = '5G NR FR1-like multiband';
acq.mod_order = cfg.mod_order;
acq.nfft = cfg.bb_nfft;
acq.cp_len = cfg.cp_len;
acq.cp_len_first = cfg.cp_len_first;
acq.target_papr_db = cfg.target_papr_db;
acq.fs_bb_hz = cfg.fs_bb_hz;
acq.fc1_hz = cfg.fc1_hz;
acq.fc2_hz = cfg.fc2_hz;
acq.fc3_hz = cfg.fc3_hz;
acq.nr_scs_hz = cfg.nr_scs_hz;
acq.nr_symbols_per_slot = cfg.nr_symbols_per_slot;
acq.nr_dmrs_symbols = cfg.nr_dmrs_symbols;
acq.nr_guard_sc_each_side = cfg.nr_guard_sc_each_side;
acq.nr_data_occupancy = cfg.nr_data_occupancy;
acq.enable_plots = cfg.enable_plots;
acq.plot_max_samples = cfg.plot_max_samples;
spec.acquisition = acq;

json_text = jsonencode(spec, 'PrettyPrint', true);
fid = fopen(fullfile(dataset_dir, 'spec.json'), 'w');
if fid < 0
    error('Unable to write spec.json in %s', dataset_dir);
end
fwrite(fid, json_text, 'char');
fclose(fid);
end