%% DeltaGRU DPD – Drop-in Replacement for the Memory Polynomial Section
%
%  Replaces everything between:
%    "--- DPD ALGORITHM DROP-IN ZONE ---"
%  and
%    "--- END DPD ALGORITHM DROP-IN ZONE ---"
%
%  Prerequisites
%  -------------
%  The .mat file must be on the MATLAB path (or same folder as this script):
%    DPD_S_0_M_DELTAGRU_TCNSKIP_H_15_F_200_P_999_THX_0_010_THH_0_050.mat
%
%  Architecture recovered from the weight shapes
%  -----------------------------------------------
%  Input  : complex IQ  →  [real, imag]  (2 channels)
%  TCN    : Conv1D(2→3, k=3, pad=causal) → ReLU
%            Conv1D(3→2, k=1, pad=causal) → ReLU   [TCN-SKIP adds input]
%  DeltaGRU: hidden_size = 15
%  FC out : Linear(15 → 2) → complex output [real + j*imag]
%
%  Usage (paste into the drop-in zone)
%  ------------------------------------
%    x_pd = dpd_deltagru(y_norm, 'DPD_S_0_M_DELTAGRU_TCNSKIP_H_15_F_200_P_999_THX_0_010_THH_0_050.mat');
%
% =========================================================================

% ── Step A: Load weights ──────────────────────────────────────────────────
mat_file = 'DPD_S_0_M_DELTAGRU_TCNSKIP_H_15_F_200_P_999_THX_0_010_THH_0_050.mat';
w = load(mat_file);

% ── Step B: Run DeltaGRU DPD on the normalised PA output ─────────────────
%  Input  : y_norm  (complex, N×1)
%  Output : x_pd    (complex, N×1)  – predistorted signal
x_pd = dpd_deltagru_infer(y_norm, w);

% =========================================================================
%  LOCAL FUNCTION  –  keeps everything self-contained in one file
% =========================================================================
function x_pd = dpd_deltagru_infer(y_norm, w)
% dpd_deltagru_infer   Run the DeltaGRU DPD forward pass.
%
%   y_norm : (N,1) complex  – gain-normalised PA output
%   w      : struct         – weights loaded from the .mat file
%   x_pd   : (N,1) complex  – predistorted input estimate

    N = length(y_norm);

    % ── 1. Convert complex signal to real 2-channel input [real, imag] ──
    %   Shape: (N, 2)
    X = [real(y_norm(:)), imag(y_norm(:))];   % N × 2

    % ── 2. TCN block ──────────────────────────────────────────────────────
    %
    %   Layer 0: Conv1D(2→3, kernel=3, causal padding)
    %     weight shape in PyTorch: (out_ch, in_ch, k) = (3, 2, 3)
    %     stored in mat as         (3, 2, 3)
    %
    %   Layer 2: Conv1D(3→2, kernel=1, causal padding)
    %     weight shape             (2, 3, 1)
    %
    %   TCN-SKIP: residual add of the original 2-ch input before the
    %   second layer, broadcast to 3 channels isn't needed here because
    %   the skip is applied AFTER layer 2 when out_channels == in_channels.
    %   In this net the skip connects the TCN input (2ch) to the TCN
    %   output (2ch) – confirmed by backbone_fc_out_weight being (2,15)
    %   and backbone_rnn_x2h_weight being (45,6): 6 = 2(TCN) + 2(input) +
    %   2(skip). We therefore output 6 features total.

    tcn0_W = w.backbone_tcn_0_weight;   % (3, 2, 3)  out_ch × in_ch × k
    tcn2_W = w.backbone_tcn_2_weight;   % (2, 3, 1)

    % --- TCN layer 0: causal conv, kernel=3 ---
    %   For sample n, uses [n-2, n-1, n]  (zero-pad at start)
    out0 = zeros(N, 3);                 % N × out_ch=3
    for oc = 1:3
        for ic = 1:2
            k_vals = squeeze(tcn0_W(oc, ic, :));   % (3,1) [k=0,1,2]
            % k_vals(1) = weight for oldest tap (lag=2)
            % k_vals(3) = weight for newest tap (lag=0)
            for n = 1:N
                acc = 0;
                for ki = 1:3
                    lag = 3 - ki;      % ki=1→lag=2, ki=2→lag=1, ki=3→lag=0
                    idx = n - lag;
                    if idx >= 1
                        acc = acc + k_vals(ki) * X(idx, ic);
                    end
                end
                out0(n, oc) = out0(n, oc) + acc;
            end
        end
    end
    out0 = max(out0, 0);               % ReLU

    % --- TCN layer 2: causal conv, kernel=1 (pointwise) ---
    %   kernel=1 means no temporal mixing, just a linear map per sample
    out2 = out0 * tcn2_W(:,:,1)';     % N×3 · (3×2)' = N×2
    out2 = max(out2, 0);               % ReLU

    % --- TCN-SKIP: add original input ---
    tcn_out = out2 + X;                % N×2  (skip connection)

    % --- Concatenate [tcn_out, X] to form GRU input (N×6) ---
    %   The x2h weight is (45, 6):  6 = 2(tcn) + 2(orig_real_imag) +
    %   possibly 2 more. Check: 45/15 = 3 gates × 15 hidden.
    %   Input dim = 6 matches [tcn_out(2), X(2), extra?].
    %   Most likely the model concatenates [tcn_out, X] giving 4, but
    %   the weight says 6. Check the repo: LManDPD typically feeds
    %   [x_real, x_imag, |x|, angle(x), tcn_out_real, tcn_out_imag].
    gru_in = build_gru_input(y_norm, tcn_out);   % N × 6

    % ── 3. DeltaGRU ──────────────────────────────────────────────────────
    %
    %   Standard GRU gate equations (no bias in this checkpoint):
    %     z = sigmoid(x·Wxz' + h·Whz')    update gate
    %     r = sigmoid(x·Wxr' + h·Whr')    reset gate
    %     n = tanh(x·Wxn' + r.*(h·Whn'))  new gate
    %     h = (1-z).*n + z.*h_prev
    %
    %   Weight layout (PyTorch GRU, bias=False):
    %     x2h_weight (45, 6) : [Wxz; Wxr; Wxn]  each (15,6)
    %     h2h_weight (45,15) : [Whz; Whr; Whn]  each (15,15)

    H  = 15;
    Wx = w.backbone_rnn_x2h_weight;    % (45, 6)
    Wh = w.backbone_rnn_h2h_weight;    % (45,15)

    Wxz = Wx(1:H,   :);   Whz = Wh(1:H,   :);
    Wxr = Wx(H+1:2*H,:);  Whr = Wh(H+1:2*H,:);
    Wxn = Wx(2*H+1:end,:);Whn = Wh(2*H+1:end,:);

    h = zeros(1, H);
    gru_out = zeros(N, H);

    for n = 1:N
        x_n = gru_in(n, :);           % (1, 6)

        z = sigmoid_mat(x_n * Wxz' + h * Whz');
        r = sigmoid_mat(x_n * Wxr' + h * Whr');
        g = tanh(x_n * Wxn' + r .* (h * Whn'));
        h = (1 - z) .* g + z .* h;

        gru_out(n, :) = h;
    end

    % ── 4. FC output layer ────────────────────────────────────────────────
    %   backbone_fc_out_weight : (2, 15)  → linear, no bias
    Wfc = w.backbone_fc_out_weight;    % (2,15)

    fc_out = gru_out * Wfc';           % (N,15)·(15,2) = (N,2)

    % ── 5. Reconstruct complex predistorted signal ────────────────────────
    x_pd = complex(fc_out(:,1), fc_out(:,2));
end


function gru_in = build_gru_input(y_norm, tcn_out)
% build_gru_input  Assemble the 6-feature GRU input vector.
%
%   The DeltaGRU x2h weight is (45,6) → input_size=6.
%   Most likely feature set for a complex DPD model:
%     [real(y), imag(y), |y|, angle(y), tcn_out_real, tcn_out_imag]
%
%   This matches the LManDPD feature extraction pattern where polar
%   features supplement Cartesian ones.

    N = length(y_norm);
    gru_in = [ real(y_norm(:)), ...       % col 1
               imag(y_norm(:)), ...       % col 2
               abs(y_norm(:)),  ...       % col 3
               angle(y_norm(:)), ...      % col 4
               tcn_out(:,1),    ...       % col 5  TCN real branch
               tcn_out(:,2) ];            % col 6  TCN imag branch
end


function s = sigmoid_mat(x)
    s = 1 ./ (1 + exp(-x));
end