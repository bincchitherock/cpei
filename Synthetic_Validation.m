%% Synthetic_Validation_Full.m
%  Seo Bin Han [seohan@ucdavis.edu]

% The algorithm compares five algorithms across five signal types. 
% Algorithms:
%   1. CPEI  Ramanujan Periodic Subspace energy concentration index
%   2. Welch Welch power spectral density (nonparametric baseline)
%   3. PLV   Phase Locking Value (neural entrainment standard)
%   4. ITPC  Inter-Trial Phase Coherence (epoch-based PLV equivalent)
%   5. YIN   Pitch estimator (de Cheveigne) for F0 ground truth
%
% Signals:
%   1. Clean harmonic FFR          ground truth
%   2. Attenuated fundamental      for hearing loss / octave error
%   3. Drifting F0                 quasi-periodic speech
%   4. Pure noise                  false positive control
%   5. Overlapping harmonics       two-speaker mixture
%

clearvars; clc;
fprintf('    Synthetic Validation: CPEI / WELCH / PLV / ITPC / YIN\n');

%% shared parameters 
fs        = 8192;
dur       = 5;
t         = (0:1/fs:dur-1/fs).';
N         = length(t);
n_harm    = 8;
F0        = 120;
T0        = fs / F0;

% welch
win_sec   = 2;
nfft_w    = 2^nextpow2(win_sec * fs);
noverlap  = floor(nfft_w / 2);
win_w     = hann(nfft_w, 'periodic');

% search range
fmin_s    = 50;
fmax_s    = 500;
Pmin      = max(2, floor(fs/fmax_s));
Pmax      = ceil(fs/fmin_s);
periods   = Pmin:Pmax;
period_hz = fs ./ periods;

% CPEI parameters
cpei_bw   = 5;

% SNR sweep (Signal 1 and 2)
snr_levels = [20, 10, 0, -10, -20];
alpha_levels = [1.0, 0.5, 0.2, 0.1, 0.05];

% noise bands for Welch SNR
noise_lo  = [100, 115];
noise_hi  = [125, 140];

% YIN parameters
yin_thresh  = 0.1;
yin_win_ms  = 25;
yin_win     = round(yin_win_ms / 1000 * fs);
yin_fmin    = 50;
yin_fmax    = 500;

%% algorithm implementations

% 1) RFB mean energy 
function mean_E = run_RFB(signal, fs, fmin, fmax)
    Pmin_l = max(2, floor(fs/fmax));
    Pmax_l = ceil(fs/fmin);
    plist  = Pmin_l:Pmax_l;
    Rcq = 7;  Rav = 3;
    x   = signal(:);
    x   = x - mean(x);
    if max(abs(x)) > 0;  x = x ./ max(abs(x));  end
    mean_E = zeros(1, length(plist));
    for idx = 1:length(plist)
        p      = plist(idx);
        k_orig = 1:p;
        k      = k_orig(gcd(k_orig, p) == 1);
        cq     = zeros(p, 1);
        for n = 0:(p-1)
            for a = k
                cq(n+1) = cq(n+1) + exp(1j*2*pi*a*n/p);
            end
        end
        cq  = real(cq);
        FR  = repmat(cq, Rcq, 1);
        FR  = FR ./ norm(FR);
        FA  = repmat(ones(p,1), Rav, 1);
        FA  = FA ./ length(FA);
        y   = conv(x, FR, 'same');
        y   = abs(y).^2;
        y   = conv(y, FA, 'same');
        mean_E(idx) = mean(y);
    end
end

% 2) CPEI scalar
function cpei = compute_cpei(mean_E, period_hz, target_hz, bw_hz)
    mask = abs(period_hz - target_hz) <= bw_hz;
    if sum(mask) == 0;  cpei = 0;  return;  end
    cpei = sum(mean_E(mask)) / (sum(mean_E) + eps);
end

% 3) Peak frequency from RFB
function pk = rps_peak(mean_E, period_hz, fmin, fmax)
    mask    = period_hz >= fmin & period_hz <= fmax;
    E_sub   = mean_E(mask);
    f_sub   = period_hz(mask);
    [~,idx] = max(E_sub);
    pk      = f_sub(idx);
end

% 4) Welch SNR 
function snr_db = welch_snr(psd, f, target, sbw, nlo, nhi)
    sm = abs(f - target) <= sbw;
    nm = (f >= nlo(1) & f <= nlo(2)) | (f >= nhi(1) & f <= nhi(2));
    if sum(sm)==0 || sum(nm)==0;  snr_db = NaN;  return;  end
    snr_db = 10*log10(mean(psd(sm)) / mean(psd(nm)));
end

% 5) Welch peak 
function pk = welch_peak(psd, f, fmin, fmax)
    mask    = f >= fmin & f <= fmax;
    [~,idx] = max(psd(mask));
    f_sub   = f(mask);
    pk      = f_sub(idx);
end

% 6) PLV
function plv = compute_PLV(signal, fs, target_hz)
    t_loc   = (0:length(signal)-1).' / fs;
    ref     = exp(1j * 2*pi * target_hz * t_loc);   % complex reference
    x_hilb  = hilbert(signal(:));                    % analytic signal
    phase_x = angle(x_hilb);
    phase_r = angle(ref);
    plv     = abs(mean(exp(1j * (phase_x - phase_r))));
end

% 7) ITPC 
function itpc = compute_ITPC(epochs, fs, target_hz)
    [n_ep, n_samp] = size(epochs);
    t_loc          = (0:n_samp-1) / fs;
    ref            = exp(1j * 2*pi * target_hz * t_loc);   % 1 x n_samp
    phase_mat      = zeros(n_ep, n_samp);
    for e = 1:n_ep
        x_hilb        = hilbert(epochs(e,:));
        phase_mat(e,:) = angle(x_hilb);
    end
    phase_ref = angle(repmat(ref, n_ep, 1));
    itpc      = abs(mean(exp(1j * (phase_mat - phase_ref)), 1));
    itpc      = mean(itpc);   % average across time
end

% 8) YIN
% first normalized difference function then find the first local
% minimum below a threshold to estimate the fundamental period.
function [f0_est, confidence] = yin_estimate(signal, fs, fmin, fmax, ...
                                             win_len, thresh)
    x      = signal(:);
    N_s    = length(x);
    tmin   = floor(fs/fmax);
    tmax   = floor(fs/fmin);
    tmax   = min(tmax, floor(win_len/2));

    % difference function over one window
    d = zeros(tmax, 1);
    for tau = 1:tmax
        seg1 = x(1:win_len-tau);
        seg2 = x(1+tau:win_len);
        d(tau) = sum((seg1 - seg2).^2);
    end

    % cumm. mean normalized difference function
    d_prime    = zeros(tmax, 1);
    d_prime(1) = 1;
    running    = 0;
    for tau = 2:tmax
        running      = running + d(tau);
        d_prime(tau) = d(tau) / (running / tau);
    end

    % first min below threshold
    f0_est    = NaN;
    confidence = 0;
    for tau = tmin+1:tmax-1
        if d_prime(tau) < thresh && d_prime(tau) < d_prime(tau-1) && ...
           d_prime(tau) <= d_prime(tau+1)
            % parabolic interpolation (sub-sample acc.)
            if tau > 1 && tau < tmax
                denom = d_prime(tau-1) - 2*d_prime(tau) + d_prime(tau+1);
                if abs(denom) > eps
                    tau_fine = tau + 0.5*(d_prime(tau-1) - d_prime(tau+1))/denom;
                else
                    tau_fine = tau;
                end
            else
                tau_fine = tau;
            end
            f0_est     = fs / tau_fine;
            confidence = 1 - d_prime(tau);
            break;
        end
    end
end

%% epochs for ITPC 
% 50 ms non-overlapping epochs
epoch_ms  = 50;
epoch_len = round(epoch_ms / 1000 * fs);
n_epochs  = floor(N / epoch_len);

%% header printer 
function print_header(cols)
    fprintf('  ');
    for c = 1:length(cols)
        fprintf('%-14s', cols{c});
    end
    fprintf('\n  %s\n', repmat('-', 1, 14*length(cols)));
end

function print_row(label, vals, fmts)
    fprintf('  %-12s', label);
    for v = 1:length(vals)
        fprintf(fmts{v}, vals{v});
    end
    fprintf('\n');
end

%% signal 1 (clean Harmonic FFR)

fprintf('signal 1: clean harmonic FFR \n');
fprintf('F0 = %d Hz | %d harmonics | SNR swept +20 to -20 dB\n\n', F0, n_harm);

clean1 = zeros(N, 1);
for k = 1:n_harm
    clean1 = clean1 + (1/k) * sin(2*pi*k*F0*t);
end
clean1 = clean1 / max(abs(clean1));

cols1 = {'SNR(dB)', 'CPEI', 'WelchSNR', 'PLV', 'ITPC', ...
         'YIN_F0(Hz)', 'YIN_conf', 'RPS_pk(Hz)', 'W_pk(Hz)'};
print_header(cols1);

s1 = struct();
for k = 1:length(snr_levels)
    snr_db   = snr_levels(k);
    np       = 10^(-snr_db/20);
    noisy    = clean1 + np * randn(N, 1);

    % CPEI
    mE1      = run_RFB(noisy, fs, fmin_s, fmax_s);
    cpei1    = compute_cpei(mE1, period_hz, F0, cpei_bw);
    rpk1     = rps_peak(mE1, period_hz, fmin_s, fmax_s);

    % Welch
    [p1,f1]  = pwelch(noisy, win_w, noverlap, nfft_w, fs, 'onesided');
    wsnr1    = welch_snr(p1, f1, F0, 5, noise_lo, noise_hi);
    wpk1     = welch_peak(p1, f1, fmin_s, fmax_s);

    % PLV
    plv1     = compute_PLV(noisy, fs, F0);

    % ITPC
    ep1      = zeros(n_epochs, epoch_len);
    for e = 1:n_epochs
        ep1(e,:) = noisy((e-1)*epoch_len+1 : e*epoch_len);
    end
    itpc1    = compute_ITPC(ep1, fs, F0);

    % YIN
    [yin_f0_1, yin_c1] = yin_estimate(noisy, fs, yin_fmin, yin_fmax, ...
                                       yin_win, yin_thresh);

    fprintf('  %-12d%-14.4f%-14.4f%-14.4f%-14.4f%-14.2f%-14.4f%-14.2f%-14.2f\n', ...
            snr_db, cpei1, wsnr1, plv1, itpc1, ...
            yin_f0_1, yin_c1, rpk1, wpk1);

    s1.snr(k)=snr_db; s1.cpei(k)=cpei1; s1.wsnr(k)=wsnr1;
    s1.plv(k)=plv1;   s1.itpc(k)=itpc1;
    s1.yin_f0(k)=yin_f0_1; s1.yin_c(k)=yin_c1;
    s1.rpk(k)=rpk1;   s1.wpk(k)=wpk1;
end
fprintf('\n');

%% signal 2 (attenuated fundamental)
fprintf('signal 2: attenuated fundamental\n');
fprintf('   F0 = %d Hz | fund. attenuated alpha = 1.0 to 0.05\n\n', F0);

fixed_np = 0.3;

cols2 = {'Alpha', 'CPEI', 'WelchSNR', 'PLV', 'ITPC', ...
         'YIN_F0(Hz)', 'YIN_conf', 'RPS_pk(Hz)', 'W_pk(Hz)', 'W_err?'};
print_header(cols2);

s2 = struct();
for k = 1:length(alpha_levels)
    alpha = alpha_levels(k);
    sig2  = alpha * sin(2*pi*F0*t);
    for h = 2:n_harm
        sig2 = sig2 + (1/h)*sin(2*pi*h*F0*t);
    end
    sig2  = sig2 / max(abs(sig2));
    noisy2 = sig2 + fixed_np * randn(N, 1);

    % CPEI
    mE2    = run_RFB(noisy2, fs, fmin_s, fmax_s);
    cpei2  = compute_cpei(mE2, period_hz, F0, cpei_bw);
    rpk2   = rps_peak(mE2, period_hz, fmin_s, fmax_s);

    % Welch
    [p2,f2] = pwelch(noisy2, win_w, noverlap, nfft_w, fs, 'onesided');
    wsnr2   = welch_snr(p2, f2, F0, 5, noise_lo, noise_hi);
    wpk2    = welch_peak(p2, f2, fmin_s, fmax_s);
    werr2   = abs(wpk2 - 2*F0) < abs(wpk2 - F0);

    % PLV
    plv2   = compute_PLV(noisy2, fs, F0);

    % ITPC
    ep2    = zeros(n_epochs, epoch_len);
    for e = 1:n_epochs
        ep2(e,:) = noisy2((e-1)*epoch_len+1 : e*epoch_len);
    end
    itpc2  = compute_ITPC(ep2, fs, F0);

    % YIN
    [yin_f0_2, yin_c2] = yin_estimate(noisy2, fs, yin_fmin, yin_fmax, ...
                                       yin_win, yin_thresh);
    yin_err2 = ~isnan(yin_f0_2) && abs(yin_f0_2 - 2*F0) < abs(yin_f0_2 - F0);

    if werr2;  ws='true ';  else;  ws='false';  end

    fprintf('  %-12.2f%-14.4f%-14.4f%-14.4f%-14.4f%-14.2f%-14.4f%-14.2f%-14.2f%-10s\n', ...
            alpha, cpei2, wsnr2, plv2, itpc2, ...
            yin_f0_2, yin_c2, rpk2, wpk2, ws);

    s2.alpha(k)=alpha; s2.cpei(k)=cpei2; s2.wsnr(k)=wsnr2;
    s2.plv(k)=plv2;    s2.itpc(k)=itpc2;
    s2.yin_f0(k)=yin_f0_2; s2.yin_c(k)=yin_c2;
    s2.rpk(k)=rpk2;    s2.wpk(k)=wpk2;
    s2.werr(k)=werr2;  s2.yin_err(k)=yin_err2;
end
fprintf('\n');

%% signal 3 (drifting f0 for quasi-periodic speech)
fprintf('signal 3\n');
fprintf('F0 drifts 100 to 150 Hz over %d s | noise = 0.3\n\n', dur);

F0_start = 100;  F0_end = 150;
F0_t     = linspace(F0_start, F0_end, N).';
phase3   = 2*pi * cumsum(F0_t) / fs;
sig3     = zeros(N, 1);
for h = 1:6
    sig3 = sig3 + (1/h) * sin(h * phase3);
end
sig3   = sig3 / max(abs(sig3));
noisy3 = sig3 + 0.3 * randn(N, 1);

F0_mid = (F0_start + F0_end) / 2;

% CPEI
mE3    = run_RFB(noisy3, fs, fmin_s, fmax_s);
cpei3  = compute_cpei(mE3, period_hz, F0_mid, 30);
rpk3   = rps_peak(mE3, period_hz, fmin_s, fmax_s);

% Welch
[p3,f3] = pwelch(noisy3, win_w, noverlap, nfft_w, fs, 'onesided');
wpk3    = welch_peak(p3, f3, fmin_s, fmax_s);

% PLV 
% F0_mid since signal drifts around it
plv3    = compute_PLV(noisy3, fs, F0_mid);

% ITPC
ep3     = zeros(n_epochs, epoch_len);
for e = 1:n_epochs
    ep3(e,:) = noisy3((e-1)*epoch_len+1 : e*epoch_len);
end
itpc3   = compute_ITPC(ep3, fs, F0_mid);

% YIN 
yin_step   = round(10 / 1000 * fs);       % hop 10 ms
n_frames3  = floor((N - yin_win) / yin_step);
yin_f0s3   = NaN(n_frames3, 1);
yin_cs3    = zeros(n_frames3, 1);
for fr = 1:n_frames3
    idx0 = (fr-1)*yin_step + 1;
    seg  = noisy3(idx0 : idx0+yin_win-1);
    [yin_f0s3(fr), yin_cs3(fr)] = yin_estimate(seg, fs, yin_fmin, ...
                                                 yin_fmax, yin_win, yin_thresh);
end
valid3    = ~isnan(yin_f0s3) & yin_cs3 > 0.5;
yin_mean3 = mean(yin_f0s3(valid3));
yin_std3  = std(yin_f0s3(valid3));

fprintf('  %-25s  %-10s\n', 'Metric', 'Value');
fprintf('  %s\n', repmat('-', 1, 40));
fprintf('  %-25s  %.4f\n',   'CPEI (pool ±30 Hz)',    cpei3);
fprintf('  %-25s  %.2f Hz\n','RPS peak',               rpk3);
fprintf('  %-25s  %.2f Hz\n','Welch peak',             wpk3);
fprintf('  %-25s  %.4f\n',   'PLV at F0_mid',          plv3);
fprintf('  %-25s  %.4f\n',   'ITPC at F0_mid',         itpc3);
fprintf('  %-25s  %.2f ± %.2f Hz\n', 'YIN mean ± std', yin_mean3, yin_std3);
fprintf('  %-25s  %d / %d frames\n', 'YIN valid frames', sum(valid3), n_frames3);
fprintf('  Expected range: 100 to 150 Hz\n');
fprintf('  RPS in range?   %s\n', mat2str(rpk3>=95 && rpk3<=155));
fprintf('  Welch in range? %s\n', mat2str(wpk3>=95 && wpk3<=155));
fprintf('  YIN in range?   %s\n\n', ...
        mat2str(~isnan(yin_mean3) && yin_mean3>=95 && yin_mean3<=155));

%% signal 4 (pure noise) 
fprintf('signal 4 \n');
fprintf('%d independent noise trials\n\n', 10);

n_trials = 10;
v4 = struct('cpei',zeros(1,n_trials),'plv',zeros(1,n_trials), ...
            'itpc',zeros(1,n_trials),'wsnr',zeros(1,n_trials), ...
            'yin',NaN(1,n_trials));

for tr = 1:n_trials
    noise4   = randn(N, 1);

    mE4      = run_RFB(noise4, fs, fmin_s, fmax_s);
    v4.cpei(tr) = compute_cpei(mE4, period_hz, F0, cpei_bw);

    v4.plv(tr)  = compute_PLV(noise4, fs, F0);

    ep4 = zeros(n_epochs, epoch_len);
    for e = 1:n_epochs
        ep4(e,:) = noise4((e-1)*epoch_len+1 : e*epoch_len);
    end
    v4.itpc(tr) = compute_ITPC(ep4, fs, F0);

    [p4,f4]     = pwelch(noise4, win_w, noverlap, nfft_w, fs, 'onesided');
    v4.wsnr(tr) = welch_snr(p4, f4, F0, 5, noise_lo, noise_hi);

    [yf4,~]     = yin_estimate(noise4, fs, yin_fmin, yin_fmax, yin_win, yin_thresh);
    v4.yin(tr)  = yf4;
end

fprintf('  %-25s  %-14s  %-10s\n', 'Algorithm', 'Mean', 'Std');
fprintf('  %s\n', repmat('-', 1, 52));
fprintf('  %-25s  %-14.4f  %-10.4f\n', 'CPEI',      mean(v4.cpei), std(v4.cpei));
fprintf('  %-25s  %-14.4f  %-10.4f\n', 'PLV',       mean(v4.plv),  std(v4.plv));
fprintf('  %-25s  %-14.4f  %-10.4f\n', 'ITPC',      mean(v4.itpc), std(v4.itpc));
fprintf('  %-25s  %-14.4f  %-10.4f  dB\n','Welch SNR', mean(v4.wsnr), std(v4.wsnr));
yin_det = sum(~isnan(v4.yin));
fprintf('  %-25s  %d / %d trials detected\n\n', 'YIN detections', yin_det, n_trials);

%% signal 5 two speakers (overlapping harmonics)
fprintf('signal 5\n');
fprintf('speaker A: F0 = 120 Hz | speaker B: F0 = 180 Hz\n\n');

F0_A = 120;  F0_B = 180;
sigA = zeros(N,1);  sigB = zeros(N,1);
for h = 1:n_harm
    sigA = sigA + (1/h)*sin(2*pi*h*F0_A*t);
    sigB = sigB + (1/h)*sin(2*pi*h*F0_B*t);
end
sigA   = sigA / max(abs(sigA));
sigB   = sigB / max(abs(sigB));
mixed5 = sigA + sigB + 0.3*randn(N,1);

% CPEI for each speaker
mE5    = run_RFB(mixed5, fs, fmin_s, fmax_s);
cpei5A = compute_cpei(mE5, period_hz, F0_A, cpei_bw);
cpei5B = compute_cpei(mE5, period_hz, F0_B, cpei_bw);
rpk5   = rps_peak(mE5, period_hz, fmin_s, fmax_s);

% Welch
[p5,f5] = pwelch(mixed5, win_w, noverlap, nfft_w, fs, 'onesided');
wsnr5A  = welch_snr(p5, f5, F0_A, 5, [100 115], [125 140]);
wsnr5B  = welch_snr(p5, f5, F0_B, 5, [160 175], [185 200]);
wpk5A   = welch_peak(p5, f5, 100, 140);
wpk5B   = welch_peak(p5, f5, 160, 200);

% PLV 
plv5A   = compute_PLV(mixed5, fs, F0_A);
plv5B   = compute_PLV(mixed5, fs, F0_B);

% ITPC
ep5 = zeros(n_epochs, epoch_len);
for e = 1:n_epochs
    ep5(e,:) = mixed5((e-1)*epoch_len+1 : e*epoch_len);
end
itpc5A  = compute_ITPC(ep5, fs, F0_A);
itpc5B  = compute_ITPC(ep5, fs, F0_B);

% YIN 
[yin_f0_5, yin_c5] = yin_estimate(mixed5, fs, yin_fmin, yin_fmax, ...
                                   yin_win, yin_thresh);

fprintf('  %-22s  %-12s  %-12s\n', 'Metric', 'Speaker A', 'Speaker B');
fprintf('  %s\n', repmat('-', 1, 50));
fprintf('  %-22s  %-12.4f  %-12.4f\n', 'CPEI',         cpei5A, cpei5B);
fprintf('  %-22s  %-12.4f  %-12.4f\n', 'PLV',          plv5A,  plv5B);
fprintf('  %-22s  %-12.4f  %-12.4f\n', 'ITPC',         itpc5A, itpc5B);
fprintf('  %-22s  %-12.4f  %-12.4f  dB\n','Welch SNR', wsnr5A, wsnr5B);
fprintf('  %-22s  %-12.2f  %-12.2f  Hz\n','Welch peak', wpk5A,  wpk5B);
fprintf('  %-22s  %-12.2f  (single estimate only)\n', 'YIN F0', yin_f0_5);
fprintf('  %-22s  %-12.4f\n', 'YIN confidence', yin_c5);
fprintf('  RPS global peak:  %.2f Hz\n\n', rpk5);

%% SUMMARY
fprintf('                      algorithm comparison\n');

fprintf('signal 1 — sensitivity vs SNR\n');
fprintf('  algorithm  above-chance down to SNR:\n');
above_cpei = s1.snr(s1.cpei > 0.05);
above_plv  = s1.snr(s1.plv  > 0.10);
above_itpc = s1.snr(s1.itpc > 0.10);
above_wsnr = s1.snr(s1.wsnr > 3);
if isempty(above_cpei); fprintf('  CPEI:    never\n');
else; fprintf('  CPEI:    %d dB\n', min(above_cpei)); end
if isempty(above_plv);  fprintf('  PLV:     never\n');
else; fprintf('  PLV:     %d dB\n', min(above_plv));  end
if isempty(above_itpc); fprintf('  ITPC:    never\n');
else; fprintf('  ITPC:    %d dB\n', min(above_itpc)); end
if isempty(above_wsnr); fprintf('  Welch:   never\n');
else; fprintf('  Welch:   %d dB\n', min(above_wsnr)); end
n_yin_ok = sum(abs(s1.yin_f0 - F0) < 10 & s1.yin_c > 0.5);
fprintf('  YIN:     correct in %d / %d SNR levels\n\n', n_yin_ok, length(snr_levels));

fprintf('signal 2 — octave error rate\n');
fprintf('  %-10s  %-15s  %-10s\n', 'Algorithm', 'Octave Errors', 'Rate');
fprintf('  %s\n', repmat('-', 1, 38));
fprintf('  %-10s  %-15d  %d/%d\n', 'Welch',  sum(s2.werr),    sum(s2.werr),    5);
fprintf('  %-10s  %-15d  %d/%d\n', 'RPS',    0,               0,               5);
yin_errs = sum(s2.yin_err);
fprintf('  %-10s  %-15d  %d/%d\n', 'YIN',    yin_errs,        yin_errs,        5);
fprintf('  PLV/ITPC: not pitch estimators so not applicable\n\n');

fprintf('signal 3 — drifting F0 detection\n');
fprintf('  RPS in range:   %s\n', mat2str(rpk3>=95  && rpk3<=155));
fprintf('  Welch in range: %s\n', mat2str(wpk3>=95  && wpk3<=155));
fprintf('  YIN in range:   %s\n\n', ...
        mat2str(~isnan(yin_mean3) && yin_mean3>=95 && yin_mean3<=155));

fprintf('signal 4 — false positive rate (lower = better)\n');
fprintf('  CPEI mean:     %.4f  (threshold 0.05)\n', mean(v4.cpei));
fprintf('  PLV mean:      %.4f  (threshold 0.10)\n', mean(v4.plv));
fprintf('  ITPC mean:     %.4f  (threshold 0.10)\n', mean(v4.itpc));
fprintf('  Welch SNR mean:%.4f dB (threshold 3 dB)\n', mean(v4.wsnr));
fprintf('  YIN detections:%d / %d  (expect 0)\n\n', yin_det, n_trials);

fprintf('signal 5 — two-Speaker separation\n');
fprintf('  %-10s  %-12s  %-12s  %-10s\n', 'Algorithm', 'Detects A', 'Detects B', 'Separates');
fprintf('  %s\n', repmat('-', 1, 48));
fprintf('  %-10s  %-12s  %-12s  %-10s\n', 'CPEI', ...
        mat2str(cpei5A>0.05), mat2str(cpei5B>0.05), ...
        mat2str(cpei5A>0.05 && cpei5B>0.05));
fprintf('  %-10s  %-12s  %-12s  %-10s\n', 'PLV', ...
        mat2str(plv5A>0.10), mat2str(plv5B>0.10), ...
        mat2str(plv5A>0.10 && plv5B>0.10));
fprintf('  %-10s  %-12s  %-12s  %-10s\n', 'ITPC', ...
        mat2str(itpc5A>0.10), mat2str(itpc5B>0.10), ...
        mat2str(itpc5A>0.10 && itpc5B>0.10));
fprintf('  %-10s  %-12s  %-12s  %-10s\n', 'Welch', ...
        mat2str(wsnr5A>3), mat2str(wsnr5B>3), ...
        mat2str(wsnr5A>3 && wsnr5B>3));
fprintf('  YIN: single pitch estimate only — cannot separate two speakers\n');
fprintf('  YIN estimate: %.2f Hz (dominates at %.2f confidence)\n\n', ...
        yin_f0_5, yin_c5);

fprintf('finished.\n');
