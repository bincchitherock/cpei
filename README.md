# chirp-speech periodicity entrainment index (cpei)
derived from projections onto ramanujan periodic subspaces (rps), for measuring neural temporal tracking in auditory eeg. 

---

## background

when a person listens to speech, their brain tracks the rhythm and pitch of the speaker's voice. this tracking degrades in people with hearing loss or age-related auditory decline. cpei is a scalar metric (0 to 1) that quantifies how strongly a neural signal is entrained to a target modulation frequency. and unlike standard fourier-based methods, it should be robust to the octave error problem that arises when hearing loss preferentially attenuates the fundamental frequency relative to its harmonics.

---

## the octave error problem

standard tools like the welch periodogram treat each frequency bin independently. when hearing loss weakens the brain's response at the fundamental frequency f0, the dominant peak migrates to 2×f0 (the first harmonic), causing a systematic octave error. rps avoids this since all harmonics of a fundamental period t collapse into the same ramanujan subspace, so the method accumulates energy from the entire harmonic series regardless of which individual harmonic is strongest.

---

## algorithms compared

| algorithm | description |
|---|---|
| **cpei / rps** | proposed method. ramanujan filter bank energy concentration index. |
| **welch** | standard power spectral density estimator. nonparametric baseline. |
| **plv** | phase locking value. measures phase coherence against a fixed reference frequency. |
| **itpc** | inter-trial phase coherence. epoch-based equivalent of plv. |
| **yin** | pitch estimator (de cheveigné & kawahara 2002). f0 ground truth reference. |

---

## synthetic signals tested

| signal | description | tests |
|---|---|---|
| 1. clean harmonic ffr | f0 = 120 hz, 8 harmonics, snr swept +20 to −20 db | sensitivity |
| 2. attenuated fundamental | fundamental attenuated from 100% to 5% | octave error immunity |
| 3. drifting f0 | pitch drifts 100 → 150 hz over 5 s | quasi-periodic robustness |
| 4. pure noise | white gaussian noise, 10 trials | false positive rate |
| 5. overlapping harmonics | two speakers at 120 hz and 180 hz mixed | multi-source separation |

---

## key results

### octave error (signal 2 — hearing loss simulation)
| algorithm | octave errors |
|---|---|
| welch | 3 / 5 conditions |
| **rps** | **0 / 5 conditions** |
| yin | 0 / 5 (but produced no estimates at all) |

### sensitivity (signal 1 — clean harmonic ffr)
| algorithm | detectable down to snr |
|---|---|
| plv | −10 db |
| itpc | −10 db |
| welch | −10 db |
| cpei | 0 db |
| yin | +20 db only |

### drifting pitch (signal 3)
| algorithm | result |
|---|---|
| cpei | detected (cpei = 0.47) |
| welch | detected |
| itpc | partial |
| plv | near noise floor |
| yin | 0 / 497 valid frames |

### noise floor (signal 4 — false positive control)
| algorithm | noise floor |
|---|---|
| cpei | 0.040 ± 0.0003 |
| plv | 0.006 ± 0.004 |
| itpc | 0.089 ± 0.003 |
| welch snr | 0.22 ± 0.77 db |
| yin | no detections |

---

## requirements

- matlab r2019b or later
- signal processing toolbox (`pwelch`, `hilbert`)

---

## usage

### synthetic validation
```matlab
