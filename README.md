# Long-Frame-Shift Neural Speech Phase Prediction With Spectral Continuity Enhancement and Interpolation Error Compensation
### Yang Ai, Ye-Xin Lu, Zhen-Hua Ling

In our [paper](https://arxiv.org/abs/2308.08850), 
we proposed a novel speech phase prediction method which predicts long-frame-shift wrapped phase spectra from amplitude spectra by combining neural networks and signal processing knowledges.<br/>
We provide our implementation as open source in this repository.

**Abstract :**
Speech phase prediction, which is a significant research focus in the field of signal processing, aims to recover speech phase spectra from amplitude-related features. However, existing speech phase prediction methods are constrained to recovering phase spectra with short frame shifts, which are considerably smaller than the theoretical upper bound required for exact waveform reconstruction of short-time Fourier transform (STFT). To tackle this issue, we present a novel long-frame-shift neural speech phase prediction (LFS-NSPP) method which enables precise prediction of long-frame-shift phase spectra from long-frame-shift log amplitude spectra. The proposed method consists of three stages: interpolation, prediction and decimation. The short-frame-shift log amplitude spectra are first constructed from long-frame-shift ones through frequency-by-frequency interpolation to enhance the spectral continuity, and then employed to predict short-frame-shift phase spectra using an NSPP model, thereby compensating for interpolation errors. Ultimately, the long-frame-shift phase spectra are obtained from short-frame-shift ones through frame-by-frame decimation. Experimental results show that the proposed LFS-NSPP method can yield superior quality in predicting long-frame-shift phase spectra than the original NSPP model and other signal-processing-based phase estimation algorithms.

Visit our [demo website](https://yangai520.github.io/LFS-NSPP) for audio samples.

## Requirements
```
torch==1.8.1+cu111
numpy==1.21.6
librosa==0.9.1
tensorboard==2.8.0
soundfile==0.10.3
matplotlib==3.1.3
```

## Data Preparation
For training, write the list paths of training set and validation set to `input_training_wav_list` and `input_validation_wav_list` in `config.json`, respectively.

For generation, we provide two ways to read data:

(1) set `test_input_log_amp_dir` to `0` in `config.json` and write the test set waveform path to `test_input_wavs_dir` in `config.json`, the generation process will load the waveform, extract the log amplitude spectra, predict the phase spectra and reconstruct the waveform;

(2) set `test_input_log_amp_dir` to `1` in `config.json` and write the log amplitude spectra (size is `(n_fft/2+1)*frames`) path to `test_input_log_amp_dir` in `config.json`, the generation process will dierctly load the log amplitude spectra, predict the phase spectra and reconstruct the waveform.

In `config.json`, `hop_size` is the long-frame-shift (unit: sample points) and `transit_hop_size` is the short-frame-shift (unit: sample points).

## Training
Run using GPU:
```
CUDA_VISIBLE_DEVICES=0 python train.py
```
Using TensorBoard to monitor the training process:
```
tensorboard --logdir=cp_NSPP/logs
```

## Generation:
Write the checkpoint path to `checkpoint_file_load` in `config.json`.

Run using GPU:
```
CUDA_VISIBLE_DEVICES=0 python generation.py
```
Run using CPU:
```
CUDA_VISIBLE_DEVICES=CPU python generation.py
```

## Citation
```
@article{ai2023long,
  title={Long-frame-shift Neural Speech Phase Prediction with Spectral Continuity Enhancement and Interpolation Error Compensation},
  author={Ai, Yang and Lu, Ye-Xin and Ling, Zhen-Hua},
  journal={IEEE Signal Processing Letters},
  year={2023}
}
```
