import numpy as np
from scipy.fft import fft, ifft

# 对信号进行FFT变换
def preprocess_signal(signal):
    fft_signal = fft(signal)
    real_part = np.real(fft_signal)
    imag_part = np.imag(fft_signal)
    return real_part, imag_part

# 将GRU输出通过IFFT变换回时域
def inverse_fft(real_part, imag_part):
    fft_output = real_part + 1j * imag_part
    return np.real(ifft(fft_output))
