'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np
import soundfile as sf


def stft(signal, window_size=2048, hop_size=512):

    num_frames = 1 + (len(signal) - window_size) // hop_size
    stft_matrix = np.empty(( num_frames,window_size // 2 + 1), dtype=np.complex128)

    for t in range(num_frames):
        start = t * hop_size
        end = start + window_size
        frame = signal[start:end]*np.hanning(window_size)
        frame_fft = np.fft.fft(frame, n=window_size)
        stft_matrix[t,:] = frame_fft[:window_size // 2 + 1]

    return stft_matrix


def istft(stft_matrix, hop_size=512):
    
    num_windows, window_size = stft_matrix.shape

    
    signal_length = (num_windows - 1) * hop_size + (window_size-1)*2
    signal = np.zeros(signal_length, dtype=np.complex128)

    for i in range(num_windows):
        istft_window = np.fft.irfft(stft_matrix[i], n=(window_size-1)*2)
       # istft_window = istft_window/np.hanning(window_size)
        start = i * hop_size

        signal[start:start + (window_size-1)*2] += istft_window

    signal = np.real(signal)

    return signal




def griffin_lim(magnitude_spectrogram, phase=None, n_iter=100, window_size=2048, hop_size=512):
    # Get the number of time frames and frequency bins from the magnitude spectrogram
    num_frames, num_bins = magnitude_spectrogram.shape
    
    # Initialize random phase if not provided
    if phase is None:
        phase = np.random.uniform(0, 2 * np.pi, size=(num_frames, num_bins))
    print((magnitude_spectrogram * np.exp(1j * phase)).shape)
    print(magnitude_spectrogram.shape)
    print(phase.shape)
    for i in range(n_iter):
        # Inverse STFT to reconstruct phase
        estimated_signal = istft(magnitude_spectrogram * np.exp(1j * phase), hop_size=hop_size)
        
        # STFT to estimate phase and magnitude
        stft_output = stft(estimated_signal, window_size=window_size, hop_size=hop_size)
        estimated_phase = np.angle(stft_output)
        
        phase=estimated_phase
   
    estimated_signal = istft(magnitude_spectrogram * np.exp(1j * phase), hop_size=hop_size) 
    
    return estimated_signal



def time_stretch_griffin_lim(magnitude_spectrogram, stretch_factor, phase=None, n_iter=100, window_size=2048, hop_size=512):
    
    new_num_freq_bins = int(stretch_factor * magnitude_spectrogram.shape[0])
    stretched_magnitude = np.zeros((new_num_freq_bins,magnitude_spectrogram.shape[1]))
    
    indices_freq_bins = np.linspace(0,magnitude_spectrogram.shape[0],new_num_freq_bins,endpoint=False) 

    for i in range(magnitude_spectrogram.shape[1]):
        stretched_magnitude[ :,i] = np.interp(indices_freq_bins, np.arange(magnitude_spectrogram.shape[0]), magnitude_spectrogram[ :,i])

    
    reconstructed_signal = griffin_lim(stretched_magnitude, phase, n_iter, window_size, hop_size)
    
    return reconstructed_signal
    

def freq_stretch_griffin_lim(magnitude_spectrogram, stretch_factor, phase=None, n_iter=100, window_size=2048, hop_size=512):
   

    num_bins = magnitude_spectrogram.shape[1]
    stretched_freq_bins = np.linspace(0, num_bins-1, int(num_bins * stretch_factor))

    stretched_magnitude = np.zeros((magnitude_spectrogram.shape[0], len(stretched_freq_bins)))
    for i in range(magnitude_spectrogram.shape[0]):
        stretched_magnitude[i, :] = np.interp(stretched_freq_bins, np.arange(num_bins), magnitude_spectrogram[i, :])

    if stretch_factor > 1:
        stretched_magnitude = stretched_magnitude[:, :num_bins]
    elif stretch_factor < 1:
        padding_length = num_bins - stretched_magnitude.shape[1]
        stretched_magnitude = np.pad(stretched_magnitude, ((0, 0), (0, padding_length)), 'constant')

    reconstructed_signal = griffin_lim(stretched_magnitude, phase, n_iter, window_size, hop_size)

    return reconstructed_signal
