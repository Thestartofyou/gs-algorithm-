import numpy as np
import matplotlib.pyplot as plt

def gerchberg_saxton_algorithm(original_signal, iterations):
    # Assume we have access to the Fourier transform of the original signal
    fourier_transform = np.fft.fft(original_signal)

    # Random initialization of the phase
    reconstructed_signal = np.random.rand(len(original_signal)) * np.exp(1j * np.random.rand(len(original_signal)))

    for _ in range(iterations):
        # Step 1: Fourier transform of the current guess
        guess_fourier = np.fft.fft(reconstructed_signal)

        # Step 2: Replace amplitude with the original amplitude
        guess_fourier = np.abs(fourier_transform) * np.exp(1j * np.angle(guess_fourier))

        # Step 3: Inverse Fourier transform to get the updated signal guess
        reconstructed_signal = np.fft.ifft(guess_fourier)

    return reconstructed_signal

# Example usage:
# Create a simple signal (e.g., a combination of sine waves)
original_signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 100)) + 0.5 * np.sin(2 * np.pi * 20 * np.linspace(0, 1, 100))

# Add noise to the signal
noisy_signal = original_signal + 0.1 * np.random.normal(size=len(original_signal))

# Reconstruct the signal using the Gerchberg-Saxton algorithm
iterations = 100
reconstructed_signal = gerchberg_saxton_algorithm(noisy_signal, iterations)

# Plot the results
plt.figure(figsize=(10, 5))
plt.subplot(3, 1, 1)
plt.title('Original Signal')
plt.plot(original_signal)

plt.subplot(3, 1, 2)
plt.title('Noisy Signal')
plt.plot(noisy_signal)

plt.subplot(3, 1, 3)
plt.title('Reconstructed Signal')
plt.plot(reconstructed_signal.real)
plt.show()

