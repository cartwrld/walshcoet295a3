
import numpy as np
import matplotlib.pyplot as plt

# Define signal parameters
T = 1  # period (seconds)
fs = 1000  # sampling frequency (Hz)
time = np.linspace(0, T, T * fs, endpoint=False)

# Generate sine wave
frequency_sine = 5  # frequency (Hz)
sine_wave = np.sin(2 * np.pi * frequency_sine * time)

# Generate square wave
frequency_square = 2  # frequency (Hz)
duty_cycle = 0.01  # percentage of the period where the signal is high
square_wave = np.where(
    np.mod(np.floor(2 * duty_cycle * fs * time), 2) == 0, 1, -1)

# Generate triangle wave
frequency_triangle = 1  # frequency (Hz)
triangle_wave = 2 * np.abs((10 * frequency_triangle * time) % 2 - 1) - 1

# Generate sawtooth wave
frequency_sawtooth = 10  # frequency (Hz)
sawtooth_wave = 2 * (frequency_sawtooth * time -
                     np.floor(frequency_sawtooth * time + 0.5))

# Plot the waves
fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 8))
axs[0].plot(time, sine_wave)
axs[0].set_title('Sine Wave')
axs[1].plot(time, square_wave)
axs[1].set_title('Square Wave')
axs[2].plot(time, triangle_wave)
axs[2].set_title('Triangle Wave')
axs[3].plot(time, sawtooth_wave)
axs[3].set_title('Sawtooth Wave')
plt.xlabel('Time (s)')
plt.show()