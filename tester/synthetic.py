import numpy as np
import matplotlib.pyplot as plt


def showSign(wl, sign, label: str):
    plt.plot(wl, sign, label=label)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.legend()
    plt.show()


# Example: Creating synthetic data for two materials (vegetation and soil)

# USGS spectral signatures (hypothetical values)
wavelengths = np.arange(400, 2501, 10)
vegetation_signature = np.sin(2 * np.pi * wavelengths / 700)
soil_signature = np.exp(-(wavelengths - 1200) ** 2 / (2 * 50 ** 2))

# Number of pixels in the synthetic scene
num_pixels = 1000

# Proportions of materials in the scene
vegetation_proportion = 0.6
soil_proportion = 0.4

# Generate synthetic spectral signatures
synthetic_signature = (
        vegetation_proportion * vegetation_signature
        + soil_proportion * soil_signature
)

# Repeat the synthetic signature to create a hyperspectral scene
synthetic_data = np.tile(synthetic_signature, (num_pixels, 1))

# Add noise (optional)
synthetic_data += np.random.normal(scale=0.05, size=synthetic_data.shape)

# Visualize the synthetic spectral signature
showSign(wavelengths,vegetation_signature,"vegitation sign ")
showSign(wavelengths,soil_signature, "soil sign ")
showSign(wavelengths,synthetic_signature,"Synthetic Signature")
