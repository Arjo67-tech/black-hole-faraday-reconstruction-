import numpy as np

# Constants
c = 2.998e8  # speed of light in m/s
RM = -5e5    # rotation measure in rad/m^2

# Frequencies in Hz
frequencies = [86e9, 213e9, 229e9, 230e9, 345e9]

# Calculate lambda and lambda^2 for each frequency
lambda_values = c / np.array(frequencies)
lambda_squared_values = lambda_values ** 2

# Print lambda and lambda^2 for each frequency
for f, lam, lam2 in zip(frequencies, lambda_values, lambda_squared_values):
    print(f"Frequency: {f} Hz")
    print(f"Lambda: {lam:.6e} m")
    print(f"Lambda^2: {lam2:.6e} m^2")

# Calculate Faraday rotation angle RM*lambda^2 in degrees for each frequency
faraday_rotation_angles = RM * lambda_squared_values / (np.pi / 180)

# Print the Faraday rotation angle in degrees for each frequency
for f, angle in zip(frequencies, faraday_rotation_angles):
    print(f"Faraday Rotation Angle at {f} Hz: {angle:.2f} degrees")

# Calculate the difference in rotation between the 213 and 229 GHz ALMA sidebands in degrees
index_213 = frequencies.index(213e9)
index_229 = frequencies.index(229e9)
rotation_difference = faraday_rotation_angles[index_213] - faraday_rotation_angles[index_229]

# Print the difference in rotation between the 213 and 229 GHz ALMA sidebands in degrees
print(f"Difference in Rotation between 213 and 229 GHz: {rotation_difference:.2f} degrees")
