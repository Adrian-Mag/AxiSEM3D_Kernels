import numpy as np
import time
import scipy.integrate as spi

# Define the function to integrate (replace with your own function)
def func_helper(radial_pos, latitude, longitude):
    return func([radial_pos, latitude, longitude])

def func(point):
    # Extract the radial position, latitude, and longitude from the point
    radial_pos, latitude, longitude = point

    # Compute the value of the function at the given point
    return radial_pos * np.deg2rad(latitude) * np.deg2rad(longitude)

# Define the limits of integration
radial_min, radial_max = 3400000, 6371000
lat_min, lat_max = 0, 90
lon_min, lon_max = 0, 90

# Number of points for the Monte Carlo method
num_points = 1000000

# Method 1: Monte Carlo Integration
start_time = time.time()

# Generate random points within the integration limits
random_points = np.random.uniform(
    [radial_min, lat_min, lon_min],
    [radial_max, lat_max, lon_max],
    size=(num_points, 3)
)

# Evaluate the function at the random points
func_values = func(random_points.T)

# Compute the volumetric integral using the Monte Carlo method
volumetric_integral_mc = np.mean(func_values) * (radial_max - radial_min) * (lat_max - lat_min) * (lon_max - lon_min)

end_time = time.time()
execution_time_mc = end_time - start_time

print("Monte Carlo Integration:")
print("Volumetric Integral:", volumetric_integral_mc)
print("Execution Time:", execution_time_mc)

# Method 2: Numerical Integration
start_time = time.time()

# Perform numerical integration using scipy
result, error = spi.tplquad(func_helper, lon_min, lon_max, lat_min, lat_max, radial_min, radial_max)

volumetric_integral_num = result

end_time = time.time()
execution_time_num = end_time - start_time

print("Numerical Integration:")
print("Volumetric Integral:", volumetric_integral_num)
print("Execution Time:", execution_time_num)

# Compare the results
accuracy = np.abs(volumetric_integral_mc - volumetric_integral_num)
print("Accuracy:", accuracy)
