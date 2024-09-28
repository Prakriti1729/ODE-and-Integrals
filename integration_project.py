import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

r_inner = float(input("Enter the value of inner radius"))
r_outer = float(input("Enter the value of outer radius"))
M = float(input("Enter the total mass of the disk"))
rho = M / (np.pi * (r_outer**2 - r_inner**2)) #mass density
print("value of rho: ", rho)
num_steps = int(input("Enter the number of steps"))
dr = (r_outer - r_inner) / num_steps

# Derivation
# I = ∫(r_inner to r_outer) r^2 dm
# dm = ρ dA = ρ (2πr dr)
# Substitute dm into I:
# I = ∫(r_inner to r_outer) r^2 (ρ (2πr dr))
# I = ∫(r_inner to r_outer) 2πρ r^3 dr

#Analytical solution
I_a = M*(r_inner**2+r_outer**2)/2
print("Calculated Moment of Inertia (I) using Analytical formula: ",I_a)

#Riemann Sum
I_r = 0
for i in range(num_steps):
    r = r_inner + i * dr 
    I_r += 2 * np.pi * rho * r**3 * dr
print("Calculated Moment of Inertia (I) using Riemann sum: ",+I_r)


#Trapezoid
I_t = 0
for i in range(num_steps):
    r1 = r_inner + i * dr 
    r2 = r_inner + (i + 1) * dr 
    I_t += 0.5 * (2 * np.pi * rho * r1**3 + 2 * np.pi * rho * r2**3) * dr
print("Calculated Moment of Inertia (I) Trapezoidal method: ",+I_t)


#Simpson's rule
I_s = 0
for i in range(0, num_steps-1, 2):
    r1 = r_inner + i * dr  # Left endpoint
    r2 = r_inner + (i + 1) * dr  # Midpoint
    r3 = r_inner + (i + 2) * dr  # Right endpoint

    I_s += (dr / 3) * (2 * np.pi * rho * r1**3 + 
                     8 * np.pi * rho * r2**3 + 
                     2 * np.pi * rho * r3**3)

print("Calculated Moment of Inertia (I) using Simpson's method: ", +I_s)

#Scipy
def integrand(r):
    return r**2 * 2 * np.pi * r * rho

I_sci, _ = quad(integrand, r_inner, r_outer)
    
print("Calculated Moment of Inertia (I) using Scipy: ", I_sci )

#error analyisis

# Step sizes for error analysis
step_sizes = [100, 200, 400, 800, 1600]
errors_riemann = []
errors_trapezoidal = []
errors_simpson = []
errors_scipy = []
truncation_errors = {
    'Riemann': [],
    'Trapezoidal': [],
    'Simpson': []
}

# Calculate errors for different step sizes
for num_steps in step_sizes:

    error_riemann = abs(I_r - I_a)
    error_trapezoidal = abs(I_t - I_a)
    error_simpson = abs(I_s - I_a)
    error_scipy = abs(I_sci - I_a)

    errors_riemann.append(error_riemann)
    errors_trapezoidal.append(error_trapezoidal)
    errors_simpson.append(error_simpson)
    errors_scipy.append(error_scipy)

    # Calculate expected truncation errors
    dt = (r_outer - r_inner) / num_steps
    truncation_errors['Riemann'].append(dt)
    truncation_errors['Trapezoidal'].append(dt**2)
    truncation_errors['Simpson'].append(dt**4)

# Plotting the errors
plt.loglog(step_sizes, errors_riemann, label='Riemann Sum Error')
plt.loglog(step_sizes, errors_trapezoidal, label='Trapezoidal Error')
plt.loglog(step_sizes, errors_simpson, label='Simpson\'s Error')

# Plotting the expected truncation errors
plt.loglog(step_sizes, truncation_errors['Riemann'], label='Riemann Truncation', linestyle='--')
plt.loglog(step_sizes, truncation_errors['Trapezoidal'], label='Trapezoidal Truncation', linestyle='--')
plt.loglog(step_sizes, truncation_errors['Simpson'], label='Simpson Truncation', linestyle='--')

plt.title('Error and Truncation Error Scaling for Moment of Inertia Calculation')
plt.xlabel('Number of Steps')
plt.ylabel('Error')
plt.grid()
plt.legend()
plt.show()
