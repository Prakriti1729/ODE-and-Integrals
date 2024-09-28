import numpy as np
import matplotlib.pyplot as plt

#Euler's method
def euler_method(m, k_eq, x0, v0, dt, num_steps):
    x = np.zeros(num_steps)
    v = np.zeros(num_steps)
    t = np.zeros(num_steps)
    KE = np.zeros(num_steps)
    PE = np.zeros(num_steps)
    TE = np.zeros(num_steps)
 
   # Initial conditions
    x[0] = x0
    v[0] = v0

    # Euler's method loop
    for i in range(1, num_steps):
        a = -k_eq * x[i-1] / m  
        v[i] = v[i-1] + a * dt   
        x[i] = x[i-1] + v[i-1] * dt  
        t[i] = t[i-1] + dt  
        KE[i] = 0.5 * m * v[i]**2
        PE[i] = 0.5 * k_eq * x[i]**2
        TE[i] = KE[i] + PE[i]

    return t, x, v, KE, PE, TE

#RK4 Method
def runge_kutta_4(m, k_eq, x0, v0, dt, num_steps):
    x = np.zeros(num_steps)
    v = np.zeros(num_steps)
    t = np.zeros(num_steps)
    KE = np.zeros(num_steps)
    PE = np.zeros(num_steps)
    TE = np.zeros(num_steps)

    # Initial conditions
    x[0] = x0
    v[0] = v0

    # Runge-Kutta method loop
    for i in range(num_steps - 1):
        a = -k_eq * x[i] / m  

        # RK4 coefficients
        k1_v = a
        k1_x = v[i]
        
        k2_v = (-k_eq * (x[i] + 0.5 * dt * k1_x)) / m
        k2_x = v[i] + 0.5 * dt * k1_v
        
        k3_v = (-k_eq * (x[i] + 0.5 * dt * k2_x)) / m
        k3_x = v[i] + 0.5 * dt * k2_v
        
        k4_v = (-k_eq * (x[i] + dt * k3_x)) / m
        k4_x = v[i] + dt * k3_v

        x[i + 1] = x[i] + (dt / 6) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        v[i + 1] = v[i] + (dt / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        t[i + 1] = t[i] + dt  
        
        KE[i + 1] = 0.5 * m * v[i + 1]**2
        PE[i + 1] = 0.5 * k_eq * x[i + 1]**2
        TE[i + 1] = KE[i + 1] + PE[i + 1]

    return t, x, v, KE, PE, TE


# User input for spring configuration
configuration = input("Are the springs in series or parallel? ").strip().lower()

# Input for spring constants and mass
k1 = float(input("Enter the spring constant k1: "))
k2 = float(input("Enter the spring constant k2: "))
m = float(input("Enter the mass m: "))


# Calculate equivalent spring constant
if configuration == "series":
    if k1!=0 and k2!=0:
        k_eq = 1 / (1 / k1 + 1 / k2)
    elif k1==0 and k2!=0:
        k_eq = k2
    elif k2==0 and k1!=0:
        k_eq = k1
    elif k1==0 and k2==0:
        k_eq = 0
    equation = "mx'' = -k_eq * x"
elif configuration == "parallel":
    k_eq = k1 + k2
    print(f"Equivalent spring constant (parallel): k_eq = {k_eq}")
    equation = "mx'' = -k_eq * x"
else:
    print("Invalid input. Please enter 'series' or 'parallel'.")
    exit()

print(f"Equation of motion: {equation}")

# Initial conditions and simulation parameters
x0 = float(input("Enter initial displacement x0: "))
v0 = float(input("Enter initial velocity v0: "))
dt = 0.01  # Time step
num_steps = 5000  # Number of time steps

# Calculate analytical solution
omega = np.sqrt(k_eq / m)
A = np.sqrt(x0**2 + (v0/omega)**2)  # Amplitude
phi =-np.arctan(v0/x0*omega)    # Phase shift
t_analytical = np.linspace(0, dt * num_steps, num_steps)
x_analytical = A * np.cos(omega * t_analytical + phi)


# Solve the ODE using Euler's method
t_e, x_e, v_e, KE_e, PE_e, TE_e = euler_method(m, k_eq, x0, v0, dt, num_steps)

# Solve the ODE using the Runge-Kutta 4th-order method
t_r, x_r, v_r, KE_r, PE_r, TE_r = runge_kutta_4(m, k_eq, x0, v0, dt, num_steps)

# Plot results
plt.plot(t_e, x_e, label='Euler method')
plt.plot(t_r, x_r, label='4th order Runge Kutta method')
plt.plot(t_analytical, x_analytical, label='Analytical Solution')
plt.title('Mass-Spring System')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.grid()
plt.legend()
plt.show()

plt.plot(t_e, KE_e, label='Kinetic Energy (KE)', color='green')
plt.plot(t_e, PE_e, label='Potential Energy (PE)', color='red')
plt.plot(t_e, TE_e, label='Total Energy (TE)', color='blue')
plt.title('Energy plot for Euler method')
plt.xlabel('Time (s)')
plt.ylabel('Energy (J)')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

plt.plot(t_r, KE_r, label='Kinetic Energy (KE)', color='green')
plt.plot(t_r, PE_r, label='Potential Energy (PE)', color='red')
plt.plot(t_r, TE_r, label='Total Energy (TE)', color='blue')
plt.title('Energy plot for RK4 method')
plt.xlabel('Time (s)')
plt.ylabel('Energy (J)')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

# Step sizes for error analysis
step_sizes = [0.1, 0.05, 0.025, 0.0125]
errors_euler = []
errors_rk4 = []

# Calculate errors for different step sizes
for dt in step_sizes:
    # Euler Method
    error_euler = np.max(np.abs(x_e - x_analytical))
    errors_euler.append(error_euler)

    # Runge-Kutta Method
    error_rk4 = np.max(np.abs(x_r - x_analytical))
    errors_rk4.append(error_rk4)

# Truncation error calculation
truncation_errors_euler = [dt**2 for dt in step_sizes]  
truncation_errors_rk4 = [dt**5 for dt in step_sizes]  

#global error
global_errors_euler = [dt for dt in step_sizes]
global_errors_rk4 = [dt**4 for dt in step_sizes]

# Plotting the global errors
plt.loglog(step_sizes, global_errors_euler, label='Euler Method global Error(O(dt))')
plt.loglog(step_sizes, global_errors_rk4, label='RK4 Method global Error(O(dt^4))')

# Plotting the expected truncation errors
plt.loglog(step_sizes, truncation_errors_euler, label='Euler Truncation error (local) (O(dt^2))')
plt.loglog(step_sizes, truncation_errors_rk4, label='RK4 Truncation error (local) (O(dt^5))')

plt.title('Global Error and Truncation Error for Euler and RK4 method')
plt.xlabel('Step Size (dt)')
plt.ylabel('Error')
plt.grid()
plt.legend()
plt.show()
