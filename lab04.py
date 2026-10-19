import numpy as np
import matplotlib.pyplot as plt

# RK4 para un paso
def rk4_step(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h * k1 / 2)
    k3 = f(t + h / 2, y + h * k2 / 2)
    k4 = f(t + h, y + h * k3)
    return y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def simular(f, y0, t0, tf, h):
    n = int((tf - t0) / h)
    t = np.linspace(t0, tf, n + 1)
    y = np.zeros((n + 1, len(y0)))
    y[0] = y0
    for i in range(n):
        y[i + 1] = rk4_step(f, t[i], y[i], h)
    return t, y


# ------------------ Ejercicio 1 ------------------
L = 1.0
g = 9.81
w0 = np.sqrt(g / L)


def f_pendulo(t, u):
    theta, omega = u
    return np.array([omega, -w0**2 * np.sin(theta)])


# calculo manual de k1 y k2 para h=0.1
h_manual = 0.1
u0 = np.array([np.pi / 3, 0.0])
k1 = f_pendulo(0, u0)
k2 = f_pendulo(h_manual / 2, u0 + h_manual * k1 / 2)
print("Pendulo: k1 =", k1)
print("Pendulo: k2 =", k2)

t1, y1 = simular(f_pendulo, np.array([0.1, 0.0]), 0, 15, 0.01)
t2, y2 = simular(f_pendulo, np.array([2.0, 0.0]), 0, 15, 0.01)

plt.figure()
plt.plot(t1, y1[:, 0], label='theta0=0.1')
plt.plot(t2, y2[:, 0], label='theta0=2.0')
plt.xlabel('t [s]')
plt.ylabel('theta [rad]')
plt.legend()
plt.grid(True)
plt.savefig('fig_pendulo_theta.png', dpi=150)

plt.figure()
plt.plot(y1[:, 0], y1[:, 1], label='theta0=0.1')
plt.plot(y2[:, 0], y2[:, 1], label='theta0=2.0')
plt.xlabel('theta [rad]')
plt.ylabel('omega [rad/s]')
plt.legend()
plt.grid(True)
plt.savefig('fig_pendulo_fases.png', dpi=150)


def periodo_aprox(t, theta):
    cruces = []
    for i in range(1, len(theta) - 1):
        if theta[i - 1] < 0 and theta[i] >= 0:
            cruces.append(t[i])
    if len(cruces) >= 2:
        return np.mean(np.diff(cruces))
    return None


T0 = 2 * np.pi * np.sqrt(L / g)
print("T0 =", T0)
print("T(0.1) =", periodo_aprox(t1, y1[:, 0]))
print("T(2.0) =", periodo_aprox(t2, y2[:, 0]))


# ------------------ Ejercicio 2 ------------------
gamma = 0.2
k = 4.0
beta = 1.0


def f_duffing(t, u, beta_val=1.0):
    x, v = u
    return np.array([v, -gamma * v - k * x - beta_val * x**3])


t_d, y_d1 = simular(lambda t, u: f_duffing(t, u, 1.0), np.array([2.0, 0.0]), 0, 20, 0.01)
t_d, y_d0 = simular(lambda t, u: f_duffing(t, u, 0.0), np.array([2.0, 0.0]), 0, 20, 0.01)

plt.figure()
plt.plot(y_d1[:, 0], y_d1[:, 1], label='beta=1.0')
plt.plot(y_d0[:, 0], y_d0[:, 1], label='beta=0.0')
plt.xlabel('x [m]')
plt.ylabel('v [m/s]')
plt.legend()
plt.grid(True)
plt.savefig('fig_duffing_fases.png', dpi=150)


# ------------------ Ejercicio 3 ------------------
gamma_f = 0.2
w0_f2 = 9.81
f0 = 0.5
wd = 0.667


def f_forzado(t, u):
    theta, omega = u
    return np.array([
        omega,
        -gamma_f * omega - w0_f2 * np.sin(theta) + f0 * np.cos(wd * t)
    ])


t_f, y_f = simular(f_forzado, np.array([0.0, 0.0]), 0, 60, 0.01)

plt.figure()
plt.plot(t_f, y_f[:, 0])
plt.xlabel('t [s]')
plt.ylabel('theta [rad]')
plt.grid(True)
plt.savefig('fig_forzado_theta.png', dpi=150)

plt.figure()
plt.plot(y_f[:, 0], y_f[:, 1])
plt.xlabel('theta [rad]')
plt.ylabel('omega [rad/s]')
plt.grid(True)
plt.savefig('fig_forzado_fases.png', dpi=150)

E_forzado = 0.5 * y_f[:, 1]**2 + w0_f2 * (1 - np.cos(y_f[:, 0]))
plt.figure()
plt.plot(t_f, E_forzado)
plt.xlabel('t [s]')
plt.ylabel('E [J/kg]')
plt.grid(True)
plt.savefig('fig_forzado_energia.png', dpi=150)


# ------------------ Ejercicio 4 ------------------
G = 4 * np.pi**2
m = 1.0
eps = 0.01

r1 = np.array([0.0, 0.0])
r2 = np.array([1.0, 0.0])
r3 = np.array([0.5, np.sqrt(3) / 2])

v1 = np.array([np.pi, -np.sqrt(3) * np.pi])
v2 = np.array([np.pi, np.sqrt(3) * np.pi])
v3 = np.array([-2 * np.pi, 0.0])

u0_nb = np.concatenate([r1, r2, r3, v1, v2, v3])


def f_nbody(t, u):
    pos = u[:6].reshape(3, 2)
    vel = u[6:].reshape(3, 2)
    acc = np.zeros((3, 2))
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            dr = pos[j] - pos[i]
            dist2 = np.dot(dr, dr) + eps**2
            acc[i] += G * m * dr / dist2**1.5
    return np.concatenate([vel.flatten(), acc.flatten()])


def energia_nbody(u):
    pos = u[:6].reshape(3, 2)
    vel = u[6:].reshape(3, 2)
    K = 0.5 * m * np.sum(vel**2)
    U = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            dr = pos[j] - pos[i]
            dist = np.sqrt(np.dot(dr, dr) + eps**2)
            U -= G * m * m / dist
    return K + U


t_nb, y_nb = simular(f_nbody, u0_nb, 0, 2.0, 0.01)
pos = y_nb[:, :6].reshape(-1, 3, 2)

plt.figure()
for i in range(3):
    plt.plot(pos[:, i, 0], pos[:, i, 1], label=f'masa {i + 1}')
plt.xlabel('x [UA]')
plt.ylabel('y [UA]')
plt.legend()
plt.grid(True)
plt.savefig('fig_nbody_trayectorias.png', dpi=150)

pasos = [0.01, 0.005, 0.0025, 0.00125]
errores = []
for h in pasos:
    t_nb, y_nb = simular(f_nbody, u0_nb, 0, 2.0, h)
    E = np.array([energia_nbody(u) for u in y_nb])
    err = np.max(np.abs((E - E[0]) / E[0]))
    errores.append(err)
    print(f"h={h:.5f}  dEmax={err:.3e}")

print("Orden p:")
for i in range(len(errores) - 1):
    p = np.log2(errores[i] / errores[i + 1])
    print(f"p entre h={pasos[i]} y h={pasos[i + 1]}: {p:.3f}")

plt.show()