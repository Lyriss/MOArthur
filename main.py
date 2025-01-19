import numpy as np
import matplotlib.pyplot as plt


def EVopt(f, dim, n_particles, max_iter, bounds):
    """
    Evolutionary Optimization Function.

    Parameters:
        f (function): Objective function to minimize.
        dim (int): Dimension of the search space.
        n_particles (int): Number of particles.
        max_iter (int): Maximum number of iterations.
        bounds (tuple): Tuple containing the lower and upper bounds of the search space.

    Returns:
        tuple: xmin (best position), fmin (minimum value), neval (function evaluations), coords (trajectory).
    """
    var_min = np.full(dim, bounds[0])
    var_max = np.full(dim, bounds[1])

    # Initialization
    particles = np.random.uniform(var_min, var_max, (n_particles, dim))
    nels = np.array([f(p) for p in particles])
    neval = n_particles  # Function evaluations

    best_idx = np.argmin(nels)
    best_nel = nels[best_idx]
    worst_nel = nels.max()
    xmin = particles[best_idx].copy()

    coords = np.zeros((max_iter, dim))
    coords[0] = xmin

    # Main Loop
    for iteration in range(max_iter):
        xnew, xnew1, xnew2 = [], [], []

        for i in range(n_particles):
            enrichment_bound = nels.mean()
            neli = nels[i]

            if neli > enrichment_bound:
                stability_bound = np.random.rand()
                stability_level = (neli - best_nel) / (worst_nel - best_nel)

                if stability_level > stability_bound:
                    alpha_index1 = np.random.randint(dim)
                    alpha_index2 = np.random.choice(dim, alpha_index1, replace=False)

                    xnew1.append(particles[i].copy())
                    xnew1[-1][alpha_index2] = xmin[alpha_index2]

                    gamma_index1 = np.random.randint(dim)
                    gamma_index2 = np.random.choice(dim, gamma_index1, replace=False)
                    dist = np.linalg.norm(particles - particles[i], axis=1)

                    nearest_idx = np.argsort(dist)[1] if len(dist) > 1 else None
                    if nearest_idx is not None:
                        xng = particles[nearest_idx].copy()
                        xnew2.append(particles[i].copy())
                        xnew2[-1][gamma_index2] = xng[gamma_index2]
                else:
                    xcp = particles.mean(axis=0)
                    xnew1.append(
                        particles[i] + (np.random.rand() * xmin - np.random.rand() * xcp) / stability_level
                    )

                    dist = np.linalg.norm(particles - particles[i], axis=1)
                    nearest_idx = np.argsort(dist)[1] if len(dist) > 1 else None
                    if nearest_idx is not None:
                        xng = particles[nearest_idx].copy()
                        xnew2.append(particles[i] + (np.random.rand() * xmin - np.random.rand() * xng))

            else:
                xnew.append(particles[i] + np.random.rand())

        # Clip to bounds and concatenate
        xnew = np.clip(np.array(xnew), var_min, var_max) if xnew else np.empty((0, dim))
        xnew1 = np.clip(np.array(xnew1), var_min, var_max) if xnew1 else np.empty((0, dim))
        xnew2 = np.clip(np.array(xnew2), var_min, var_max) if xnew2 else np.empty((0, dim))

        particles = np.vstack([particles, xnew, xnew1, xnew2])
        nels = np.array([f(p) for p in particles])
        neval += len(particles)

        # Select the best particles
        order = np.argsort(nels)
        particles = particles[order][:n_particles]
        nels = nels[order][:n_particles]

        best_nel = nels.min()
        worst_nel = nels.max()
        xmin = particles[np.argmin(nels)].copy()

        coords[iteration] = xmin

    fmin = best_nel
    return xmin, fmin, neval, coords


def testSuit(function_name):
    functions = {
        # Many local minima
        "Ackley": (lambda x: -20 * np.exp(-0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))) -
                             np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))) + np.e + 20,
                   2, [-5, 5], [0, 0], 0),
        "Eggholder": (lambda x: -(x[1] + 47) * np.sin(np.sqrt(abs(x[1] + x[0] / 2 + 47))) - x[0] * np.sin(
            np.sqrt(abs(x[0] - (x[1] + 47)))),
                      2, [-512, 512], [512, 404.2319], -959.6407),
        # Bowl-shaped
        "Sphere": (lambda x: sum(xi ** 2 for xi in x), 2, [-5.12, 5.12], [0, 0], 0),
        "Trid": (lambda x: sum((xi - 1) ** 2 for xi in x) - sum(x[i] * x[i - 1] for i in range(1, len(x))),
                 2, [-4, 4], [2, 2], -2),
        # Plate-shaped
        "McCormick": (lambda x: np.sin(x[0] + x[1]) + (x[0] - x[1]) ** 2 - 1.5 * x[0] + 2.5 * x[1] + 1,
                      2, [-1.5, 4], [-0.54719, -1.54719], -1.9133),
        "Booth": (lambda x: (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2,
                  2, [-10, 10], [1, 3], 0),
        # Valley-shaped
        "Rosenbrock": (lambda x: sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2 for i in range(len(x) - 1)),
                       2, [-2.048, 2.048], [1, 1], 0),
        "Six-Hump Camel": (
        lambda x: (4 - 2.1 * x[0] ** 2 + (x[0] ** 4) / 3) * x[0] ** 2 + x[0] * x[1] + (-4 + 4 * x[1] ** 2) * x[1] ** 2,
        2, [-3, 3], [0.0898, -0.7126], -1.0316),
        # Steep ridges/drops
        "Michalewicz": (
        lambda x: -sum(np.sin(x[i]) * np.sin(((i + 1) * x[i] ** 2) / np.pi) ** 20 for i in range(len(x))),
        2, [0, np.pi], [2.20, 1.57], -1.8013),
        "Easom": (lambda x: -np.cos(x[0]) * np.cos(x[1]) * np.exp(-((x[0] - np.pi) ** 2 + (x[1] - np.pi) ** 2)),
                  2, [-100, 100], [np.pi, np.pi], -1),
        "Himmelblau": (lambda x: (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2, 2, [-4, 4], [3, 2], 0)
    }
    return functions[function_name]


def plot_trajectory(f, bounds, coords):
    x = np.linspace(bounds[0], bounds[1], 500)
    y = np.linspace(bounds[0], bounds[1], 500)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[f([xi, yi]) for xi, yi in zip(x_row, y_row)] for x_row, y_row in zip(X, Y)])

    plt.figure()
    plt.contour(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar()
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.title('Optimization Trajectory')

    plt.plot(coords[:, 0], coords[:, 1], 'r-o', label='Trajectory')
    plt.plot(coords[-1, 0], coords[-1, 1], 'g*', label='Final Point', markersize=10)
    plt.legend()
    plt.show()


def main():
    function_name = "Himmelblau"  # Change to desired function
    f, dim, bounds, xmin_true, fmin_true = testSuit(function_name)

    n_particles = 30
    max_iter = 100

    xmin, fmin, neval, coords = EVopt(f, dim, n_particles, max_iter, bounds)
    plot_trajectory(f, bounds, coords)

    print(f"Test function: {function_name}")
    print(f"True minimum: x = {xmin_true}, f(x) = {fmin_true}")
    print(f"Obtained minimum: x = {xmin}, f(x) = {fmin}")
    print(f"Error in x: {np.linalg.norm(np.array(xmin) - np.array(xmin_true))}")
    print(f"Error in f(x): {abs(fmin - fmin_true)}")
    print(f"Number of Func Evaluations: {neval}")


if __name__ == "__main__":
    main()