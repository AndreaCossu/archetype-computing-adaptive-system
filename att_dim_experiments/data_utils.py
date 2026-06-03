import numpy as np

def get_mg17(train_len, val_len, test_len,
             tau=17, beta=0.2, gamma=0.1, n=10, forecasting_delay=1):


    """Generate Mackey–Glass τ=17 time series (Euler discretization)."""

    total_len = train_len + val_len + test_len
    history_length = tau
    dt = 1.0

    # Initialize history with constant value
    x = np.zeros((total_len + history_length, 1))
    x[:history_length] = 1.2

    for t in range(history_length, total_len + history_length - 1):
        x_tau = x[t - tau]
        dx = beta * x_tau / (1 + x_tau ** n) - gamma * x[t]
        x[t + 1] = x[t] + dx * dt

    data = x[history_length:]

    train_data = data[:train_len]
    val_data = data[train_len:train_len + val_len]
    test_data = data[train_len + val_len:-1]
    train_labels = data[forecasting_delay:train_len + forecasting_delay]
    val_labels = data[train_len + forecasting_delay:train_len + val_len + forecasting_delay]
    test_labels = data[train_len + val_len + forecasting_delay:]


    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)


def get_lorenz(train_len, val_len, test_len, dt=0.01, forecasting_delay=1):

    """Generate Lorenz system using RK4 integration."""

    def lorenz(x, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
        dx = sigma * (x[1] - x[0])
        dy = x[0] * (rho - x[2]) - x[1]
        dz = x[0] * x[1] - beta * x[2]
        return np.array([dx, dy, dz])

    total_len = train_len + val_len + test_len
    x = np.zeros((total_len, 3))
    x[0] = np.array([1.0, 1.0, 1.0])

    for t in range(total_len - 1):
        k1 = lorenz(x[t])
        k2 = lorenz(x[t] + 0.5 * dt * k1)
        k3 = lorenz(x[t] + 0.5 * dt * k2)
        k4 = lorenz(x[t] + dt * k3)
        x[t + 1] = x[t] + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    train_data = x[:train_len]
    val_data = x[train_len:train_len + val_len]
    test_data = x[train_len + val_len:-1]

    train_labels = x[forecasting_delay:train_len + forecasting_delay]
    val_labels = x[train_len + forecasting_delay:train_len + val_len + forecasting_delay]
    test_labels = x[train_len + val_len + forecasting_delay:]

    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)


def get_narma(train_len, val_len, test_len, c1=0.3, c2=0.05, c3=1.5, c4=0.1, order=30):
    
    """Generate NARMA time series dataset.
    Args:
        train_len (int): Length of the training set.
        val_len (int): Length of the validation set.
        test_len (int): Length of the test set.
    """
    seed = 0
    np.random.seed(seed)
    total_len = train_len + val_len + test_len
    u = np.random.uniform(0, 0.5, (total_len, 1))
    y = np.zeros((total_len,1), dtype=np.float32)

    for t in range(order, total_len):
        y[t] = (
            c1 * y[t - 1]
            + c2 * y[t - 1] * np.sum(y[t - order : t])
            + c3 * u[t - 1] * u[t - order]
            + c4
        )

    train_data = y[:train_len]
    val_data = y[train_len:train_len + val_len]
    test_data = y[train_len + val_len:]

    return (u[:train_len], train_data), (u[train_len:train_len + val_len], val_data), (u[train_len + val_len:], test_data)



# ---- assume get_mg17, get_lorenz, get_narma10 are already defined ----

def main():
    import matplotlib.pyplot as plt
    train_len = 2000
    val_len = 0
    test_len = 0

    # Generate data
    mg, _, _ = get_mg17(train_len, val_len, test_len)
    narma, _, _ = get_narma(train_len, val_len, test_len)
    lorenz, _, _ = get_lorenz(train_len, val_len, test_len)

    t = np.arange(train_len)

    # ---- Mackey–Glass ----
    plt.figure()
    plt.plot(t, mg[0], mg[1])
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.title("Mackey–Glass τ=17")
    plt.show()

    # ---- NARMA10 ----
    plt.figure()
    plt.plot(t, narma[0])
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.title("NARMA10")
    plt.show()

    # ---- Lorenz phase plot ----
    plt.figure()
    plt.plot(lorenz[0][:, 0], lorenz[0][:, 1])
    plt.xlabel("x(t)")
    plt.ylabel("y(t)")
    plt.title("Lorenz Attractor (x–y projection)")
    plt.show()


if __name__ == "__main__":
    main()
