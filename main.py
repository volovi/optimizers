import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

learning_rate = 0.02
beta = 0.95
epsilon = 1e-8
epochs = 100
start_point = (0.25, 0.45)


def cost(x, y):
    return 0.5 * x ** 2 + 10 * y ** 2


def dcost(x, y):
    return np.array((x, 20 * y))


def gradient_descent():
    k, p = np.array(start_point), np.zeros((3, epochs))

    r = p[:, 0] = *k, cost(*k)
    i = 1

    while r[-1] > epsilon and i < epochs:
        k -= learning_rate * dcost(*k)
        r = p[:, i] = *k, cost(*k)
        i += 1

    return 'GD', *p


def momentum_optimizer():
    k, p = np.array(start_point), np.zeros((3, epochs))
    v = np.zeros(2)

    r = p[:, 0] = *k, cost(*k)
    i = 1

    while r[-1] > epsilon and i < epochs:
        v = beta * v + learning_rate * (1 - beta) * dcost(*k)
        k -= v
        r = p[:, i] = *k, cost(*k)
        i += 1

    return 'Momentum', *p


def nesterov_optimizer():
    k, p = np.array(start_point), np.zeros((3, epochs))
    v = np.zeros(2)

    r = p[:, 0] = *k, cost(*k)
    i = 1

    while r[-1] > epsilon and i < epochs:
        v = beta * v + learning_rate * (1 - beta) * dcost(*(k - beta * v))
        k -= v
        r = p[:, i] = *k, cost(*k)
        i += 1

    return 'Nesterov', *p


def rmsprop_optimizer():
    k, p = np.array(start_point), np.zeros((3, epochs)) 
    m, v = np.zeros(2), np.zeros(2)

    r = p[:, 0] = *k, cost(*k)
    i = 1

    while r[-1] > epsilon and i < epochs:
        dk = dcost(*k)
        v = beta * v + (1 - beta) * dk ** 2
        m = beta * m + (1 - beta) * learning_rate / (np.sqrt(v) + epsilon) * dk
        k -= m
        r = p[:, i] = *k, cost(*k)
        i += 1

    return 'RMSProp with Momentum', *p


def adam_optimizer():
    k, p = np.array(start_point), np.zeros((3, epochs))
    m, v = np.zeros(2), np.zeros(2)

    r = p[:, 0] = *k, cost(*k)
    i = 1

    while r[-1] > epsilon and i < epochs:
        dk = dcost(*k)
        m = beta * m + (1 - beta) * dk
        v = beta * v + (1 - beta) * dk ** 2
        m_hat = m / (1 - beta ** i)
        v_hat = v / (1 - beta ** i)
        k -= learning_rate / (np.sqrt(v_hat) + epsilon) * m_hat
        r = p[:, i] = *k, cost(*k)
        i += 1

    return 'Adam', *p


def update(i):
    for x, y, z, l1, l2, l3 in zip(X, Y, Z, lines1, lines2, lines3):
        j, k = slice(i, i + 1), slice(i + 1)
        l1.set_data(range(i + 1), z[k])
        l2.set_data(x[j], y[j])
        l3.set_data_3d(x[j], y[j], z[j])

    return (*lines1, *lines2, *lines3)


fig = plt.figure(figsize=(10, 6), layout="constrained")

ax1 = fig.add_subplot(2, 3, 1)
ax2 = fig.add_subplot(2, 3, 4)
ax3 = fig.add_subplot(2, 3, (2, 6), projection='3d')

X, Y = np.meshgrid(np.linspace(-0.5, 0.5), np.linspace(-0.5, 0.5))
Z = cost(X, Y)

ax2.contour(X, Y, Z, levels=26, linewidths=0.3, cmap='CMRmap')
ax3.plot_surface(X, Y, Z, alpha=0.1, cmap='CMRmap')

labels, X, Y, Z = zip(gradient_descent(), momentum_optimizer(), nesterov_optimizer(), rmsprop_optimizer(), adam_optimizer())

lines1, lines2, lines3 = zip(*[(ax1.plot(z, label=l)[0], ax2.plot(x, y, 'o', label=l)[0], ax3.plot(x, y, z, 'o', label=l)[0]) 
    for l, x, y, z in zip(labels, X, Y, Z)])

ax1.legend()

ani = animation.FuncAnimation(fig=fig, func=update, frames=epochs, interval=50, blit=True)
plt.show()
