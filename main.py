import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

learning_rate = 0.02
beta = 0.95
epsilon = 1e-8
num = 100
start_point = (0.25, 0.45)


def cost(x, y):
    return 0.5 * x ** 2 + 10 * y ** 2


def dcost(x, y):
    return np.array((x, 20 * y))


def gradient_descent():
    p, w = create_path('GD'), np.array(start_point)

    while p(w):
        w -= learning_rate * dcost(*w)

    return p.get_data()


def momentum_optimizer():
    p, w = create_path('Momentum'), np.array(start_point)
    m = np.zeros(2)

    while p(w):
        m = beta * m + (1 - beta) * learning_rate * dcost(*w)
        w -= m

    return p.get_data()


def nesterov_optimizer():
    p, w = create_path('Nesterov'), np.array(start_point)
    m = np.zeros(2)

    while p(w):
        m = beta * m + (1 - beta) * learning_rate * dcost(*(w - beta * m))
        w -= m

    return p.get_data()


def rmsprop_optimizer():
    p, w = create_path('RMSProp with Momentum'), np.array(start_point)
    m, v = np.zeros(2), np.zeros(2)

    while p(w):
        dw = dcost(*w)
        v = beta * v + (1 - beta) * dw ** 2
        m = beta * m + (1 - beta) * learning_rate / (np.sqrt(v) + epsilon) * dw
        w -= m

    return p.get_data()


def adam_optimizer():
    p, w = create_path('Adam'), np.array(start_point)
    m, v = np.zeros(2), np.zeros(2)

    while p(w):
        dw = dcost(*w)
        m = beta * m + (1 - beta) * dw
        v = beta * v + (1 - beta) * dw ** 2
        m_hat = m / (1 - beta ** p.get_i())
        v_hat = v / (1 - beta ** p.get_i())
        w -= learning_rate / (np.sqrt(v_hat) + epsilon) * m_hat

    return p.get_data()


def create_path(label):
    p = np.zeros((3, num))
    i = 0

    def inner(k):
        nonlocal p, i
        z = cost(*k)
        r = z > epsilon and i < num
        if r:
            p[:, i] = *k, z
            i += 1
        return r

    inner.get_data = lambda: (label, *p)
    inner.get_i = lambda: i

    return inner


def update(i):
    j, k = slice(i, i + 1), slice(i + 1)
    for x, y, z, l1, l2, l3 in zip(X, Y, Z, lines1, lines2, lines3):
        l1.set_data(range(i + 1), z[k])
        l2.set_data(x[j], y[j])
        l3.set_data_3d(x[j], y[j], z[j])

    return (*lines1, *lines2, *lines3)


fig = plt.figure(figsize=(10, 6), layout='constrained')

ax1 = fig.add_subplot(2, 3, 1)
ax2 = fig.add_subplot(2, 3, 4)
ax3 = fig.add_subplot(2, 3, (2, 6), projection='3d')

X, Y = np.meshgrid(np.linspace(-0.5, 0.5), np.linspace(-0.5, 0.5))
Z = cost(X, Y)

ax2.contour(X, Y, Z, levels=26, linewidths=0.3, cmap='CMRmap')
ax3.plot_surface(X, Y, Z, alpha=0.1, cmap='CMRmap')

labels, X, Y, Z = zip(gradient_descent(), momentum_optimizer(), nesterov_optimizer(), rmsprop_optimizer(), adam_optimizer())

lines1, lines2, lines3 = zip(*[(*ax1.plot(z), *ax2.plot(x, y, 'o'), *ax3.plot(x, y, z, 'o')) for x, y, z in zip(X, Y, Z)])

ax1.legend(labels)

ani = animation.FuncAnimation(fig=fig, func=update, frames=num, interval=50, blit=True)
plt.show()
