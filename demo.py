import os
from typing import cast, Any, Callable, List, Tuple

import torch
import numpy as np
from matplotlib import pyplot
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation

from iftopt import HyperOptimizer


Tensor = torch.Tensor
Model = Callable[[Tensor, Tensor], Tensor]
History = List[Tuple[float, float]]


def val_model(x: Tensor, y: Tensor) -> Tensor:
    return x ** 2 # + y ** 2


def train_model(x: Tensor, y: Tensor) -> Tensor:
    return (x + y) ** 2


def log(hist: History, x: Tensor, y: Tensor) -> None:
    p = (float(x), float(y))
    hist.append(p)


def optimize(train_model: Model, val_model: Model) -> History:
    hist: History = []
    x = torch.nn.Parameter(torch.Tensor([1.0]))  # inner parameter
    y = torch.nn.Parameter(torch.Tensor([1.0]))  # hyper-parameter
    opt = torch.optim.SGD([x], lr=0.1)
    hopt = HyperOptimizer(
        [y], torch.optim.SGD([y], lr=0.1), vih_lr=0.1, vih_iterations=5)
    log(hist, x, y)
    for _ in range(25):
        for _ in range(5):
            z = train_model(x, y)
            opt.zero_grad()
            z.backward()
            opt.step()
            log(hist, x, y)
        hopt.set_train_parameters([x])
        z = train_model(x, y)
        hopt.train_step(z)
        v = val_model(x, y)
        hopt.val_step(v)
        print(
            f'train_param={float(x):.3g}, val_param={float(y):.3g}, '
            f'train_loss={float(z):.3g}, val_loss={float(v):.3g}.')
        hopt.grad()
        hopt.step()
        log(hist, x, y)
    return hist


def aniplot(hist: History) -> FuncAnimation:
    xhist, yhist = list(zip(*hist))
    scale = 1.1
    fig, ax = pyplot.subplots()
    ax.axis(np.array([-1, 1, -1, 1]) * scale)
    ax.set_aspect('equal')
    line, = ax.plot([], [], '-')
    point, = ax.plot([], [], marker='o')
    def anifunc(frame: int) -> Tuple[Line2D, Line2D]:
        point.set_data([xhist[frame]], [yhist[frame]])
        line.set_data(xhist[:frame + 1], yhist[:frame + 1])
        return point, line
    return FuncAnimation(fig, cast(Any, anifunc), len(hist), blit=True)


def main():
    hist = optimize(train_model, val_model)
    ani = aniplot(hist)
    os.makedirs('assets', exist_ok=True)
    ani.save('assets/demo.mp4')


if __name__ == '__main__':
    main()
