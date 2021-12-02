# reference:
#   * https://github.com/lorraine2/implicit-hyper-opt
from typing import Optional, Literal, Tuple

import torch
from torch import Tensor

from . import vih
from .utils import flat_grad, Parameters, StateDict, ItemOrSequence


VihMethod = Literal['true', 'neumann']


class HyperOptimizer:
    hyper_parameters: Tuple[torch.nn.Parameter, ...]
    vih_method: VihMethod
    vih_lr: float
    vih_iterations: int
    dtrain_dparam: Optional[Tensor]
    dval_dparam: Optional[Tensor]
    dval_dhyper: Optional[Tensor]

    def __init__(
        self, hyper_parameters: Parameters, optimizer: torch.optim.Optimizer,
        vih_method: VihMethod = 'neumann',
        vih_lr: float = 0.1, vih_iterations: int = 5
    ):
        super().__init__()
        self.hyper_parameters = tuple(hyper_parameters)
        self._hyper_optimizer = optimizer
        self.vih_method = vih_method
        self.vih_lr = vih_lr
        self.vih_iterations = vih_iterations

    def load_state_dict(self, state_dict: StateDict) -> None:
        self._hyper_optimizer.load_state_dict(state_dict)

    def state_dict(self) -> StateDict:
        return self._hyper_optimizer.state_dict()  # type: ignore

    def set_train_parameters(self, train_parameters: ItemOrSequence[Tensor]):
        self.train_parameters = train_parameters
        self.zero_train_grad()
        self.zero_hyper_grad()

    def zero_train_grad(self):
        if isinstance(self.train_parameters, Tensor):
            self.train_parameters.grad = None
        else:
            for p in self.train_parameters:
                p.grad = None
        self.dtrain_dparam = None

    def zero_hyper_grad(self):
        self.dval_dparam = None
        self.dval_dhyper = None
        self._hyper_optimizer.zero_grad()

    def _set_hyper_grad(self, grad: Tensor):
        i = 0
        for p in self.hyper_parameters:
            j = i + p.numel()
            pg = grad[i:j].view(p.shape)
            p.grad = pg if p.grad is None else p.grad + pg
            i = j
        if i != grad.numel():
            raise ValueError(
                f'The size of grad ({grad.numel()}) must match '
                f'the total number ({i}) of all hyperparameters.')

    def train_step(self, train_loss: Tensor) -> Tensor:
        dtrain_dparam = flat_grad(
            train_loss, self.train_parameters,
            retain_graph=True, create_graph=True)
        if self.dtrain_dparam is None:
            self.dtrain_dparam = dtrain_dparam
        else:
            self.dtrain_dparam += dtrain_dparam
        return dtrain_dparam

    def val_step(
        self, val_loss: Tensor, retain_graph: bool = False,
        use_dval_dhyper: bool = True
    ) -> Tuple[Tensor, Optional[Tensor]]:
        dval_dparam = flat_grad(
            val_loss, self.train_parameters,
            retain_graph=retain_graph or use_dval_dhyper)
        if self.dval_dparam is None:
            self.dval_dparam = dval_dparam
        else:
            self.dval_dparam += dval_dparam
        dval_dhyper = None
        if use_dval_dhyper:
            dval_dhyper = flat_grad(
                val_loss, self.hyper_parameters, allow_unused=True)
            if self.dval_dhyper is None:
                self.dval_dhyper = dval_dhyper
            else:
                self.dval_dhyper += dval_dhyper
        return dval_dparam, dval_dhyper

    def _vec_inverse_hessian_product(
            self, dval_dparam: Tensor, dtrain_dparam: Tensor,
            train_parameters: ItemOrSequence[Tensor]
    ) -> Tensor:
        func = getattr(vih, self.vih_method)
        return func(
            dval_dparam, dtrain_dparam, train_parameters,
            lr=self.vih_lr, iterations=self.vih_iterations)

    def grad(self) -> Tuple[Tensor, Optional[Tensor], Tensor]:
        if self.dval_dparam is None or self.dtrain_dparam is None:
            raise ValueError(
                'Call .train_step() and .val_step() before this method.')
        # vihp = dLv/dw * [d2Lt/dw2]^-1
        vihp = self._vec_inverse_hessian_product(
            self.dval_dparam, self.dtrain_dparam, self.train_parameters)
        # -vihp * d2Lt/dwdh
        hyper_grad = -flat_grad(
            self.dtrain_dparam, self.hyper_parameters, grad_outputs=vihp,
            allow_unused=True)
        if self.dval_dhyper is not None:
            # dLv/dh
            hyper_grad += self.dval_dhyper
        self._set_hyper_grad(hyper_grad)
        return vihp, self.dval_dhyper, hyper_grad

    def step(self) -> None:
        self._hyper_optimizer.step()
