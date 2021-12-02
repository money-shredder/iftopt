from typing import Sequence, List, Any

import torch
from torch import Tensor

from .utils import flat_grad, ItemOrSequence


def _hessian(
    dtrain_dmodel: ItemOrSequence[Tensor],
    model_parameters: ItemOrSequence[Tensor]
) -> Tensor:
    grads: List[Tensor] = []
    if isinstance(dtrain_dmodel, Tensor):
        dtdms: Sequence[Tensor] = dtrain_dmodel.split(1)  # type: ignore
    else:
        dtdms = dtrain_dmodel
    for dtdm in dtdms:
        grad = flat_grad(dtdm, model_parameters, retain_graph=True)
        grads.append(grad)
    return torch.stack(grads, dim=0)


def true(
    dval_dmodel: Tensor, dtrain_dmodel: Tensor,
    model_parameters: ItemOrSequence[Tensor], *args: Any, **kwargs: Any
) -> Tensor:
    """Performs true inverse-hessian-vector product.  """
    # FIXME untested
    hess = _hessian(dtrain_dmodel, model_parameters)
    inv = torch.pinverse(hess)
    return dval_dmodel @ inv


def neumann(
    dval_dmodel: Tensor, dtrain_dmodel: Tensor,
    model_parameters: ItemOrSequence[Tensor],
    lr: float, iterations: int, *args: Any, **kwargs: Any
) -> Tensor:
    """Neumann approximation of inverse-hessian-vector product."""
    p, v = (dval_dmodel.clone().detach() for _ in range(2))
    vnorm = v.norm()
    for _ in range(iterations):
        d = lr * flat_grad(
            dtrain_dmodel, model_parameters, grad_outputs=v, retain_graph=True)
        vn = v - d
        vnnorm = vn.norm()
        if vnnorm > vnorm:
            lr /= 2
            print(f'Setting learning rate to {lr} for a contractive Jacobian.')
        else:
            v = vn
            vnorm = vnnorm
        p += v
    return lr * p
