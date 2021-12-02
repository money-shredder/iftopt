import functools
from typing import (
    TypeVar, Union, Any, Mapping, OrderedDict, Iterable, Sequence)

import torch
from torch import Tensor


S = TypeVar('S')
T = TypeVar('T')
ItemOrIterable = Union[T, Iterable[T]]
ItemOrSequence = Union[T, Sequence[T]]
IterableOrMapping = Union[Iterable[T], Mapping[Any, T]]
Parameters = Iterable[torch.nn.Parameter]
StateDict = OrderedDict[str, Tensor]
Tensors = Iterable[Tensor]


def flatten(x: IterableOrMapping[Tensor]) -> Tensor:
    if isinstance(x, Mapping):
        x = x.values()
    return torch.cat([v.view(-1) for v in x])


@functools.wraps(torch.autograd.grad)
def flat_grad(
        outputs: Tensor,
        inputs: ItemOrSequence[Tensor], **kwargs: Any
    ) -> Tensor:
    grads = torch.autograd.grad(outputs, inputs, **kwargs)
    grads = [
        g if g is not None else torch.zeros_like(i)
        for g, i in zip(grads, inputs)]
    return flatten(grads)
