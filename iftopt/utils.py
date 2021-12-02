import io
import functools
from typing import (
    TypeVar, Generic, Union, Optional, Any, Mapping, OrderedDict,
    Iterable, Sequence, Tuple, Generator)

import torch
from torch import Tensor
import tqdm as tqdm_old  # type: ignore


S = TypeVar('S')
T = TypeVar('T')
ItemOrIterable = Union[T, Iterable[T]]
ItemOrSequence = Union[T, Sequence[T]]
IterableOrMapping = Union[Iterable[T], Mapping[Any, T]]
Parameters = Iterable[torch.nn.Parameter]
StateDict = OrderedDict[str, Tensor]
Tensors = Iterable[Tensor]


class tqdm(tqdm_old.tqdm, Generic[T]):
    n: int

    # pylint: disable=useless-super-delegation
    def __init__(
        self,
        iterable: Optional[Iterable[T]] = None, desc: Optional[str] = None,
        total: Optional[int] = None, leave: Optional[bool] = True,
        file: Union[None, io.TextIOWrapper, io.StringIO] = None,
        ncols: Optional[int] = None,
        mininterval: float = 0.1, maxinterval: float = 10.0,
        miniters: Optional[int] = None,
        ascii: Union[None, bool, str] = None,
        disable: bool = False, unit: str = 'it', unit_scale: bool = False,
        dynamic_ncols: bool = False, smoothing: float = 0.3,
        bar_format: Optional[str] = None, initial: float = 0,
        position: Optional[int] = None,
        postfix: Optional[Mapping[str, Any]] = None,
        unit_divisor: int = 1000, write_bytes: Optional[bool] = None,
        lock_args: Optional[Tuple[Any, ...]] = None,
        nrows: Optional[int] = None, colour: Optional[str] = None,
        gui: bool = False, **kwargs: Mapping[str, Any]
    ) -> None:
        super().__init__(  # type: ignore
            iterable, desc, total, leave, file, ncols,
            mininterval, maxinterval, miniters, ascii, disable,
            unit, unit_scale, dynamic_ncols, smoothing, bar_format,
            initial, position, postfix, unit_divisor, write_bytes,
            lock_args, nrows, colour, gui, **kwargs)

    def set_description(
        self, desc: Optional[str] = None, refresh: bool = True
    ) -> None:
        return super().set_description(desc=desc, refresh=refresh)  # type: ignore

    def __iter__(self) -> Generator[T, None, None]:
        return super().__iter__()  # type: ignore


def trange(start: int, *args: Optional[int], **kwargs: Any) -> tqdm[int]:
    if not args:
        return tqdm(range(start), **kwargs)
    try:
        stop = args[0]
    except IndexError:
        stop = None
    try:
        step = args[1] if args[1] is not None else 1
    except IndexError:
        step = 1
    if stop is not None:
        return tqdm(range(start, stop, step), **kwargs)
    def inf_loop(start: int, step: int) -> Generator[int, None, None]:
        i = start
        while True:
            yield start
            i += step
    return tqdm(inf_loop(start, step), **kwargs)  # type: ignore


def flatten(x: IterableOrMapping[Tensor]) -> Tensor:
    if isinstance(x, Mapping):
        x = x.values()
    return torch.cat([v.view(-1) for v in x])


def flat_numel(x: IterableOrMapping[Tensor]) -> int:
    if isinstance(x, Mapping):
        x = x.values()
    return sum(v.numel() for v in x)


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
