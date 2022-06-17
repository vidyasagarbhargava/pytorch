import torch
import torch.distributed as dist


class AllReduceState(object):
    r"""
    Stores parameters, needed to perform ``all_reduce`` algorithm
    within a communication hook.
    """

    __slots__ = [
        "process_group",
        "world_size",
        "gradient_predivide_factor",
        "gradient_postdivide_factor"
    ]

    def __init__(
        self,
        process_group,
        world_size
    ):
        self.process_group = process_group
        self.world_size = world_size
        self.gradient_predivide_factor = self._get_gradient_predivide_factor(
            self.world_size
        )
        self.gradient_postdivide_factor = self.world_size / self.gradient_predivide_factor

    # Same as in `FullyShardedDataParallel` class.
    # Required to perform gradient pre-and post division
    def _get_gradient_predivide_factor(self, world_size: int) -> float:
        factor: int = 1
        while world_size % factor == 0 and world_size / factor > factor:
            factor *= 2
        return float(factor)


def allreduce_hook(state: AllReduceState, grad: torch.Tensor):
    r"""
    This FSDP communication hook implements ``all_reduce`` algorithm
    and neccessary pre-and post devision of gradients.
    """
    if state.gradient_predivide_factor > 1:
        grad.div_(state.gradient_predivide_factor)
    dist.all_reduce(grad, group=state.process_group)
    if state.gradient_postdivide_factor > 1:
        # Average grad by world_size for consistency with PyTorch DDP.
        grad.div_(state.gradient_postdivide_factor)
