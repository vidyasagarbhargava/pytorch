# Owner(s): ["oncall: distributed"]

import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import distributed as dist
from torch.distributed.algorithms.fsdp_comm_hooks import allreduce_hook
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
from torch.distributed.fsdp.wrap import (
    enable_wrap,
    wrap,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

class TestCommunicationHooks(FSDPTest):

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "has_wrapping", [True, False])
    @parametrize(
        "sharding_strategy",
        [
            ShardingStrategy.NO_SHARD,
            ShardingStrategy.FULL_SHARD,
            ShardingStrategy.SHARD_GRAD_OP
        ])
    def test_communication_hook_allreduce(
        self,
        has_wrapping: bool,
        sharding_strategy: Optional[ShardingStrategy]
    ):

        """
        Tests FSDP's all_reduce communication hook parity with default all_reduce.

        Arguments:

            has_wrapping (bool): Configures wrapping of a module.

            sharding_strategy (Optional[ShardingStrategy]): Configures the FSDP algorithm.
        """

        class Net(nn.Module):

            def __init__(self, has_wrapping):
                # to ensure determinizm
                torch.manual_seed(0)
                torch.cuda.manual_seed(0)
                super().__init__()
                if has_wrapping:
                    with enable_wrap(wrapper_cls=FSDP, device_id=torch.cuda.current_device()):
                        self.fc1 = wrap(nn.Linear(8, 16))
                else:
                    self.fc1 = nn.Linear(8, 16)
                self.relu = F.relu
                self.fc2 = nn.Linear(16, 4)

            def forward(self, x):
                return self.fc2(self.relu(self.fc1(x)))

        # Initialize the model and inputs
        device = torch.device("cuda")
        fsdp_model_with_hook = FSDP(
            Net(has_wrapping),
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy
        ).to(device)
        fsdp_model_no_hook = FSDP(
            Net(has_wrapping),
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy
        ).to(device)
        optimizer1 = optim.SGD(fsdp_model_with_hook.parameters(), lr=0.01)
        optimizer2 = optim.SGD(fsdp_model_no_hook.parameters(), lr=0.01)

        input = torch.randn(7, 8, device=device)
        target = torch.randn(7, 4, device=device)

        # At this point we didn't register any communication hooks.
        for entry in FSDP.fsdp_modules(fsdp_model_with_hook):
            self.assertFalse(hasattr(entry, "communication_hook"))
            self.assertFalse(hasattr(entry, "communication_hook_state"))

        # Initialize allreduce hook state
        allreduce_state = allreduce_hook.AllReduceState(
            process_group=None,
            world_size=dist.get_world_size()
        )

        # FSDP currently suports communication hooks for a NO_SHARD strategy
        # Check that a Runtime Error is raised for other strategies
        if sharding_strategy != ShardingStrategy.NO_SHARD:

            with self.assertRaises(RuntimeError) as captured:
                fsdp_model_with_hook.register_comm_hook(allreduce_state, allreduce_hook.allreduce_hook)

            # Check that the logger has an expected entry
            self.assertEqual(
                str(captured.exception),
                "Communication hooks are currently only available for a NO_SHARD strategy."
            )
        else:

            with self.assertLogs() as captured:
                fsdp_model_with_hook.register_comm_hook(allreduce_state, allreduce_hook.allreduce_hook)

            # Check that the logger has only one entry
            self.assertEqual(len(captured.records), 1)

            # Check that the logger has an expected entry
            self.assertEqual(
                captured.records[0].getMessage(),
                f"NOTE: {allreduce_hook.allreduce_hook.__qualname__} will be shared across all submodules."
            )

            # Check that the hook was registered for the root and all submodules if any
            # At this point we didn't register any communication hooks.
            for entry in FSDP.fsdp_modules(fsdp_model_with_hook):
                self.assertTrue(hasattr(entry, "communication_hook"))
                self.assertTrue(hasattr(entry, "communication_hook_state"))

            # Check a correct hook was registered for the root and all submodules if any
            for entry in FSDP.fsdp_modules(fsdp_model_with_hook):
                self.assertTrue(
                    fsdp_model_with_hook.communication_hook,
                    allreduce_hook.allreduce_hook.__qualname__
                )
                self.assertTrue(
                    fsdp_model_with_hook.communication_hook_state,
                    allreduce_state
                )

            for _ in range(10):
                optimizer2.zero_grad()
                out2 = fsdp_model_no_hook(input)
                loss2 = F.mse_loss(out2, target)
                loss2.backward()
                optimizer2.step()

            dist.barrier()

            for _ in range(10):
                optimizer1.zero_grad()
                out1 = fsdp_model_with_hook(input)
                loss1 = F.mse_loss(out1, target)
                loss1.backward()
                optimizer1.step()


            for orig_param, hook_param in zip(fsdp_model_no_hook.parameters(), fsdp_model_with_hook.parameters()):
                self.assertEqual(orig_param.grad, hook_param.grad)


instantiate_parametrized_tests(TestCommunicationHooks)

if __name__ == "__main__":
    run_tests()
