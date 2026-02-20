import ipaddress
import socket
from datetime import timedelta

import torch
import torch.distributed
from torch.distributed import TCPStore
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup


def torch_dist_barrier_and_cuda_sync():
    """Synchronize distributed training and CUDA operations.
    This function ensures that:
    1. All distributed processes reach this point (barrier)
    2. All CUDA operations are completed (synchronize)
    """
    torch.distributed.barrier()
    torch.cuda.synchronize()


def _is_ipv6(ip_str: str) -> bool:
    """Check if the given string is an IPv6 address.

    Handles bracket-wrapped addresses like ``[::1]`` as well as bare ``::1``.
    """
    try:
        ipaddress.IPv6Address(ip_str.strip("[]"))
        return True
    except (ipaddress.AddressValueError, ValueError):
        return False


def stateless_init_process_group(master_address, master_port, rank, world_size, device):
    """Create a StatelessProcessGroup and initialize NCCL communication.

    vLLM provides ``StatelessProcessGroup`` to create a process group without
    touching the global process group in ``torch.distributed``.  We build one
    with an explicit ``TCPStore`` instead of ``StatelessProcessGroup.create``
    for the following reasons:

    - IPv6 support: automatically selects AF_INET6 when the host is an IPv6 address.
    - Disables libuv to avoid flakiness on some PyTorch builds
      (ref: https://github.com/pytorch/pytorch/pull/150215).
    - Pre-binds the listen socket on rank 0 to avoid address/port races.
    """
    launch_server = rank == 0
    if launch_server:
        family = socket.AF_INET6 if _is_ipv6(master_address) else socket.AF_INET
        listen_socket = socket.socket(family, socket.SOCK_STREAM)
        listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if family == socket.AF_INET6:
            try:
                listen_socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            except (AttributeError, OSError):
                pass
        bind_host = master_address.strip("[]")
        listen_socket.bind((bind_host, master_port))
        listen_socket.listen()
        listen_fd = listen_socket.fileno()
    else:
        listen_socket = None
        listen_fd = None

    store = TCPStore(
        host_name=master_address,
        port=master_port,
        world_size=world_size,
        is_master=launch_server,
        timeout=timedelta(seconds=300),
        use_libuv=False,
        master_listen_fd=listen_fd,
    )

    pg = StatelessProcessGroup(
        rank=rank,
        world_size=world_size,
        store=store,
        socket=listen_socket,
        data_expiration_seconds=3600,
    )
    return PyNcclCommunicator(pg, device=device)
