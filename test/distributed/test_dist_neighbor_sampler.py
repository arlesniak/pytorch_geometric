import atexit
import socket
from typing import Optional, List, Tuple, Callable

# from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import pytest
import torch

from torch_geometric.data import Data
from torch_geometric.datasets import FakeHeteroDataset
from torch_geometric.distributed import (
    DistNeighborSampler,
    LocalFeatureStore,
    LocalGraphStore,
    Partitioner,
)
from torch_geometric.distributed.dist_context import DistContext
from torch_geometric.distributed.event_loop import ConcurrentEventLoop
from torch_geometric.distributed.rpc import init_rpc, shutdown_rpc
from torch_geometric.sampler import NeighborSampler, NodeSamplerInput
from torch_geometric.sampler.neighbor_sampler import node_sample
from torch_geometric.testing import onlyDistributedTest
from functools import wraps

from io import StringIO
import sys
from multiprocessing import Manager
from multiprocessing import Queue
manager = Manager()


def create_data(rank: int, world_size: int, time_attr: Optional[str] = None):
    if rank == 0:  # Partition 0:
        node_id = torch.tensor([0, 1, 2, 3, 4, 5, 9])
        edge_index = torch.tensor([  # Sorted by destination.
            [1, 2, 3, 4, 5, 0, 0],
            [0, 1, 2, 3, 4, 4, 9],
        ])
    else:  # Partition 1:
        node_id = torch.tensor([0, 4, 5, 6, 7, 8, 9])
        edge_index = torch.tensor([  # Sorted by destination.
            [5, 6, 7, 8, 9, 5, 0],
            [4, 5, 6, 7, 8, 9, 9],
        ])
    feature_store = LocalFeatureStore.from_data(node_id)
    graph_store = LocalGraphStore.from_data(
        edge_id=None,
        edge_index=edge_index,
        num_nodes=10,
        is_sorted=True,
    )

    graph_store.node_pb = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    graph_store.meta.update({'num_parts': 2})
    graph_store.partition_idx = rank
    graph_store.num_partitions = world_size

    edge_index = torch.tensor([  # Create reference data:
        [1, 2, 3, 4, 5, 0, 5, 6, 7, 8, 9, 0],
        [0, 1, 2, 3, 4, 4, 9, 5, 6, 7, 8, 9],
    ])
    data = Data(x=None, y=None, edge_index=edge_index, num_nodes=10)

    if time_attr == 'time':  # Create node-level time data:
        data.time = torch.tensor([5, 0, 1, 3, 3, 4, 4, 4, 4, 4])
        feature_store.put_tensor(data.time, group_name=None,
                                 attr_name=time_attr)

    elif time_attr == 'edge_time':  # Create edge-level time data:
        data.edge_time = torch.tensor([0, 1, 2, 3, 4, 5, 7, 7, 7, 7, 7, 11])

        if rank == 0:
            edge_time = torch.tensor([0, 1, 2, 3, 4, 5, 11])
        if rank == 1:
            edge_time = torch.tensor([4, 7, 7, 7, 7, 7, 11])

        feature_store.put_tensor(edge_time, group_name=None,
                                 attr_name=time_attr)

    return (feature_store, graph_store), data


def create_hetero_data(
    tmp_path: str,
    rank: int,
):
    graph_store = LocalGraphStore.from_partition(tmp_path, pid=rank)
    feature_store = LocalFeatureStore.from_partition(tmp_path, pid=rank)

    return feature_store, graph_store


def dist_neighbor_sampler(
    world_size: int,
    rank: int,
    master_port: int,
    disjoint: bool = False,
):
    dist_data, data = create_data(rank, world_size)

    current_ctx = DistContext(
        rank=rank,
        global_rank=rank,
        world_size=world_size,
        global_world_size=world_size,
        group_name='dist-sampler-test',
    )

    dist_sampler = DistNeighborSampler(
        data=dist_data,
        current_ctx=current_ctx,
        num_neighbors=[-1, -1],
        shuffle=False,
        disjoint=disjoint,
    )
    # Close RPC & worker group at exit:
    atexit.register(shutdown_rpc)

    init_rpc(
        current_ctx=current_ctx,
        master_addr='localhost',
        master_port=master_port,
    )
    dist_sampler.init_sampler_instance()
    dist_sampler.register_sampler_rpc()
    dist_sampler.event_loop = ConcurrentEventLoop(2)
    dist_sampler.event_loop.start_loop()

    if rank == 0:  # Seed nodes:
        input_node = torch.tensor([1, 6])
    else:
        input_node = torch.tensor([4, 9])

    inputs = NodeSamplerInput(input_id=None, node=input_node)

    # Evaluate distributed node sample function:
    out_dist = dist_sampler.event_loop.run_task(
        coro=dist_sampler.node_sample(inputs))

    sampler = NeighborSampler(
        data=data,
        num_neighbors=[-1, -1],
        disjoint=disjoint,
    )

    # Evaluate node sample function:
    out = node_sample(inputs, sampler._sample)

    # Compare distributed output with single machine output:
    assert torch.equal(out_dist.node, out.node)
    assert torch.equal(out_dist.row, out.row)
    assert torch.equal(out_dist.col, out.col)
    if disjoint:
        assert torch.equal(out_dist.batch, out.batch)
    assert out_dist.num_sampled_nodes == out.num_sampled_nodes
    assert out_dist.num_sampled_edges == out.num_sampled_edges


def dist_neighbor_sampler_temporal(
    world_size: int,
    rank: int,
    master_port: int,
    seed_time: torch.tensor = None,
    temporal_strategy: str = 'uniform',
    time_attr: str = 'time',
):
    dist_data, data = create_data(rank, world_size, time_attr)

    current_ctx = DistContext(
        rank=rank,
        global_rank=rank,
        world_size=world_size,
        global_world_size=world_size,
        group_name='dist-sampler-test',
    )

    num_neighbors = [-1, -1] if temporal_strategy == 'uniform' else [1, 1]
    dist_sampler = DistNeighborSampler(
        data=dist_data,
        current_ctx=current_ctx,
        num_neighbors=num_neighbors,
        shuffle=False,
        disjoint=True,
        temporal_strategy=temporal_strategy,
        time_attr=time_attr,
    )
    # Close RPC & worker group at exit:
    atexit.register(shutdown_rpc)

    init_rpc(
        current_ctx=current_ctx,
        master_addr='localhost',
        master_port=master_port,
    )
    dist_sampler.init_sampler_instance()
    dist_sampler.register_sampler_rpc()
    dist_sampler.event_loop = ConcurrentEventLoop(2)
    dist_sampler.event_loop.start_loop()

    if rank == 0:  # Seed nodes:
        input_node = torch.tensor([1, 6], dtype=torch.int64)
    else:
        input_node = torch.tensor([4, 9], dtype=torch.int64)

    inputs = NodeSamplerInput(
        input_id=None,
        node=input_node,
        time=seed_time,
    )

    # Evaluate distributed node sample function:
    out_dist = dist_sampler.event_loop.run_task(
        coro=dist_sampler.node_sample(inputs))

    sampler = NeighborSampler(
        data=data,
        num_neighbors=num_neighbors,
        disjoint=True,
        temporal_strategy=temporal_strategy,
        time_attr=time_attr,
    )

    # Evaluate node sample function:
    out = node_sample(inputs, sampler._sample)

    # Compare distributed output with single machine output:
    assert torch.equal(out_dist.node, out.node)
    assert torch.equal(out_dist.row, out.row)
    assert torch.equal(out_dist.col, out.col)
    assert torch.equal(out_dist.batch, out.batch)
    assert out_dist.num_sampled_nodes == out.num_sampled_nodes
    assert out_dist.num_sampled_edges == out.num_sampled_edges


class MPCaptureOutput:
    def __enter__(self):
        self._stdout_output = ''
        self._stderr_output = ''

        self._stdout = sys.stdout
        sys.stdout = StringIO()

        self._stderr = sys.stderr
        sys.stderr = StringIO()

        return self

    def __exit__(self, *args):
        self._stdout_output = sys.stdout.getvalue()
        sys.stdout = self._stdout

        self._stderr_output = sys.stderr.getvalue()
        sys.stderr = self._stderr

    def get_stdout(self):
        return self._stdout_output

    def get_stderr(self):
        return self._stderr_output


# queue = Queue()
# queue = manager.Queue()


def dist_neighbor_sampler_biased(
    world_size: int,
    rank: int,
    master_port: int,

    edge_weights: torch.tensor = None,
    weight_attr: str = 'edge_weight',
    queue: Queue = None,
):
    print("o co chodzi")
    import traceback
    # with CaptureOutput() as capturer:
    try:
        # pass
        # try:
            # pass
            # dist_data, data = create_data(rank, world_size, time_attr)
            #
            # current_ctx = DistContext(
            #     rank=rank,
            #     global_rank=rank,
            #     world_size=world_size,
            #     global_world_size=world_size,
            #     group_name='dist-sampler-test',
            # )
            #
            # # num_neighbors = [-1, -1] if temporal_strategy == 'uniform' else [1, 1]
            # num_neighbors = [1, 1]
            # dist_sampler = DistNeighborSampler(
            #     data=dist_data,
            #     current_ctx=current_ctx,
            #     num_neighbors=num_neighbors,
            #     shuffle=False,
            #     disjoint=True,
            #     # temporal_strategy=temporal_strategy,
            #     # time_attr=time_attr,
            #     weight_attr=weight_attr,
            # )
            # # Close RPC & worker group at exit:
            # atexit.register(shutdown_rpc)
            #
            # init_rpc(
            #     current_ctx=current_ctx,
            #     master_addr='localhost',
            #     master_port=master_port,
            # )
            # dist_sampler.init_sampler_instance()
            # dist_sampler.register_sampler_rpc()
            # dist_sampler.event_loop = ConcurrentEventLoop(2)
            # dist_sampler.event_loop.start_loop()
            #
            # if rank == 0:  # Seed nodes:
            #     input_node = torch.tensor([1, 6], dtype=torch.int64)
            # else:
            #     input_node = torch.tensor([4, 9], dtype=torch.int64)
            #
            # inputs = NodeSamplerInput(
            #     input_id=None,
            #     node=input_node,
            #     # time=seed_time,
            #     weight=edge_weights,
            # )
            #
            # # Evaluate distributed node sample function:
            # out_dist = dist_sampler.event_loop.run_task(
            #     coro=dist_sampler.node_sample(inputs))
            #
            # sampler = NeighborSampler(
            #     data=data,
            #     num_neighbors=num_neighbors,
            #     disjoint=True,
            #     # temporal_strategy=temporal_strategy,
            #     # time_attr=time_attr,
            #     weight_attr=weight_attr
            # )
            #
            # # Evaluate node sample function:
            # out = node_sample(inputs, sampler._sample)
            #
            # # Compare distributed output with single machine output:
            # assert torch.equal(out_dist.node, out.node)
            # assert torch.equal(out_dist.row, out.row)
            # assert torch.equal(out_dist.col, out.col)
            # assert torch.equal(out_dist.batch, out.batch)
            # assert out_dist.num_sampled_nodes == out.num_sampled_nodes
            # assert out_dist.num_sampled_edges == out.num_sampled_edges
        assert False, str(rank)
        # pass
    except Exception as e:
        # pass
        queue.put((None, traceback.format_exc()))
        raise e
        # queue.put((capturer.get_stdout(), traceback.format_exc()))

            # exc_info = sys.exc_info()
            # print(traceback.format_exc(), file=sys.stderr)
            # traceback.print_exception(*exc_info)
            # traceback.print_stack(file=sys.stderr)
            # del exc_info
    queue.put((None, traceback.format_exc()))

    # queue.put("du")
    # queue.put((capturer.get_stdout(), traceback.format_exc()))

    # queue.put((capturer.get_stdout(), capturer.get_stderr()))
    # 1/0
    # assert False


def dist_neighbor_sampler_hetero(
    data: FakeHeteroDataset,
    tmp_path: str,
    world_size: int,
    rank: int,
    master_port: int,
    input_type: str,
    disjoint: bool = False,
):
    dist_data = create_hetero_data(tmp_path, rank)

    current_ctx = DistContext(
        rank=rank,
        global_rank=rank,
        world_size=world_size,
        global_world_size=world_size,
        group_name='dist-sampler-test',
    )

    num_neighbors = [-1, -1]
    dist_sampler = DistNeighborSampler(
        data=dist_data,
        current_ctx=current_ctx,
        rpc_worker_names={},
        num_neighbors=num_neighbors,
        shuffle=False,
        disjoint=disjoint,
    )

    # Close RPC & worker group at exit:
    atexit.register(shutdown_rpc)

    init_rpc(
        current_ctx=current_ctx,
        master_addr='localhost',
        master_port=master_port,
    )
    dist_sampler.init_sampler_instance()
    dist_sampler.register_sampler_rpc()
    dist_sampler.event_loop = ConcurrentEventLoop(2)
    dist_sampler.event_loop.start_loop()

    # Create inputs nodes such that each belongs to a different partition
    node_pb_list = dist_data[1].node_pb[input_type].tolist()
    node_0 = node_pb_list.index(0)
    node_1 = node_pb_list.index(1)

    input_node = torch.tensor([node_0, node_1], dtype=torch.int64)

    inputs = NodeSamplerInput(
        input_id=None,
        node=input_node,
        input_type=input_type,
    )

    # Evaluate distributed node sample function:
    out_dist = dist_sampler.event_loop.run_task(
        coro=dist_sampler.node_sample(inputs))

    sampler = NeighborSampler(
        data=data,
        num_neighbors=num_neighbors,
        disjoint=disjoint,
    )

    # Evaluate node sample function:
    out = node_sample(inputs, sampler._sample)

    # Compare distributed output with single machine output:
    for k in data.node_types:
        assert torch.equal(out_dist.node[k].sort()[0], out.node[k].sort()[0])
        assert out_dist.num_sampled_nodes[k] == out.num_sampled_nodes[k]
        if disjoint:
            assert torch.equal(
                out_dist.batch[k].sort()[0],
                out.batch[k].sort()[0],
            )


def dist_neighbor_sampler_temporal_hetero(
    data: FakeHeteroDataset,
    tmp_path: str,
    world_size: int,
    rank: int,
    master_port: int,
    input_type: str,
    seed_time: torch.tensor = None,
    temporal_strategy: str = 'uniform',
    time_attr: str = 'time',
):
    dist_data = create_hetero_data(tmp_path, rank)

    current_ctx = DistContext(
        rank=rank,
        global_rank=rank,
        world_size=world_size,
        global_world_size=world_size,
        group_name='dist-sampler-test',
    )

    dist_sampler = DistNeighborSampler(
        data=dist_data,
        current_ctx=current_ctx,
        rpc_worker_names={},
        num_neighbors=[-1, -1],
        shuffle=False,
        disjoint=True,
        temporal_strategy=temporal_strategy,
        time_attr=time_attr,
    )

    # Close RPC & worker group at exit:
    atexit.register(shutdown_rpc)

    init_rpc(
        current_ctx=current_ctx,
        rpc_worker_names={},
        master_addr='localhost',
        master_port=master_port,
    )

    dist_sampler.init_sampler_instance()
    dist_sampler.register_sampler_rpc()
    dist_sampler.event_loop = ConcurrentEventLoop(2)
    dist_sampler.event_loop.start_loop()

    # Create inputs nodes such that each belongs to a different partition:
    node_pb_list = dist_data[1].node_pb[input_type].tolist()
    node_0 = node_pb_list.index(0)
    node_1 = node_pb_list.index(1)

    input_node = torch.tensor([node_0, node_1], dtype=torch.int64)

    inputs = NodeSamplerInput(
        input_id=None,
        node=input_node,
        time=seed_time,
        input_type=input_type,
    )

    # Evaluate distributed node sample function:
    out_dist = dist_sampler.event_loop.run_task(
        coro=dist_sampler.node_sample(inputs))

    sampler = NeighborSampler(
        data=data,
        num_neighbors=[-1, -1],
        disjoint=True,
        temporal_strategy=temporal_strategy,
        time_attr=time_attr,
    )

    # Evaluate node sample function:
    out = node_sample(inputs, sampler._sample)

    # Compare distributed output with single machine output:
    for k in data.node_types:
        assert torch.equal(out_dist.node[k].sort()[0], out.node[k].sort()[0])
        assert torch.equal(out_dist.batch[k].sort()[0], out.batch[k].sort()[0])
        assert out_dist.num_sampled_nodes[k] == out.num_sampled_nodes[k]


@onlyDistributedTest
@pytest.mark.parametrize('disjoint', [False, True])
def test_dist_neighbor_sampler(disjoint):
    mp_context = torch.multiprocessing.get_context('spawn')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]

    world_size = 2
    w0 = mp_context.Process(
        target=dist_neighbor_sampler,
        args=(world_size, 0, port, disjoint),
    )

    w1 = mp_context.Process(
        target=dist_neighbor_sampler,
        args=(world_size, 1, port, disjoint),
    )

    w0.start()
    w1.start()
    w0.join()
    w1.join()


@onlyDistributedTest
@pytest.mark.parametrize('seed_time', [None, torch.tensor([3, 6])])
@pytest.mark.parametrize('temporal_strategy', ['uniform'])
def test_dist_neighbor_sampler_temporal(seed_time, temporal_strategy):
    mp_context = torch.multiprocessing.get_context('spawn')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]

    world_size = 2
    w0 = mp_context.Process(
        target=dist_neighbor_sampler_temporal,
        args=(world_size, 0, port, seed_time, temporal_strategy, 'time'),
    )

    w1 = mp_context.Process(
        target=dist_neighbor_sampler_temporal,
        args=(world_size, 1, port, seed_time, temporal_strategy, 'time'),
    )

    w0.start()
    w1.start()
    w0.join()
    w1.join()


@onlyDistributedTest
@pytest.mark.parametrize('seed_time', [[3, 7]])
@pytest.mark.parametrize('temporal_strategy', ['last'])
def test_dist_neighbor_sampler_edge_level_temporal(
    seed_time,
    temporal_strategy,
):
    seed_time = torch.tensor(seed_time)

    mp_context = torch.multiprocessing.get_context('spawn')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]

    world_size = 2
    w0 = mp_context.Process(
        target=dist_neighbor_sampler_temporal,
        args=(world_size, 0, port, seed_time, temporal_strategy, 'edge_time'),
    )

    w1 = mp_context.Process(
        target=dist_neighbor_sampler_temporal,
        args=(world_size, 1, port, seed_time, temporal_strategy, 'edge_time'),
    )

    w0.start()
    w1.start()
    w0.join()
    w1.join()



# def onlyDistributedTest(func: Callable) -> Callable:


# def worker_wrap(func: Callable) -> Callable:
#     return func

def wrapper(f, *args, **kwargs):
    # queue = kwargs.queue
    import traceback
    print(f"kwargs={kwargs}")
    # res = None
    with MPCaptureOutput() as capturer:
        try:
            res = f(*args, **kwargs)
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
    # return res, capturer.get_stdout(), capturer.get_stderr()


def worker_wrap(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        queue = kwargs.queue
        import traceback
        print(f"kwargs={kwargs}")
        res = None
        with MPCaptureOutput() as capturer:
            try:
                res = f(*args, **kwargs)
            except Exception as e:
                traceback.print_exc(file=sys.stderr)
        return res, capturer.get_stdout(), capturer.get_stderr()
    return wrapper
#
    # with MPCaptureOutput() as capturer:
    #     try:
    #         print("nic takiego")
    #         # 1/0
    #         assert False
    #     except Exception as e:
    #         # _, _, tb = sys.exc_info()
    #         # traceback.print_tb(tb=tb, file=sys.stderr)
    #         # print(e.__traceback__, file=sys.stderr)
    #         # print(traceback.format_exc(), file=sys.stderr)
    #         traceback.print_exc(file=sys.stderr)
    # _queue.put((capturer.get_stdout(), capturer.get_stderr()))
    # # queue.put(("gg"), block=False)

from time import sleep

# @worker_wrap
# def worker(*args, **kwargs):
# # def worker(ctx, _queue):
#     import traceback
#     # queue.cancel_join_thread()
#     with MPCaptureOutput() as capturer:
#         try:
#             print("nic takiego")
#             # 1/0
#             assert False
#         except Exception as e:
#             # _, _, tb = sys.exc_info()
#             # traceback.print_tb(tb=tb, file=sys.stderr)
#             # print(e.__traceback__, file=sys.stderr)
#             # print(traceback.format_exc(), file=sys.stderr)
#             traceback.print_exc(file=sys.stderr)
#     # _queue.put((capturer.get_stdout(), capturer.get_stderr()))
#     # queue.put(("gg"), block=False)
# #
# from time import sleep
#


# class MyProcess(torch.multiprocessing.Process):
#     def __init__(self, *args, **kwargs):
#         # if kwargs.get('target'):
#         #     kwargs['target'] =
#         self.p = torch.multiprocessing.Process(*args, **kwargs)
#
#     def start(self):
#         self.p.start()
#
#     def join(self):
#         self.p.join()

def worker(*args):
    # queue.cancel_join_thread()
    print("worker_orig *args=", *args)
    if args[1] == 0:
        assert False
    else:
        1/0
    # _queue.put((ctx, "gg"))
    # queue.put(("gg"), block=False)


# def worker(ctx, _queue):
# @worker_wrap
def worker_capture(func, queue, *args, **kwargs):
    import traceback
    # queue.cancel_join_thread()
    try:
        with MPCaptureOutput() as capturer:
            try:
                print("Tutaj worker", args, kwargs)
                # 1/0
                func(*args, **kwargs)
                # assert False
            except Exception as e:
                # _, _, tb = sys.exc_info()
                # traceback.print_tb(tb=tb, file=sys.stderr)
                # print(e.__traceback__, file=sys.stderr)
                # print(traceback.format_exc(), file=sys.stderr)

                traceback.print_exc(file=sys.stderr)
                print("idzie maly wyjatek")
                raise e
    finally:
        queue.put((capturer.get_stdout(), capturer.get_stderr()))
        # queue.put((capturer.get_stdout(), traceback.format_exc()))


    # queue.put(("alt", "ble "+str(args[1])))

    # queue.put((capturer.get_stdout(), capturer.get_stderr()))



    # return (capturer.get_stdout(), capturer.get_stderr())
    # _queue.put((capturer.get_stdout(), capturer.get_stderr()))
    # queue.put(("gg"), block=False)
#

def run_mproc(mp_context, procs: Tuple[torch.multiprocessing.Process, ...]):
    world_size = len(procs)
    # queue = manager.Queue()
    # mp_context = torch.multiprocessing.get_context('spawn')
    queues = (mp_context.Queue(), mp_context.Queue())
    args1 = list(procs[0]._args)
    args1.insert(1, queues[0])
    args1.insert(2, world_size)
    procs[0]._args = args1

    args2 = list(procs[1]._args)
    args2.insert(1, queues[1])
    args2.insert(2, world_size)
    procs[1]._args = args2



    # stdout1 = stderr1 = ''
    # queue1.put("f")
    for p in procs:
        p.start()
    results = []
    for p, q in zip(procs, queues):
        stdout, stderr = q.get(timeout=5)
        results.append((p, stdout, stderr))
        # if stderr1:
        print(stdout)


    for p in procs:
        p.join()

    # while True:
    #     try:
    # stdout1, stderr1 = queue1.get(timeout=5)
    # stdout2, stderr2 = queue2.get(timeout=5)
        # except Exception as e:
        #     print(e)
        #     break

    for p, stdout, stderr in results:
        print(stdout)
        if p.exitcode != 0:
            pytest.fail(pytrace=False, reason=stderr)



    # for p, q in zip(procs, queues):
    #     stdout, stderr = q.get(timeout=5)
    #     # if stderr1:
    #     print(stdout)
    #     if p.exitcode != 0:
    #         pytest.fail(pytrace=False, reason=stderr)
    #     # else:
    #         # pytest.fail(pytrace=False, reason="no coz")
    #         # pytest.fail(pytrace=False, reason=p._args[1].get_std_err())
    #         # pytest.fail(pytrace=False, reason=p._args[1].get_std_err())
    #         # pytest.fail(pytrace=False, reasonp._args[1].get_std_err()="no coz")


# q1 = Queue()
# q2 = Queue()

# q1 = manager.Queue()
# q2 = manager.Queue()
#
@onlyDistributedTest
@pytest.mark.parametrize('seed_time', [[3, 7]])
def test_dist_neighbor_sampler_edge_exper(
        seed_time,
        # capsys,
):

    seed_time = torch.tensor(seed_time)

    mp_context = torch.multiprocessing.get_context('spawn')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]

    world_size = 2
    # world_size = 1
    # w0 = mp_context.Process(
    #     target=worker,
    #     args=("bb", queue, ),
    # )
        # args=(world_size, 0, port, seed_time, 'edge_weight', queue),),

    run_mproc(mp_context,
        (
            # mp_context.Process(target=wrapper, args=(worker, queue, world_size, 0, port, seed_time, 'edge_weight')),
            # mp_context.Process(target=wrapper, args=(worker, queue, world_size, 1, port, seed_time, 'edge_weight'))
            # mp_context.Process(target=worker_capture, args=(worker, queue, world_size, 0, port, seed_time, 'edge_weight')),
            # mp_context.Process(target=worker_capture, args=(worker, queue, world_size, 1, port, seed_time, 'edge_weight'))
            mp_context.Process(target=worker_capture, args=(worker, 0, port, seed_time, 'edge_weight')),
            mp_context.Process(target=worker_capture, args=(worker, 1, port, seed_time, 'edge_weight'))
        )
    )

    #
    # w0.start()
    # w1.start()
    # w0.join()
    # w1.join()

    # queue.put("null")

    # queue.put(("aa"))
    # while True:
    #     try:
    #         # self.results.append(self.q.get(block=True))
    #         stdout_output, stderr_output = queue.get(block=False)
    #     except Queue.Empty:
    #         break
    # sleep(5)

    # queue.get(block=False)

    # stdout_output, stderr_output = queue.get(block=False)
    # queue.task_done()



    # while True:
    #     try:
    #         stdout_output, stderr_output = queue.get(timeout=5)
    #     except Exception as e:
    #         print(e)
    #         break

    # out, err = capsys.readouterr()
    # print("out", stdout_output)
    # print("err", stderr_output)
    # print("err", err)

    # if w0.exitcode != 0:
    #     pytest.fail(pytrace=False, reason="no coz")

        # pytest.fail(pytrace=False, reason=stderr_output)
    # assert w1.exitcode == 0


@onlyDistributedTest
@pytest.mark.parametrize('seed_time', [[3, 7]])
def test_dist_neighbor_sampler_edge_weight(
        seed_time,
        # capsys,
):
    queue = manager.Queue()

    seed_time = torch.tensor(seed_time)

    mp_context = torch.multiprocessing.get_context('spawn')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]

    world_size = 2
    # world_size = 1
    # w0 = mp_context.Process(
    #     target=worker,
    #     args=("bb", queue, ),
    # )

    w0 = mp_context.Process(
        target=dist_neighbor_sampler_biased,
        args=(world_size, 0, port, seed_time, 'edge_weight', queue),
        # args=(world_size, 0, port, seed_time, 'edge_weight', queue),
    )

    w1 = mp_context.Process(
        target=dist_neighbor_sampler_biased,
        args=(world_size, 1, port, seed_time, 'edge_weight', queue),
    )

    # queue.put("null")

    w0.start()
    w1.start()
    # queue.put(("aa"))
    # while True:
    #     try:
    #         # self.results.append(self.q.get(block=True))
    #         stdout_output, stderr_output = queue.get(block=False)
    #     except Queue.Empty:
    #         break
    # sleep(5)

    # queue.get(block=False)

    # stdout_output, stderr_output = queue.get(block=False)
    # queue.task_done()

    while True:
        try:
            stdout_output, stderr_output = queue.get(timeout=5)
        except Exception as e:
            print(e)
            break

    w0.join()
    w1.join()
    # out, err = capsys.readouterr()
    print("out", stdout_output)
    print("err", stderr_output)
    # print("err", err)
    if w0.exitcode != 0:
        pytest.fail(pytrace=False, reason=stderr_output)
    # assert w1.exitcode == 0


@onlyDistributedTest
@pytest.mark.parametrize('disjoint', [False, True])
def test_dist_neighbor_sampler_hetero(tmp_path, disjoint):
    mp_context = torch.multiprocessing.get_context('spawn')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]

    world_size = 2
    data = FakeHeteroDataset(
        num_graphs=1,
        avg_num_nodes=100,
        avg_degree=3,
        num_node_types=2,
        num_edge_types=4,
        edge_dim=2,
    )[0]

    partitioner = Partitioner(data, world_size, tmp_path)
    partitioner.generate_partition()

    w0 = mp_context.Process(
        target=dist_neighbor_sampler_hetero,
        args=(data, tmp_path, world_size, 0, port, 'v0', disjoint),
    )

    w1 = mp_context.Process(
        target=dist_neighbor_sampler_hetero,
        args=(data, tmp_path, world_size, 1, port, 'v1', disjoint),
    )

    w0.start()
    w1.start()
    w0.join()
    w1.join()
    assert w0.exitcode != 0, "lala_0"
    assert w1.exitcode == 0, "lala_1"


@onlyDistributedTest
@pytest.mark.parametrize('seed_time', [None, [0, 0], [2, 2]])
@pytest.mark.parametrize('temporal_strategy', ['uniform', 'last'])
def test_dist_neighbor_sampler_temporal_hetero(
    tmp_path,
    seed_time,
    temporal_strategy,
):
    if seed_time is not None:
        seed_time = torch.tensor(seed_time)

    mp_context = torch.multiprocessing.get_context('spawn')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]

    world_size = 2
    data = FakeHeteroDataset(
        num_graphs=1,
        avg_num_nodes=100,
        avg_degree=3,
        num_node_types=2,
        num_edge_types=4,
        edge_dim=2,
    )[0]

    data['v0'].time = torch.full((data.num_nodes_dict['v0'], ), 1,
                                 dtype=torch.int64)
    data['v1'].time = torch.full((data.num_nodes_dict['v1'], ), 2,
                                 dtype=torch.int64)

    partitioner = Partitioner(data, world_size, tmp_path)
    partitioner.generate_partition()

    w0 = mp_context.Process(
        target=dist_neighbor_sampler_temporal_hetero,
        args=(data, tmp_path, world_size, 0, port, 'v0', seed_time,
              temporal_strategy, 'time'),
    )

    w1 = mp_context.Process(
        target=dist_neighbor_sampler_temporal_hetero,
        args=(data, tmp_path, world_size, 1, port, 'v1', seed_time,
              temporal_strategy, 'time'),
    )

    w0.start()
    w1.start()
    w0.join()
    w1.join()


@onlyDistributedTest
@pytest.mark.parametrize('seed_time', [[0, 0], [1, 2]])
@pytest.mark.parametrize('temporal_strategy', ['uniform', 'last'])
def test_dist_neighbor_sampler_edge_level_temporal_hetero(
    tmp_path,
    seed_time,
    temporal_strategy,
):
    seed_time = torch.tensor(seed_time)

    mp_context = torch.multiprocessing.get_context('spawn')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]

    world_size = 2
    data = FakeHeteroDataset(
        num_graphs=1,
        avg_num_nodes=100,
        avg_degree=3,
        num_node_types=2,
        num_edge_types=4,
        edge_dim=2,
    )[0]

    for i, edge_type in enumerate(data.edge_types):
        data[edge_type].edge_time = torch.full(
            (data[edge_type].edge_index.size(1), ), i, dtype=torch.int64)

    partitioner = Partitioner(data, world_size, tmp_path)
    partitioner.generate_partition()

    w0 = mp_context.Process(
        target=dist_neighbor_sampler_temporal_hetero,
        args=(data, tmp_path, world_size, 0, port, 'v0', seed_time,
              temporal_strategy, 'edge_time'),
    )

    w1 = mp_context.Process(
        target=dist_neighbor_sampler_temporal_hetero,
        args=(data, tmp_path, world_size, 1, port, 'v1', seed_time,
              temporal_strategy, 'edge_time'),
    )

    w0.start()
    w1.start()
    w0.join()
    w1.join()
