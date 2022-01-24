import socket
import time
from collections import Counter

import ray

ray.init(address="auto")

print(
    f"""This cluster consists of
    {len(ray.nodes())} nodes in total
    {ray.cluster_resources()["CPU"]} CPU resources in total
"""
)


@ray.remote
def echo_local_ip() -> str:
    """Wait for a bit and return the local IP address.

    Returns:
        str: The local IP address
    """
    time.sleep(0.001)
    return socket.gethostbyname(socket.gethostname())


object_ids = [echo_local_ip.remote() for _ in range(10000)]
ip_addresses = ray.get(object_ids)

print("Tasks executed")
for ip_address, num_tasks in Counter(ip_addresses).items():
    print(f"    {num_tasks} tasks on {ip_address}")
