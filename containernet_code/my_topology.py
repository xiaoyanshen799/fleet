import ipaddress
import itertools
import os
import random
import subprocess
from typing import Dict, List, Optional, Iterator, Any, Type

import topohub.mininet
from docker.types import DeviceRequest
from mininet.node import Docker
from mininet.topo import Topo

from common.loggers import info, warning
from common.static import *
from common.configs import TopologyConfig, NetConfig, ClientLimitsConfig


class TopoProcessor:
    """Base class for processing topologies."""
    NO_BG_WARNING = ("Warning: link {src} -> {dst} does not have a Link Utilization config."
                     "No BG traffic will run through it.")

    def __init__(self, cfg: TopologyConfig, *args, **kwargs):
        self.cfg = cfg
        self.args = args
        self.kwargs = kwargs

    def load_topology(self) -> None:
        """Load topology from file."""
        raise NotImplementedError("Subclasses must implement this method.")

    def get_topo(self) -> Topo:
        """Process topology into nodes and links."""
        raise NotImplementedError("Subclasses must implement this method.")


class CustomTopoProcessor(TopoProcessor):
    """Processor for custom Mininet topology."""

    def __init__(self, cfg: TopologyConfig, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.path = cfg.custom_topology.path
        self.class_name = cfg.custom_topology.class_name
        self.topo: Optional[Topo] = None
        self.loaded = False
        if not self.path or not self.class_name:
            raise ValueError("Custom topology must specify 'path' and 'class_name'.")

    def load_topology(self) -> None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("module", self.path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        topo_class = getattr(module, self.class_name)
        self.topo = topo_class()
        self.loaded = True

    def get_topo(self) -> Topo:
        """
        Process custom Mininet topology into nodes and links.
        The user is supposed to add all other details in the custom topology class.
        """
        if not self.loaded:
            self.load_topology()

        self.process_switches()
        self.process_links()
        return self.topo

    def process_switches(self):
        switches = self.topo.switches()
        for switch in switches:
            switch_info = self.topo.nodeInfo(switch)
            switch_info['degree'] = len(self.topo.g[switch])
            self.topo.setNodeInfo(
                name=switch.name,
                info=switch_info
            )

    def process_links(self):
        for link in self.topo.links():
            src, dst = link
            link_info = self.topo.linkInfo(src, dst)
            if self.cfg.link_util_key not in link_info:
                warning(self.NO_BG_WARNING.format(src=src, dst=dst))

            link_util = link_info.pop(self.cfg.link_util_key, 0.0)
            link_info['util'] = dict(fwd=link_util, bwd=link_util)
            self.topo.setlinkInfo(
                src=src,
                dst=dst,
                info=link_info,
            )


class TopohubTopoProcessor(TopoProcessor):
    def __init__(self, cfg: TopologyConfig, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)
        self.topohub_id = cfg.topohub_id
        self.topo: Optional[Topo] = None
        self.loaded = False
        # if not self.topohub_id in topohub.mininet.TOPO_NAMED_CLS:
        #     raise ValueError(f"Invalid Topohub topology ID: {self.topohub_id} {topohub.mininet.TOPO_NAMED_CLS.keys()}")

    def load_topology(self) -> None:
        """Load TopoHub topology."""
        topo_cls = topohub.mininet.TOPO_NAMED_CLS[self.topohub_id]
        self.topo = topo_cls()
        self.loaded = True

    def get_topo(self) -> Topo:
        """Process TopoHub topology into nodes and links."""
        if not self.loaded:
            self.load_topology()

        self.process_switches()
        self.process_links()
        return self.topo

    def process_switches(self):
        switches = self.topo.switches()
        switch_configs = self.cfg.switch_config
        for switch in switches:
            switch_info = self.topo.nodeInfo(switch)
            switch_info['degree'] = len(self.topo.g[switch])
            switch_info.update(switch_configs)  # Add switch-specific configurations
            self.topo.setNodeInfo(
                name=switch,
                info=switch_info
            )

    def process_links(self):
        link_configs = self.cfg.link_config
        for link in self.topo.links():
            src, dst = link
            link_info = self.topo.linkInfo(src, dst)
            link_info.update(link_configs)
            if self.cfg.link_util_key not in link_info["ecmp_fwd"]:
                warning(self.NO_BG_WARNING.format(src=src, dst=dst))

            link_info['util'] = dict(
                fwd=link_info.pop("ecmp_fwd").get(self.cfg.link_util_key, 0.0),
                bwd=link_info.pop("ecmp_bwd").get(self.cfg.link_util_key, 0.0),
            )
            self.topo.setlinkInfo(
                src=src,
                dst=dst,
                info=link_info,
            )


TOPOLOGY_PROCESSORS: Dict[str, Type[TopoProcessor]] = {
    "CUSTOM": CustomTopoProcessor,
    "TOPOHUB": TopohubTopoProcessor,
}


def highest_degree(nodes, **kwargs):
    single = kwargs.get("single", False)
    return max(nodes, key=lambda n: nodes[n]['degree']) if single else sorted(nodes, key=lambda n: nodes[n]['degree'],
                                                                              reverse=True)


def lowest_degree(nodes, **kwargs):
    single = kwargs.get("single", False)
    return min(nodes, key=lambda n: nodes[n]['degree']) if single else sorted(nodes, key=lambda n: nodes[n]['degree'])


def random_nodes(nodes, **kwargs):
    single = kwargs.get("single", False)
    return random.choice(nodes) if single else random.sample(nodes, k=len(nodes))


def specific_node(nodes, **kwargs):
    node_id = kwargs.get("node_id")
    if node_id is None or node_id not in nodes:
        raise ValueError("specific_node strategy requires 'node_id'")
    return node_id


PLACEMENT_STRATEGIES = {
    "highest_degree": highest_degree,
    "lowest_degree": lowest_degree,
    "specific_node": specific_node,
    "random": random_nodes,
}


def client_limits_generator(limits: ClientLimitsConfig) -> Iterator[Dict[str, Any]]:
    if not limits:
        yield {}
        return

    if limits.distribution == "heterogeneous":
        pair_generator = itertools.cycle(limits.cpu_mem_tuple)
    else:
        pair_generator = itertools.cycle([(limits.cpu, limits.mem)])

    for cpu, mem in pair_generator:
        yield {
            "mem_limit": f"{mem}m",
            "memswap_limit": f"{mem}m",
            "cpu_period": 100000,
            "cpu_quota": int(cpu * 100000),
        }


class TopologyHandler:
    """Class to handle network topology creation and management."""

    def __init__(self, log_path, net_cfg: NetConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.net_cfg = net_cfg
        self.log_path = log_path
        self.fl_server: Optional[str] = None
        self.fl_clients: List[str] = []
        self.bg_clients: List[str] = []
        self.gpu_available = self.check_gpu_available()
        self.topo = self._load_topology()
        self.setup_fl_components()

    def _load_topology(self) -> Topo:
        """Load and process topology data based on configuration."""
        try:
            source = self.net_cfg.topology.source.upper()
            processor = TOPOLOGY_PROCESSORS[source](self.net_cfg.topology)
        except KeyError:
            raise ValueError(f"Invalid topology source name: {self.net_cfg.topology.source}")

        return processor.get_topo()

    def setup_fl_components(self) -> None:
        """Build the complete network topology."""
        info("Building network topology...")

        fl_network_hosts = ipaddress.ip_network(self.net_cfg.fl.network).hosts()
        server_node_id = self._create_fl_server(fl_network_hosts)
        self._create_fl_clients(server_node_id, fl_network_hosts)

        if self.net_cfg.bg.enabled:
            if self.net_cfg.bg.network == self.net_cfg.fl.network:
                bg_network_hosts = fl_network_hosts  # continue from where FL clients left off
            else:
                bg_network_hosts = ipaddress.ip_network(self.net_cfg.bg.network).hosts()
            self._create_background_hosts(bg_network_hosts)
        info(f"Topology built: {len(self.topo.switches())} switches, {len(self.topo.links())} links")

    def _create_fl_server(self, fl_network_hosts) -> str:
        """Create FL server and connect it to the network."""
        switches = self.get_switches()
        placement_name = self.net_cfg.fl.server_placement.id
        placement_kwargs = self.net_cfg.fl.server_placement.kwargs
        server_switch = PLACEMENT_STRATEGIES[placement_name](switches, single=True, **placement_kwargs)

        server_limits = next(client_limits_generator(self.net_cfg.fl.server_limits))
        ip = str(next(fl_network_hosts))  # assigns the first IP to the server
        self.fl_server = self.topo.addHost(
            FL_SERVER_NAME,
            ip=ip,
            mac=self._ip_to_mac(ip),
            dimage=self.net_cfg.fl.image,
            **self._get_container_commons(self.log_path),
            **server_limits
        )
        self.topo.addLink(self.fl_server, server_switch)
        info(f"FL server placed on node {server_switch}")
        return server_switch

    def _create_fl_clients(self, server_node_id: str, fl_network_hosts) -> None:
        """Create FL clients and connect them to the network."""
        switches = self.get_switches()
        switches.pop(server_node_id)  # exclude server switch from client placement

        placement_name = self.net_cfg.fl.client_placement.id
        placement_kwargs = self.net_cfg.fl.client_placement.kwargs
        client_nodes = PLACEMENT_STRATEGIES[placement_name](switches, **placement_kwargs)

        limits_generator = client_limits_generator(self.net_cfg.fl.clients_limits)
        for i in range(1, self.net_cfg.fl.clients_number + 1):
            ip = str(next(fl_network_hosts))
            fl_client = self.topo.addHost(
                FL_NAME_FORMAT.format(id=i),
                ip=ip, mac=self._ip_to_mac(ip),
                dimage=self.net_cfg.fl.image,
                **self._get_container_commons(self.log_path),
                **self._get_gpu_configs(),
                **next(limits_generator)
            )

            client_switch = client_nodes[i % len(client_nodes)]
            self.topo.addLink(fl_client, client_switch)
            self.fl_clients.append(fl_client)

        info(f"Created {len(self.fl_clients)} FL clients")

    def _create_background_hosts(self, bg_network_hosts) -> None:
        """Create background traffic hosts."""
        limits_generator = client_limits_generator(self.net_cfg.bg.clients_limits)
        for switch in self.topo.switches():
            ip = str(next(bg_network_hosts))
            bg_host = self.topo.addHost(
                BG_NAME_FORMAT.format(switch=switch),
                ip=ip, mac=self._ip_to_mac(ip),
                dimage=self.net_cfg.bg.image,
                **self._get_container_commons(self.log_path),
                **next(limits_generator)
            )
            self.topo.addLink(bg_host, switch)
            self.bg_clients.append(bg_host)

        info(f"Created {len(self.bg_clients)} background hosts")

    def get_switches(self) -> Dict[str, Dict]:
        """Return list of switch info."""
        return {n: self.topo.nodeInfo(n) for n in self.topo.switches()}

    @staticmethod
    def _get_container_commons(log_path) -> Dict:
        """Return containernet configuration parameters."""
        absolute_path = os.getcwd()
        return {
            "volumes": [
                f"{absolute_path}/{log_path}:{CONTAINER_LOG_PATH}",
                f"{absolute_path}/{LOCAL_DATA_PATH}:{CONTAINER_DATA_PATH}",
                f"{absolute_path}/{LOCAL_SCRIPTS_PATH}:{CONTAINER_SCRIPTS_PATH}",
                f"{absolute_path}/{LOCAL_RESOLVED_CONFIG_PATH}:{CONTAINER_RESOLVED_CONFIG_PATH}",
                "/etc/localtime:/etc/localtime:ro",
                "/etc/timezone:/etc/timezone:ro"
            ],
            "sysctls": {"net.ipv4.tcp_congestion_control": "cubic"},
            "cls": Docker
        }

    def _get_gpu_configs(self) -> Dict:
        disable_gpu = self.net_cfg.fl.extra.get("disable_gpu", False)
        return {
            "device_requests": [DeviceRequest(count=-1, capabilities=[['gpu']])],
        } if self.gpu_available and not disable_gpu else {}

    @staticmethod
    def check_gpu_available() -> bool:
        """Check if NVIDIA GPU is available."""
        try:
            subprocess.check_output(['nvidia-smi'])
            info("NVIDIA GPU detected, enabling GPU support for FL clients.")
            return True
        except FileNotFoundError:
            info("NVIDIA GPU not detected or nvidia-smi not found, running without GPU support.")
            return False

    @staticmethod
    def _ip_to_mac(ip: str) -> str:
        """Convert IPv4 string to MAC address format with leading zeros."""
        return ':'.join(f'{b:02x}' for b in [0, 0] + list(map(int, ip.split('.'))))
