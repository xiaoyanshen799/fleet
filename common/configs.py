from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from omegaconf import OmegaConf


@dataclass
class IDKwargsConfig:
    id: str
    kwargs: Dict = field(default_factory=dict)


@dataclass
class DatasetConfig:
    path: str = "static/data"
    name: str = "cifar10"
    partitioner: IDKwargsConfig = field(default_factory=lambda: IDKwargsConfig(id="IidPartitioner"))
    force_create: bool = False
    test_size: float = 0.2
    server_eval: bool = True
    train_split_key: str = "train"
    test_split_key: str = "test"
    input_features: list[str] = field(default_factory=lambda: ["img"])
    target_features: list[str] = field(default_factory=lambda: ["label"])
    num_classes: int = 10
    input_channels: int = 3
    image_size: int = 32


@dataclass
class ZMQConfig:
    enable: bool = False
    host: str = "localhost"
    port: int = 5555


@dataclass
class FLServerConfig:
    log_to_stream: bool = True
    logging_level: str = "INFO"
    strategy: str = "FedAvg"
    min_fit_clients: int = 1
    min_evaluate_clients: int = min_fit_clients
    min_available_clients: int = min_fit_clients
    num_rounds: int = 1
    fraction_fit: float = 1
    fraction_evaluate: float = 1
    server_eval: bool = False
    val_batch_size: int = 128
    server_param_init: bool = True
    stop_by_accuracy: bool = False
    accuracy_level: float = 0.8
    collect_metrics: bool = False
    # For now, collecting metrics periodically doesn't work as expected since Flower uses a single-threaded gRPC channel
    # Follow up on Flower discussions: https://discuss.flower.ai/t/concurrent-grpc-calls/1116
    # collect_metrics_interval: int = 60
    zmq: ZMQConfig = field(default_factory=ZMQConfig)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FLClientConfig:
    log_to_stream: bool = True
    logging_level: str = "INFO"
    train_batch_size: int = 32
    val_batch_size: int = 128
    local_epochs: int = 1
    learning_rate: float = 1e-3
    log_interval: int = 100
    collect_metrics: bool = False
    collect_metrics_interval: int = 5
    zmq: ZMQConfig = field(default_factory=ZMQConfig)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CustomTopologyConfig:
    path: str
    class_name: str


@dataclass
class TopologyConfig:
    """Topology configuration"""
    source: str = "topohub"
    topohub_id: Optional[str] = None
    custom_topology: Optional[CustomTopologyConfig] = None
    link_util_key: str = "deg"  # {deg, uni, org} for topohub, and user-defined for custom topologies
    link_config: Dict = field(default_factory=dict)
    switch_config: Dict = field(default_factory=lambda: {"failMode": "standalone", "stp": True})
    extra: Dict = field(default_factory=dict)


@dataclass
class ClientLimitsConfig:
    distribution: str = "homogeneous"  # {homogeneous, heterogeneous}
    cpu: float = 0.7
    mem: int = 1024
    cpu_mem_tuple: Optional[tuple] = None  # (cpu, mem) for heterogeneous distribution


@dataclass
class FLHostConfig:
    """FL configuration"""
    clients_number: int = 10
    image: str = "fl-app:latest"
    network: str = "10.0.0.0/16"
    server_placement: IDKwargsConfig = field(default_factory=lambda: IDKwargsConfig(id="highest_degree"))
    client_placement: IDKwargsConfig = field(default_factory=lambda: IDKwargsConfig(id="lowest_degree"))
    clients_limits: ClientLimitsConfig = field(default_factory=ClientLimitsConfig)
    server_limits: Optional[ClientLimitsConfig] = field(default_factory=ClientLimitsConfig)
    extra: Dict = field(default_factory=dict)


@dataclass
class BGConfig:
    """Background traffic configuration"""
    enabled: bool = False
    image: str = "bg-traffic:latest"
    network: str = "10.1.0.0/16"
    clients_limits: ClientLimitsConfig = field(default_factory=ClientLimitsConfig)
    rate_distribution: IDKwargsConfig = field(default_factory=lambda: IDKwargsConfig(id="poisson"))
    time_distribution: IDKwargsConfig = field(default_factory=lambda: IDKwargsConfig(id="poisson"))
    generator: IDKwargsConfig = field(default_factory=lambda: IDKwargsConfig(id="iperf"))
    extra: Dict = field(default_factory=dict)


@dataclass
class SDNConfig:
    """SDN configuration"""
    sdn_enabled: bool = False
    controller_ip: str = "localhost"
    controller_port: int = 6633
    controller_type: str = "openflow"
    extra: Dict = field(default_factory=dict)


@dataclass
class NetConfig:
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    fl: FLHostConfig = field(default_factory=FLHostConfig)
    bg: BGConfig = field(default_factory=BGConfig)
    sdn: SDNConfig = field(default_factory=SDNConfig)


def get_configs_from_file(path, configs_name, data_class_type):
    cfg = OmegaConf.load(path)
    cfg = OmegaConf.to_container(getattr(cfg, configs_name), resolve=True)
    cfg = OmegaConf.merge(data_class_type(), cfg)
    return cfg
