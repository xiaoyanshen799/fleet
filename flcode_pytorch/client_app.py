"""pytorchexample: A Flower / PyTorch app."""
import time
from collections import OrderedDict
from pathlib import Path

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, Config, Scalar

from common.dataset_utils import get_dataloader, basic_img_transform, get_client_partition
from common.loggers import init_zmq, configure_logger, info, debug, warning, to_zmq
from common.static import CONTAINER_LOG_PATH, CONTAINER_DATA_PATH, CONTAINER_RESOLVED_CONFIG_PATH
from .utils import client_metrics_utils
from .utils.client_metrics_utils import MetricsCollector
from common.configs import FLClientConfig, get_configs_from_file, DatasetConfig
from .utils.contexts import ClientContext
from .utils.model_utils import Net, get_weights, set_weights, test, train


def _model_size_bytes(parameters) -> int:
    """Return total parameter size in bytes."""
    total = 0
    for tensor in parameters:
        if hasattr(tensor, "nbytes"):
            total += tensor.nbytes
        elif hasattr(tensor, "size") and hasattr(tensor, "itemsize"):
            total += tensor.size * tensor.itemsize
    return total


class FlowerClient(NumPyClient):
    def __init__(self, ctx: ClientContext, train_dataloader, eval_dataloader, metrics_collector):
        self.ctx = ctx
        self.net = Net(
            num_classes=ctx.dataset_cfg.num_classes,
            input_channels=ctx.dataset_cfg.input_channels,
            image_size=ctx.dataset_cfg.image_size
        )
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.metrics_collector: MetricsCollector = metrics_collector

    def fit(self, parameters, config):
        # we can add batch size an quantization bits to config if needed
        local_epochs = config.get("local_epochs", self.ctx.client_cfg.local_epochs)
        learning_rate = config.get("learning_rate", self.ctx.client_cfg.learning_rate)

        model_size_mb = _model_size_bytes(parameters) / (1024 * 1024)
        recv_complete_ts = time.perf_counter()
        server_dispatch_ts = config.get("server_dispatch_ts")
        server_to_client_time = 0.0
        if isinstance(server_dispatch_ts, (int, float)):
            server_to_client_time = max(recv_complete_ts - server_dispatch_ts, 0.0)

        set_weights(self.net, parameters)
        tik = time.perf_counter()
        info(f"Starting Training - Round {config['server-round']}")

        loss = train(
            self.net,
            self.train_dataloader,
            self.ctx.device,
            epochs=local_epochs,
            lr=learning_rate,
            input_features=self.ctx.dataset_cfg.input_features,
            target_features=self.ctx.dataset_cfg.target_features,
        )
        tok = time.perf_counter()
        client_upload_timestamp = time.perf_counter()
        metrics = OrderedDict(
            client=self.ctx.simple_id,
            computing_start_time=tik,
            computing_finish_time=tok,
            loss=loss,
            model_size_mb=model_size_mb,
            server_to_client_time=server_to_client_time,
            client_upload_timestamp=client_upload_timestamp,
        )
        debug(f"Training Metrics: {metrics}")
        info(f"Finished Training - Round {config['server-round']}")
        info(f"Time taken for training: {tok - tik} seconds in Round {config['server-round']}")
        return get_weights(self.net), len(self.train_dataloader.dataset), metrics

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        if not self.eval_dataloader:
            warning("No validation data found, returning 0.0 for loss")
            return 0.0, 0, {}

        set_weights(self.net, parameters)
        tik = time.perf_counter()
        info(f"Starting Evaluation - Round {config['server-round']}")
        loss, accuracy = test(
            self.net,
            self.eval_dataloader,
            self.ctx.device,
            input_features=self.ctx.dataset_cfg.input_features,
            target_features=self.ctx.dataset_cfg.target_features,
        )
        metrics = OrderedDict(
            client=self.ctx.simple_id,
            computing_start_time=tik,
            computing_finish_time=time.perf_counter(),
            loss=loss,
            accuracy=accuracy,
        )
        debug(f"Eval Metrics: {metrics}")
        info(f"Finished Evaluation - Round {config['server-round']}")
        return loss, len(self.eval_dataloader.dataset), metrics

    def get_properties(self, config: Config) -> dict[str, Scalar]:
        result = {}
        props = config.get("props", '').split(",")
        metrics_agg = config.get("metrics_agg", "last")
        if "system" in props:
            result["system"]: client_metrics_utils.get_client_properties()
        elif "metrics" in props and self.metrics_collector:
            result["metrics"] = self.metrics_collector.get_metrics(aggregation=metrics_agg)
        elif "dataset" in props:
            result["dataset"] = client_metrics_utils.get_dataset_info(self.train_dataloader, self.eval_dataloader)
        debug(f"Properties: {result}")
        return result


def init_client(context: Context):
    # torch.set_num_threads(1)
    # # torch.set_num_interop_threads(1)

    client_cfg: FLClientConfig = get_configs_from_file(CONTAINER_RESOLVED_CONFIG_PATH, "fl_client", FLClientConfig)
    dataset_cfg: DatasetConfig = get_configs_from_file(CONTAINER_RESOLVED_CONFIG_PATH, "dataset", DatasetConfig)
    simple_id = context.node_config["cid"]
    log_file = Path(CONTAINER_LOG_PATH) / f"client_{context.node_config['cid']}.log"
    configure_logger("default", client_cfg.log_to_stream, log_file, client_cfg.logging_level)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ctx = ClientContext(
        cid=context.node_id,
        simple_id=simple_id,
        flwr_ctx=context,
        client_cfg=client_cfg,
        dataset_cfg=dataset_cfg,
        device=device
    )

    data_partition = get_client_partition(CONTAINER_DATA_PATH, dataset_cfg.name, simple_id)
    train_loader = get_dataloader(
        data_partition["train"],
        transform=basic_img_transform(dataset_cfg.input_features[0]),
        batch_size=client_cfg.train_batch_size,
        shuffle=True,
    )
    info(f"Training dataset size: {len(train_loader.dataset)}")
    eval_loader = None
    if "test" in data_partition:
        eval_loader = get_dataloader(
            data_partition["test"],
            transform=basic_img_transform(dataset_cfg.input_features[0]),
            batch_size=client_cfg.val_batch_size,
        )
        info(f"Validation dataset size: {len(eval_loader.dataset)}")

    client_info = {
        "cid": context.node_id, "simple_id": simple_id,
        "props": client_metrics_utils.get_client_properties(),
        "dataset": client_metrics_utils.get_dataset_info(train_loader, eval_loader)
    }

    debug(f"Client Info: {client_info}")
    if client_cfg.zmq.enable:
        init_zmq("default", client_cfg.zmq.host, client_cfg.zmq.port)
        to_zmq(f"client-init", client_info)

    metrics_collector = None
    if client_cfg.collect_metrics:
        metrics_collector = MetricsCollector(
            interval=client_cfg.collect_metrics_interval,
            publish_callback=lambda metrics: to_zmq(
                "client-metrics",
                {"cid": context.node_id, "metrics": metrics}
            ) if client_cfg.zmq.enable else None
        )

    return FlowerClient(ctx, train_loader, eval_loader, metrics_collector)


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""
    global flwr_client
    if not flwr_client:
        flwr_client = init_client(context)
    return flwr_client.to_client()


# Flower ClientApp
flwr_client = None
app = ClientApp(client_fn)
