import time
from typing import List, Tuple

from flwr.common import Metrics
from flwr.common import ndarrays_to_parameters
from flwr.server import strategy
from flwr.server.strategy import Strategy
from torch import nn

from common.loggers import warning
from common.static import CONTAINER_DATA_PATH
from flcode_pytorch.utils.contexts import ServerContext
from common.dataset_utils import get_dataloader, get_server_eval_dataset, basic_img_transform
from flcode_pytorch.utils.model_utils import get_weights, set_weights, test


# Define metric aggregation function
def get_aggregation_fn(metrics_agg_map: dict[str, str]) -> callable:
    def aggregate_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        aggregated_metrics = {}
        total_weight = sum(num_examples for num_examples, _ in metrics)
        num_clients = len(metrics)

        for metric, agg_method in metrics_agg_map.items():
            if agg_method == "weighted_average":
                aggregated_metrics[metric] = (
                        sum(num_examples * m[metric] for num_examples, m in metrics) / total_weight
                )
            elif agg_method == "average":
                aggregated_metrics[metric] = (
                        sum(m[metric] for _, m in metrics) / num_clients
                )
            elif agg_method == "sum":
                aggregated_metrics[metric] = sum(m[metric] for _, m in metrics)

        return aggregated_metrics

    return aggregate_metrics


def get_evaluate_fn(ctx, model):
    server_eval_dataset = get_server_eval_dataset(CONTAINER_DATA_PATH, ctx.dataset_cfg.name)
    if not server_eval_dataset:
        warning("Server evaluation dataset not found. Skipping server evaluation.")
        return None

    server_eval_dataloader = get_dataloader(
        server_eval_dataset,
        transform=basic_img_transform(ctx.dataset_cfg.input_features[0]),
        batch_size=ctx.server_cfg.val_batch_size
    )

    def evaluate(server_round, parameters, configs):
        set_weights(model, parameters)
        loss, accuracy = test(model, server_eval_dataloader, ctx.device,
                              input_features=ctx.dataset_cfg.input_features,
                              target_features=ctx.dataset_cfg.target_features)
        return loss, {"loss": loss, "accuracy": accuracy}

    return evaluate


def on_fit_config_fn(server_round: int) -> dict:
    # https://github.com/adap/flower/issues/5596 to enhance this to be able to customize configs per client
    return {
        'server-round': server_round,
        'server_dispatch_ts': time.perf_counter()
    }


def get_strategy(ctx: ServerContext, model: nn.Module) -> Strategy:
    fit_metrics_aggregation_fn = get_aggregation_fn({"loss": "average"})
    evaluate_metrics_aggregation_fn = get_aggregation_fn({"accuracy": "weighted_average", "loss": "average"})
    parameters = ndarrays_to_parameters(get_weights(model)) if ctx.server_cfg.server_param_init else None
    evaluate_fn = get_evaluate_fn(ctx, model) if ctx.server_cfg.server_eval else None
    strategy_class = getattr(strategy, ctx.server_cfg.strategy)
    strategy_kwargs = ctx.server_cfg.extra.get("strategy_kwargs", {})
    return strategy_class(
        fraction_fit=ctx.server_cfg.fraction_fit,
        fraction_evaluate=0 if ctx.server_cfg.server_eval else ctx.server_cfg.fraction_evaluate,
        min_fit_clients=ctx.server_cfg.min_fit_clients,
        min_evaluate_clients=ctx.server_cfg.min_evaluate_clients,
        min_available_clients=ctx.server_cfg.min_available_clients,
        evaluate_fn=evaluate_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters=parameters,
        on_fit_config_fn=on_fit_config_fn,
        on_evaluate_config_fn=on_fit_config_fn,
        **strategy_kwargs
    )
