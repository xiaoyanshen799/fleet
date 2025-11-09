import time
from collections import OrderedDict
from logging import INFO

import flwr
import flwr.server.server as flwr_server
from flwr.common import log
from flwr.server import History

from common.loggers import *
from common.static import CONTAINER_LOG_PATH
from flcode_pytorch.utils.contexts import ServerContext


def custom_fit_client(client, ins, timeout, group_id):
    fit_round_start_time = time.perf_counter()
    client, fit_res = default_fit_client(client, ins, timeout=timeout, group_id=group_id)
    fit_round_finish_time = time.perf_counter()
    fit_res.metrics["fit_start_time"] = fit_round_start_time
    fit_res.metrics["fit_finish_time"] = fit_round_finish_time
    client_upload_ts = fit_res.metrics.get("client_upload_timestamp")
    if isinstance(client_upload_ts, (int, float)):
        fit_res.metrics["client_to_server_time"] = max(fit_round_finish_time - client_upload_ts, 0.0)
    else:
        fit_res.metrics["client_to_server_time"] = 0.0
    fit_res.metrics.pop("client_upload_timestamp", None)
    fit_res.metrics.setdefault("model_size_mb", 0.0)
    fit_res.metrics.setdefault("server_to_client_time", 0.0)
    return client, fit_res


def custom_eval_client(client, ins, timeout, group_id):
    eval_round_start_time = time.perf_counter()
    client, evaluate_res = default_eval_client(client, ins, timeout, group_id)
    eval_round_finish_time = time.perf_counter()
    evaluate_res.metrics["eval_start_time"] = eval_round_start_time
    evaluate_res.metrics["eval_finish_time"] = eval_round_finish_time
    return client, evaluate_res


# Monkey-patch
default_fit_client = flwr_server.fit_client
default_eval_client = flwr_server.evaluate_client
flwr_server.fit_client = custom_fit_client
flwr_server.evaluate_client = custom_eval_client


def log_metrics_federated(metrics_type, current_round, fit_metrics_federated, log_path):
    for client, fit_res in fit_metrics_federated:
        metrics = OrderedDict(round=current_round, client_cid=client.cid, **fit_res.metrics)
        to_csv(f"{log_path}/{metrics_type}_federated_metrics.csv", row_dict=metrics)
        to_zmq(f"{metrics_type}-metrics", metrics)
        debug(f"{metrics_type} Metrics: {metrics}")


def log_failures_federated(metrics_type, current_round, fit_failures):
    for failure in fit_failures:
        # failure_info = OrderedDict(round=current_round, client_cid=client_cid, failure=failure)
        error(f"{metrics_type} Failure: {failure}")


def log_aggregated_metrics(aggregate_round_metrics, log_path):
    flatten_metrics = {}
    for key, value in aggregate_round_metrics.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flatten_metrics[f"{key}-{sub_key}"] = sub_value
        else:
            flatten_metrics[key] = value
    to_csv(f"{log_path}/aggregate_metrics.csv", flatten_metrics)
    to_zmq("aggregate-metrics", flatten_metrics)
    debug(f"Aggregated Metrics: {flatten_metrics}")


class MyServer(flwr.server.Server):
    """Custom server class that overrides the fit_client method."""

    def __init__(self, ctx, client_manager=None, strategy=None):
        """Initialize the server with a custom client manager and strategy."""
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.ctx: ServerContext = ctx

    def fit(self, num_rounds: int, timeout: Optional[float]) -> tuple[History, float]:
        """Run federated averaging for a number of rounds."""
        history = History()
        # Initialize parameters
        log(INFO, "[INIT]")
        self.parameters = self._get_initial_parameters(server_round=0, timeout=timeout)
        log(INFO, "Starting evaluation of initial global parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])
        else:
            log(INFO, "Evaluation returned no results (`None`)")

        # Run federated learning for num_rounds
        fit_start_time = time.perf_counter()

        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s]", current_round)

            aggregate_round_metrics = OrderedDict()
            aggregate_round_metrics["round"] = current_round

            round_start_time = time.perf_counter()
            res_fit = self.fit_round(
                server_round=current_round,
                timeout=timeout,
            )
            fit_end_time = time.perf_counter()
            if res_fit is not None:
                parameters_aggregated, fit_metrics_aggregated, (fit_metrics_federated, fit_failures) = res_fit
                if parameters_aggregated:
                    self.parameters = parameters_aggregated
                history.add_metrics_distributed_fit(server_round=current_round, metrics=fit_metrics_aggregated)
                log_metrics_federated("fit", current_round, fit_metrics_federated, CONTAINER_LOG_PATH)
                log_failures_federated("fit", current_round, fit_failures)
                aggregate_round_metrics["fit_time"] = fit_end_time - round_start_time
                aggregate_round_metrics["fit_metrics"] = fit_metrics_aggregated

            # Evaluate model using strategy implementation
            cen_eval_start_time = time.perf_counter()
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            cen_eval_end_time = time.perf_counter()
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "Fit Progress: R %s | M: %s | RT: %.2f sec | CT: %.2f sec",
                    current_round,
                    metrics_cen,
                    time.perf_counter() - round_start_time,
                    time.perf_counter() - fit_start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(server_round=current_round, metrics=metrics_cen)
                aggregate_round_metrics["cen_eval_time"] = cen_eval_end_time - cen_eval_start_time
                aggregate_round_metrics["cen_eval_metrics"] = metrics_cen

            # Federated Evaluation
            fed_eval_start_time = time.perf_counter()
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            fed_eval_end_time = time.perf_counter()
            if res_fed is not None:
                loss_aggregated, eval_metrics_aggregated, (eval_metrics_federated, eval_failures) = res_fed
                if loss_aggregated is not None:
                    history.add_loss_distributed(server_round=current_round, loss=loss_aggregated)
                    history.add_metrics_distributed(server_round=current_round, metrics=eval_metrics_aggregated)
                    total_time = fed_eval_end_time - fed_eval_start_time
                    log_metrics_federated("eval", current_round, eval_metrics_federated, CONTAINER_LOG_PATH)
                    log_failures_federated("eval", current_round, eval_failures)
                    aggregate_round_metrics["fed_eval_metrics"] = eval_metrics_aggregated
                    aggregate_round_metrics["fed_eval_total_time"] = total_time
            log_aggregated_metrics(aggregate_round_metrics, CONTAINER_LOG_PATH)

            if self.ctx.server_cfg.stop_by_accuracy:
                if self.reached_accuracy(aggregate_round_metrics, self.ctx.server_cfg.accuracy_level):
                    info("Reaching Accuracy Level, Breaking!")
                    break

        # Bookkeeping
        end_time = time.perf_counter()
        elapsed = end_time - fit_start_time
        return history, elapsed

    def disconnect_all_clients(self, timeout: Optional[float]) -> None:
        super().disconnect_all_clients(timeout)
        all_clients = self._client_manager.all()
        clients = [all_clients[k] for k in all_clients.keys()]
        for client in clients:
            self._client_manager.unregister(client)

    @staticmethod
    def reached_accuracy(aggregate_round_metrics, accuracy_level):
        cen_accuracy = aggregate_round_metrics.get('cen_eval_metrics', {}).get('accuracy', 0)
        fed_accuracy = aggregate_round_metrics.get('fed_eval_metrics', {}).get('accuracy', 0)
        return cen_accuracy >= accuracy_level or fed_accuracy >= accuracy_level
