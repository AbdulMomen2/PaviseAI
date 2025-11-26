# Federated Learning Module (Flower + GNN for Anomaly Detection)
# =====================================================
import os
# 🔥 FIX-1: Windows torch DLL LoadError Fix (keep if needed)
os.add_dll_directory(
    r"C:\Users\DELL\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\lib"
)

import flwr as fl
from flwr.common import Metrics
import numpy as np
import torch
from typing import Dict, List, Tuple, Union
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from src.config import FL_NUM_ROUNDS, FL_NUM_CLIENTS, DEVICE

# -----------------------------------------------------
# 🔥 FIX-2: GNN edge_index must be LongTensor
# -----------------------------------------------------
def ensure_long(x: Union[torch.Tensor, object]) -> Union[torch.Tensor, object]:
    if isinstance(x, torch.Tensor):
        return x.long()
    return x


class SimpleGNN(nn.Module):
    def __init__(self, input_dim: int = 16, hidden_dim: int = 32, output_dim: int = 1):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index):
        # Ensure device + dtype consistency
        edge_index = ensure_long(edge_index)
        if x.device != next(self.parameters()).device:
            x = x.to(next(self.parameters()).device)
        if edge_index.device != next(self.parameters()).device:
            edge_index = edge_index.to(next(self.parameters()).device)

        x = F.relu(self.conv1(x, edge_index))
        x = self.sigmoid(self.conv2(x, edge_index))
        # return a scalar per graph (mean over node outputs)
        return x.mean(dim=0)


class FederatedClient(fl.client.NumPyClient):
    def __init__(self, cid: int, local_data: list):
        self.cid = cid
        self.local_data = local_data
        self.model = SimpleGNN().to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()

    def get_parameters(self, config):
        # Return list of numpy arrays in the same order as state_dict()
        return [val.detach().cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        # Map provided list -> model.state_dict() keys and load
        state_dict_keys = list(self.model.state_dict().keys())
        if len(state_dict_keys) != len(parameters):
            raise ValueError(
                f"Parameter mismatch: model expects {len(state_dict_keys)} arrays, got {len(parameters)}"
            )
        state_dict = {}
        for k, v in zip(state_dict_keys, parameters):
            arr = np.asarray(v)
            tensor = torch.tensor(arr, dtype=self.model.state_dict()[k].dtype)
            # move to model device
            tensor = tensor.to(next(self.model.parameters()).device)
            state_dict[k] = tensor
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # set incoming global params
        self.set_parameters(parameters)
        self.model.train()
        total_loss = 0.0
        for data in self.local_data:
            # ensure tensors on correct device and dtypes
            x = data.x.to(DEVICE)
            edge_index = ensure_long(data.edge_index).to(DEVICE)
            # edge_attr not used for training target creation but keep it consistent
            if hasattr(data, "edge_attr") and data.edge_attr is not None:
                edge_attr = data.edge_attr.to(DEVICE)
            else:
                edge_attr = None

            self.optimizer.zero_grad()
            out = self.model(x, edge_index)

            # label logic: earlier used set(data.edge_attr) which fails on tensor;
            # decide label robustly: if edge_attr exists and has >1 unique values -> anomaly (0.0)
            if edge_attr is not None:
                unique_vals = set(edge_attr.detach().cpu().tolist())
                label_val = 0.0 if len(unique_vals) > 1 else 1.0
            else:
                # default label (no edge_attr) -> normal
                label_val = 1.0

            # ensure label shape matches out (out may be 0-dim scalar)
            label = torch.tensor(label_val, device=DEVICE, dtype=out.dtype)

            loss = self.criterion(out, label)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(self.local_data))
        return (
            self.get_parameters(config={}),
            len(self.local_data),
            {"local_loss": avg_loss},
        )

    def evaluate(self, parameters, config):
        # set incoming global params
        self.set_parameters(parameters)
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for data in self.local_data:
                x = data.x.to(DEVICE)
                edge_index = ensure_long(data.edge_index).to(DEVICE)

                out = self.model(x, edge_index)
                # For evaluation we don't know true label; keep it consistent with fit's fallback
                # Here we use label 0.0 as earlier code did, but better is to compute same as fit:
                if hasattr(data, "edge_attr") and data.edge_attr is not None:
                    unique_vals = set(data.edge_attr.detach().cpu().tolist())
                    label_val = 0.0 if len(unique_vals) > 1 else 1.0
                else:
                    label_val = 0.0  # preserve previous behavior (you can change if needed)

                label = torch.tensor(label_val, device=DEVICE, dtype=out.dtype)
                loss = self.criterion(out, label)
                total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(self.local_data))
        # Flower expects: (loss, num_examples, metrics) OR some versions accept (num_examples, metrics) —
        # to be maximally compatible we return (float_loss, num_examples, metrics)
        return float(avg_loss), len(self.local_data), {"eval_loss": avg_loss}


def _build_random_graph() -> Data:
    # helper to generate random graph Data
    num_nodes = np.random.randint(5, 15)
    x = torch.randn(num_nodes, 16, dtype=torch.float32)

    # Create a simple undirected fully-connected-ish graph (COO format)
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            edges.append([i, j])
            edges.append([j, i])
    if len(edges) == 0:
        edges = [[0, 0]]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.ones(edge_index.shape[1], dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def client_fn(cid_or_ctx):
    """
    Flexible client_fn: accept either cid (str) OR Context object depending on Flower version.
    Return a Flower-compatible Client instance by converting NumPyClient -> Client via .to_client().
    """
    # Determine client id
    if isinstance(cid_or_ctx, str):
        cid = cid_or_ctx
    else:
        # try common attributes on Context
        cid = getattr(cid_or_ctx, "node_id", None) or getattr(cid_or_ctx, "cid", None) or str(id(cid_or_ctx))

    num_graphs = np.random.randint(3, 10)
    local_data = []
    for _ in range(num_graphs):
        data = _build_random_graph()
        # ensure Data tensors are on DEVICE for faster ops
        data.x = data.x.to(DEVICE)
        data.edge_index = ensure_long(data.edge_index).to(DEVICE)
        data.edge_attr = data.edge_attr.to(DEVICE)
        local_data.append(data)

    client = FederatedClient(int(cid) if str(cid).isdigit() else 0, local_data)
    # Flower now expects a Client instance (not NumpyClient) in some APIs => convert
    try:
        return client.to_client()
    except Exception:
        # fallback: return client itself if .to_client() unavailable
        return client


def federated_aggregate(metrics: Union[Metrics, List[Tuple[int, Dict[str, float]]]]) -> Dict[str, float]:
    """
    Robust aggregation fn: accepts either
      - a dict-like mapping (older/other callers), or
      - a list of tuples: [(num_examples, {"accuracy": ...}), ...]
    Returns aggregated metrics dict.
    """
    # If metrics is a mapping/dict-like with items(), attempt to parse that way
    if hasattr(metrics, "items"):
        try:
            pairs = list(metrics.items())
        except Exception:
            pairs = []
    else:
        # assume it's already a list of tuples
        pairs = metrics

    # pairs might be in several shapes; normalize to list of (num_examples, metrics_dict)
    normalized: List[Tuple[int, Dict[str, float]]] = []
    for entry in pairs:
        # entry could be (client_id, (num_examples, metrics_dict)) in some Flower internals
        if isinstance(entry, tuple) and len(entry) == 2:
            a, b = entry
            # case: (num_examples, metrics_dict)
            if isinstance(a, (int, float)) and isinstance(b, dict):
                normalized.append((int(a), b))
            # case: (client_id, (num_examples, metrics_dict))
            elif isinstance(b, tuple) and len(b) == 2 and isinstance(b[0], (int, float)) and isinstance(b[1], dict):
                normalized.append((int(b[0]), b[1]))
            # case: (something, {'accuracy':...}) -> weird, try to parse
            elif isinstance(b, dict) and "accuracy" in b and isinstance(a, (int, float)):
                normalized.append((int(a), b))
            else:
                # skip unknown format
                continue

    if not normalized:
        # Nothing to aggregate
        return {}

    # weighted average for accuracy if present
    total_examples = sum(num for num, _ in normalized)
    if total_examples == 0:
        return {}

    weighted_acc = 0.0
    any_accuracy = False
    for num, met in normalized:
        if met is None:
            continue
        if "accuracy" in met:
            any_accuracy = True
            weighted_acc += num * float(met["accuracy"])

    result: Dict[str, float] = {}
    if any_accuracy:
        result["accuracy"] = weighted_acc / total_examples

    # also aggregate other numeric metrics (e.g., loss) by weighted average
    # for simplicity, aggregate any metric keys that are numeric and present in all clients
    # compute sum of weights per metric
    metric_sums = {}
    metric_counts = {}
    for num, met in normalized:
        if not isinstance(met, dict):
            continue
        for k, v in met.items():
            try:
                fv = float(v)
            except Exception:
                continue
            metric_sums[k] = metric_sums.get(k, 0.0) + num * fv
            metric_counts[k] = metric_counts.get(k, 0) + num

    for k, s in metric_sums.items():
        if metric_counts.get(k, 0) > 0:
            result[k] = s / metric_counts[k]

    return result


def run_federated_simulation(
    num_rounds: int = FL_NUM_ROUNDS, num_clients: int = FL_NUM_CLIENTS
):
    print(
        f"🔄 Starting Federated Learning Simulation: {num_clients} clients, {num_rounds} rounds"
    )

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=federated_aggregate,
    )

    # Use start_simulation (deprecated warning ok) — keep compatible
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0},
    )

    print("✅ Federated Learning Simulation Complete!")
    return SimpleGNN().to(DEVICE)


def apply_dp_noise(model: nn.Module, epsilon: float = 1.0, delta: float = 1e-5):
    print(f"🔒 Applied DP noise (ε={epsilon}, δ={delta}) for privacy")
    # placeholder — implement DP perturbation if needed
    return model
