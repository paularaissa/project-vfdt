# Federated Very Fast Decision Tree (Fed-VFDT)

This repository provides a prototype implementation of a **federated decision tree learning system** based on the **Very Fast Decision Tree (VFDT)** algorithm. The system is designed for **streaming, distributed, and potentially unlabeled data**, with global coordination across clients using federated aggregation strategies.

The implementation combines [Flower](https://flower.dev/) for federated orchestration and [River](https://riverml.xyz/) for data stream learning. Each client trains a local VFDT model and periodically shares split proposals with the server. The server aggregates these proposals and enforces a **synchronized global split** decision, ensuring model consistency across clients.

## Features

- Federated adaptation of the VFDT algorithm (Hoeffding Trees).
- Client-side training with local split proposals.
- Server-side coordination of tree growth via aggregation.
- Global synchronization of splits using feature-level consensus.
- Support for binary and multinomial splits.
- Communication-efficient: only sends messages on split attempts.
- Designed for online and incremental learning on streaming data.

## Requirements

- Python 3.13
- `flower` (= 1.15.0)
- `river`
- NumPy, pandas

## Directory Structure

To ensure proper execution, the project directory should be organized as follows:

```text
project-vfdt/
├── client_app/
│   ├── client_app.py        # Client logic (Flower Client)
│   ├── client_evaluation.py # Custom metrics (Kappa, G-Mean, etc.)
│   └── fht.py               # Federated Hoeffding Tree implementation
├── server_app/
│   ├── server_app.py        # Server strategy and startup
│   └── my_server.py         # Custom Server logic (dynamic rounds)
├── kdd99/                   # ⚠️ Dataset Directory
│   └── nodes/
│       └── 10nodes/         # Partitioned datasets
│           ├── client_0_dataset.csv
│           ├── client_1_dataset.csv
│           └── ...
├── logs/                # Output folder for logs/metrics
├── requirements.txt         # Project dependencies
└── README.md
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd project-vfdt
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment (venv or conda).
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Configuration

The system is configured to load data from `kdd99/nodes/10nodes/`.

1. **Format:** The data must be in CSV format.
2. **Naming Convention:** Files must be named `client_{id}_dataset.csv` (e.g., `client_0_dataset.csv`, `client_9_dataset.csv`).
3. **Location:** Place the files inside the `kdd99/nodes/10nodes/` directory.

*To change the dataset path, modify the `start_client` function in `client_app/client_app.py`:*
```python
abs_path = Path("your/custom/path") / "nodes" / f"{n_clients}nodes"
```

## Usage

The experiment requires the server to be running before any clients can connect.

### 1. Starting the Server
Open a terminal in the project root directory and execute the following command.

```bash
python server_app/server_app.py
```
*The server will initialize on port `8083`.*

### 2. Starting the Clients
Open a new terminal in the project root. The default configuration supports **10 clients**.

**Option A: Single Client Execution**
To run a specific client (e.g., client 1), use:
```bash
python client_app/client_app.py 1
```

**Option B: Batch Execution (Bash Script)**
To launch all 10 clients sequentially in the background:
```bash
for i in {1..9}
do
   echo "Starting client $i..."
   python client_app/client_app.py $i &
   sleep 2
done
wait
```

## Configuration & Parameters

### Server Configuration (`server_app/server_app.py`)
Key parameters in `StrategyVFDT` initialization:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `n_clients` | `10` | Total number of clients expected. |
| `global_grace_period` | `200` | Target for total instances across clients. |
| `aggregation_strategy` | `"majority-vote"` | `"quorum"`, `"majority-vote"`, or `"best-merit"`. |
| `support_percent` | `60` | Minimum consensus for `"quorum"` strategy. |

### Client Configuration (`client_app/client_app.py`)
Key parameters in `FedVFDTClient`:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `grace_period` | `20` | Instances observed before split attempt. |
| `delta` | `1e-5` | Hoeffding Bound confidence threshold. |
| `split_criterion` | `"gini"` | Criterion for evaluation (e.g., `"gini"`). |

## Logs & Results

Metrics are recorded in the `logs/` directory:
- **`server_log.csv`**: Global metrics (Accuracy, F1, Kappa, splits).
- **`client_{id}.csv`**: Local metrics per instance.
- **`client_{id}_time.csv`**: Total execution time per client.