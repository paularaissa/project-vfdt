import timeit
import concurrent.futures
from typing import Optional, Tuple

from flwr.common import ReconnectIns
from flwr.server.server import Server
from flwr.server.client_manager import SimpleClientManager
from flwr.server.history import History
from flwr.server.strategy import Strategy

class MyServer(Server):
    """Custom Federated Learning server with dynamic rounds and graceful shutdown."""

    def __init__(self, strategy: Strategy) -> None:
        """Initializes the server with a client manager."""
        client_manager = SimpleClientManager()
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.final_round_triggered = False  # Flag to control the shutdown round.

    def fit(self, num_rounds: int, timeout: Optional[float] = None) -> Tuple[History, float]:
        """
        Executes training rounds dynamically.
        If the strategy signals 'stop_training', force a final round to notify clients.
        """
        history = History()

        print("🔄 Initializing global parameters...")
        self.parameters = self._get_initial_parameters(server_round=0, timeout=timeout)

        # Initial evaluation
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res:
            print(f"📊 Initial evaluation - Loss: {res[0]:.4f}, Metrics: {res[1]}")
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])

        start_time = timeit.default_timer()
        current_round = 0

        # The loop continues while rounds remain OR if the final round is required.
        while num_rounds > 0 or self.final_round_triggered:
            current_round += 1

            # Visual state check
            stop_status = getattr(self.strategy, "stop_training", False)
            status_msg = "🛑 SHUTDOWN" if stop_status else "🚀 TRAINING"
            print(f"\n{status_msg} Round {current_round} | Remaining 'official' rounds: {num_rounds}")

            # --- 1. Executes the round (configure_fit -> fit_clients -> aggregate_fit) ---
            # This is where the config `{"stop_training": "True"}` will be sent if `stop_status` is True.
            res_fit = self.fit_round(server_round=current_round, timeout=timeout)

            if res_fit:
                parameters_prime, fit_metrics, _ = res_fit
                if parameters_prime:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(server_round=current_round, metrics=fit_metrics)

            # --- 2. Post-round Evaluation ---
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen:
                loss_cen, metrics_cen = res_cen
                print(f"📊 Post-round Evaluation {current_round} - Loss: {loss_cen:.4f}, Metrics: {metrics_cen}")
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(server_round=current_round, metrics=metrics_cen)

            # --- 3. Loop Control Logic (HEART OF THE SYSTEM) ---

            # If we have already executed the final round (where we sent the stop), we can exit now.
            if self.final_round_triggered:
                print("🏁 Shutdown round completed. Shutting down server.")
                break

            # Verifica se a estratégia solicitou a paragem (todos os clientes passivos).
            stop_signal = getattr(self.strategy, "stop_training", False)

            if stop_signal:
                print("🛑 Strategy signaled stop. Scheduling FINAL round to notify clients.")
                # Force the loop to run one more time, ignoring `num_rounds`.
                self.final_round_triggered = True
                num_rounds = 1 # Ensures the `while` loop executes.
            else:
                # Standard dynamic logic.
                extra_rounds = getattr(self.strategy, "increase_num_rounds_by", 0)
                num_rounds += extra_rounds
                num_rounds -= 1 # We consume a round.

                if num_rounds <= 0 and extra_rounds == 0:
                    print("✅ Rounds exhausted without explicit stop request.")
                    break

        # Finalizes
        elapsed = timeit.default_timer() - start_time
        print(f"✅ Training completed in {elapsed:.2f} seconds.")

        # Optional: Uncomment if you want to ensure forced disconnection,
        # but with the above logic, clients should exit on their own.
        self.disconnect_all_clients(timeout=timeout)

        return history, elapsed

    def disconnect_all_clients(self, timeout: Optional[float]) -> None:
        """Sends forced disconnect signal (backup)."""
        print("🔌 Ensuring all channels are closed...")
        all_clients = self._client_manager.all()
        clients = list(all_clients.values())
        instruction = ReconnectIns(seconds=None)
        client_instructions = [(client, instruction) for client in clients]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(client.reconnect, ins, timeout) for client, ins in client_instructions]
            concurrent.futures.wait(futures)