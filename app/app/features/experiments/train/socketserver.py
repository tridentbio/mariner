from app.db.session import SessionLocal
from app.features.experiments.controller import (
    log_hyperparams,
    log_metrics,
    send_ws_epoch_update,
)
from sockets.socket_server import ClusterSocketsController


class LoggersServer(ClusterSocketsController):
    addr = ("", 8889)

    def on_experiment_update(self, parsed_msg):
        msg_type = parsed_msg["type"]
        data = parsed_msg["data"]
        experiment_id = parsed_msg["experiment_id"]
        db = SessionLocal()
        if msg_type == "epoch_metrics":
            if not ("user_id" in parsed_msg and type(parsed_msg["int"]) == int):
                db.close()
                raise Exception("Needs user_id")
            user_id = parsed_msg["user_id"]
            send_ws_epoch_update(
                user_id=user_id,
                experiment_id=experiment_id,
                experiment_name="",
                metrics=data["metrics"],
                epoch=data["epoch"],
            )
        elif msg_type == "metrics":
            log_metrics(
                db,
                experiment_id=experiment_id,
                metrics=data["metrics"],
                history=data["history"],
                stage="train",
            )
        elif msg_type == "hyperparams":
            log_hyperparams(db, experiment_id, parsed_msg)

        db.close()
