"""Prepare original data and run the configured full experiment with durable status."""
import _bootstrap
import argparse
import os
import time
from acpsa_flids.utils.config import load_config, resolve
from acpsa_flids.utils.logger import write_json
from acpsa_flids.data.dataset import prepare
from acpsa_flids.federated.trainer import train

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--resume")
    a=p.parse_args(); config=load_config(a.config)
    out=resolve(config["output"]); out.mkdir(parents=True,exist_ok=True)
    state={"pid":os.getpid(),"config":str(resolve(a.config)),"started_unix":time.time(),"stage":"preparing"}
    try:
        write_json(out/"runner_status.json",state)
        data=prepare(config)
        state["stage"]="training"; write_json(out/"runner_status.json",state)
        result=train(config,data,a.resume)
        state.update(stage="completed",completed_rounds=result["completed_rounds"],finished_unix=time.time())
        write_json(out/"runner_status.json",state)
    except BaseException as error:
        state.update(stage="failed",error=repr(error),finished_unix=time.time())
        write_json(out/"runner_status.json",state)
        raise
