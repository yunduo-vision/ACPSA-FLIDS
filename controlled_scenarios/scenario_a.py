from run import run, load_config

if __name__ == "__main__":
    run(load_config("configs/controlled_scenarios.yaml"), "A", ["classwise", "bayesian"])
