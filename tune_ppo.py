from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog
from ray.tune.experiment.trial import ExportFormat
from ray.util.client import ray
from ray.tune.registry import register_env
from src.game import scotland_yard_game_logic as scotland_yard_game
from src.environments.rlib.scotland_yard_environment import ScotlandYardEnvironment

from ray.rllib.examples.models.centralized_critic_models import (
    TorchCentralizedCriticModel,
)


def get_all_subdirs(directory):
    import os
    all_subdirs = [os.path.join(directory, d) for d in os.listdir(directory) if
                   os.path.isdir(os.path.join(directory, d))]
    return all_subdirs


def get_latest_checkpoint():
    import os
    tuned_results_dir = 'tuned_results'
    if not os.path.exists(tuned_results_dir):
        return None
    all_subdirs = get_all_subdirs(tuned_results_dir)
    if len(all_subdirs) == 0:
        return None
    latest_checkpoint_dir = max(all_subdirs, key=os.path.getmtime)

    all_subdirs = get_all_subdirs(latest_checkpoint_dir)

    if len(all_subdirs) == 0:
        return None
    latest_checkpoint_dir = max(all_subdirs, key=os.path.getmtime)

    all_subdirs = get_all_subdirs(latest_checkpoint_dir)
    if len(all_subdirs) == 0:
        return None
    latest_checkpoint_dir = max(all_subdirs, key=os.path.getmtime)
    return latest_checkpoint_dir


if __name__ == "__main__":
    ray.init(num_gpus=1)

    def env_creator(env_config):
        return ScotlandYardEnvironment({})  # return an env instance


    register_env("scotland_env", env_creator)

    ModelCatalog.register_custom_model(
        "cc_model",
        TorchCentralizedCriticModel
    )

    repeat = 10
    directory = "C:/Users/cmich/Documents/Projects/school/scotland_yard_thesis/tuned_results"
    tune_config = {
        "env": "scotland_env",
        "num_workers": 4,
        "num_gpus": 0.2,
        "num_gpus_per_worker": 0.2,
        "num_envs_per_worker": 1,
        "model": {
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
        },
        "lr": 3e-4,
        "optimization": {
            "optimizer": "adam",
            "adam_epsilon": 1e-8,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
        },
        "gamma": 0.60,
        "num_sgd_iter": 20,
        "sgd_minibatch_size": 1000,
        "rollout_fragment_length": 1000,
        "train_batch_size": 8000,
        "stop": {"training_iteration": 20},
        "exploration_config": {},
        "multiagent": {
            "policies_to_train": ["mr_x_policy", "cop_policy"],
            "policies": {
                "mr_x_policy": scotland_yard_game.MR_X_POLICY_SPEC,
                "cop_policy": scotland_yard_game.COP_POLICY_SPEC,
            },
            "policy_mapping_fn": lambda agent_id, episode, worker,
                                        *kw: "mr_x_policy" if agent_id == "mr_x" else "cop_policy"
        },
        "framework": "torch",
        "custom_model": "cc_model",
        "checkpoint_freq": 5,
        "name": "PPO",
        "local_dir": directory,
        "checkpoint_at_end": True,
        "reuse_actors": True,
    }

    checkpoint_path = get_latest_checkpoint()
    print(checkpoint_path)
    if checkpoint_path:
        tune_config["restore"] = checkpoint_path
        print("restoring from checkpoint")

    result = tune.run(
        run_or_experiment=PPO,
        config=tune_config,
        local_dir=directory,
        checkpoint_freq=5,
        checkpoint_at_end=True,
        stop={"training_iteration": 20},
        export_formats=[ExportFormat.H5],
        reuse_actors=True,
    )
    print(result)
    ray.shutdown()
