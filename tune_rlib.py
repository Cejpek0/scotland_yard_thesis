from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog
from ray.tune.experiment.trial import ExportFormat
from ray.util.client import ray
from ray.tune.registry import register_env
from src.states import scotland_yard as scotland_yard_game
from environments.rlib.scotland_yard_environment import ScotlandYardEnvironment

from ray.rllib.examples.models.centralized_critic_models import (
    TorchCentralizedCriticModel,
)

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
        "num_workers": 1,
        "num_gpus": 1,
        "num_gpus_per_worker": 0.5,
        "num_envs_per_worker": 1,
        "model": {
            "fcnet_hiddens": [512, 512, 256],
            "fcnet_activation": "relu",
        },
        "lr": 3e-4,
        "optimization": {
            "optimizer": "adam",
            "adam_epsilon": 1e-8,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
        },
        "gamma": 0.99,
        "num_sgd_iter": 10,
        "sgd_minibatch_size": 500,
        "rollout_fragment_length": 500,
        "train_batch_size": 4000,
        "stop": {"iter": 20},
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
    }

    

    result = tune.run(
        run_or_experiment=PPO,
        config=tune_config,
        local_dir=directory,
        checkpoint_freq=5,
        checkpoint_at_end=True,
        stop={"iter": 20},
        export_formats=[ExportFormat.H5]
    )
    print(result)
    ray.shutdown()
