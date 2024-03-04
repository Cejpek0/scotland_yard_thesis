import os

from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.models import ModelCatalog
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
        return ScotlandYardEnvironment({})


    register_env("scotland_env", env_creator)

    ModelCatalog.register_custom_model(
        "cc_model",
        TorchCentralizedCriticModel
    )

    my_config = (PPOConfig()
                 .training(model={"custom_model": "cc_model"})
                 .rl_module(_enable_rl_module_api=False)
                 .training(_enable_learner_api=False))

    my_config["policies"] = {
        "mr_x_policy": scotland_yard_game.MR_X_POLICY_SPEC,
        "cop_policy": scotland_yard_game.COP_POLICY_SPEC,
    }


    def policy_mapping_fn(agent_id, episode, worker):
        return "mr_x_policy" if agent_id == "mr_x" else "cop_policy"


    my_config["policy_mapping_fn"] = policy_mapping_fn

    my_config["num_iterations"] = 20
    my_config["num_rollout_workers"] = 3
    my_config["reuse_actors"] = True
    my_config.resources(num_gpus=1, num_gpus_per_worker=0.2)
    my_config.framework("torch")

    # Set the config object's env.
    algo = PPO(env="scotland_env", config=my_config)

    repeat = 1000
    # check if trained policies exist
    directory = "trained_policies"

    if os.path.exists(directory):
        print("Loading policies")
        algo.restore(directory)
    for i in range(repeat):
        print("Training iteration {} of {}".format(i + 1, repeat))
        print(algo.train())
        if i % 4 == 0:
            print("Saving policies")
            algo.save(directory)
    algo.save(directory)

    ray.shutdown()

    print("Done")
