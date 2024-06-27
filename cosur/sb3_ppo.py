import json
import numpy as np
import time

from argparse import ArgumentParser
from pathlib import Path
from stable_baselines3.ppo.ppo import PPO
from torch import nn
from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from sb3_utils import configure_learning_pipeline, linear_schedule
from dvrk_point_reach_env import DVRKEnv, ActionType


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--show", "-s", type=Path)
    args = arg_parser.parse_args()

    add_render_callback = False
    normalize_reward = True
    normalize_observations = True
    reward_clip = np.inf
    obs_clip = np.inf
    target_dt = 0.1
    time_limit = 25.0  # seconds
    simulation_hz = 50

    config = {
        "total_timesteps": int(2e6),  # How many environment steps to run for training
        "number_of_envs": 12,  # Number of parallel environments to collect interactions from
        "checkpoint_distance": int(5e5),  # How often to save the model in terms of environment steps
        "frame_stack": 2,  # Number of frames to stack for each observation
        "videos_per_run": 5,  # How many videos to record per run
        "video_length": 2 * time_limit / target_dt,  # How many steps to record in each video
        "max_episode_steps": time_limit / target_dt,  # Time limit for each episode
    }

    ppo_kwargs = {
        "policy": "MlpPolicy",
        "n_steps": 2048,
        "batch_size": 256,
        "learning_rate": linear_schedule(5e-4),  # default: 2.5e-4
        "n_epochs": 4,
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "clip_range": linear_schedule(0.2),  # default: 0.1
        "clip_range_vf": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": None,
        "policy_kwargs": {
            "activation_fn": nn.ReLU,
            "net_arch": {
                "pi": [256, 256],
                "vf": [256, 256],
            },
        },
    }

    env_kwargs = {
        # "fixed_target_position": [0.0, 0.5, 0.05],
        # "fixed_target_orientation": [0.0, 0.0, 0.0, 1.0],
        "manually_standardize_obs": False,
        "render_mode": "rgb_array",
        "action_type": ActionType.RELATIVE_POSITION,
        "simulation_hz": simulation_hz,
        "frame_skip": int(round(target_dt * simulation_hz)),
        "randomize_initial_joint_values": True,
        "egl_rendering": True,
        "target_position_threshold": 0.01,  # 1 cm
        "target_orientation_threshold": 10.0,  # 10 degrees
        "reward_feature_weights": {
            "num_collision_with_floor": 0.0,
            "simulation_unstable": 0.0,
            "orientation_difference": -1.0,
            "position_difference": -10.0,
            "done": 100.0,
        },
    }

    info_logging_keywords = [
        "reward_num_collision_with_floor",
        "reward_position_difference",
        "reward_orientation_difference",
        "reward_done",
        "distance_to_target_position",
        "distance_to_target_orientation",
        "euler_error_x",
        "euler_error_y",
        "euler_error_z",
        "success",
    ]

    if args.show is None:
        model, callback = configure_learning_pipeline(
            env_class=DVRKEnv,
            env_kwargs=env_kwargs,
            pipeline_config=config,
            monitoring_keywords=info_logging_keywords,
            normalize_observations=normalize_observations,
            normalize_reward=normalize_reward,
            ppo_kwargs=ppo_kwargs,
            render=add_render_callback,
            reward_clip=reward_clip,
            obs_clip=obs_clip,
        )
        # Train the model
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=callback,
            tb_log_name=f"PPO_DVRK_PointReach",
        )

        log_path = str(model.logger.dir)
        model.save(log_path + "/saved_model")

        norm_env = model.get_vec_normalize_env()

        standardization_info = {}
        if normalize_observations:
            standardization_info["obs_mean"] = norm_env.obs_rms.mean
            standardization_info["obs_var"] = norm_env.obs_rms.var
        if normalize_reward:
            standardization_info["rew_mean"] = norm_env.ret_rms.mean
            standardization_info["rew_var"] = norm_env.ret_rms.var

        # Save as json
        with open(log_path + "/standardization_info.json", "w") as f:
            json.dump(standardization_info, f)

        print("Saved model to", log_path + "/saved_model")
    else:
        # Show the learned model
        if not args.show.exists():
            raise ValueError(f"Model path {args.show} is not a file.")

        if "best" in args.show.stem:
            standardization_info_path = args.show.parent / "standardization_info_best_model.json"
        else:
            standardization_info_path = args.show.parent / "standardization_info.json"

        if not standardization_info_path.exists():
            raise ValueError(f"Standardization info path {standardization_info_path} does not exist")

        adapted_env_kwargs = env_kwargs.copy()
        adapted_env_kwargs["render_mode"] = "human"
        make_env = lambda: TimeLimit(DVRKEnv(**adapted_env_kwargs), max_episode_steps=config["max_episode_steps"])

        eval_env = VecFrameStack(
            VecNormalize(
                DummyVecEnv([make_env]),
                training=False,
                norm_obs=normalize_observations,
                norm_reward=normalize_reward,
                clip_obs=obs_clip,
                clip_reward=reward_clip,
            ),
            n_stack=config["frame_stack"],
        )

        standardization_info = json.load(open(standardization_info_path, "r"))
        norm_env = eval_env.venv
        if normalize_observations:
            norm_env.obs_rms.mean = standardization_info["obs_mean"]
            norm_env.obs_rms.var = standardization_info["obs_var"]
        if normalize_reward:
            norm_env.ret_rms.mean = standardization_info["rew_mean"]
            norm_env.ret_rms.var = standardization_info["rew_var"]

        model = PPO.load(str(args.show))

        while True:
            obs = eval_env.reset()
            done = False
            while not done:
                time_start = time.time()
                action, _ = model.predict(obs, deterministic=False)
                obs, reward, dones, info = eval_env.step(action)
                success = info[0]["success"]
                done = np.all(dones)
                time_end = time.time()
                time.sleep(max(0, target_dt - (time_end - time_start)))

                if done:
                    print(f"Episode done. {'Success' if success else 'Timeout'}")
                    time.sleep(1.0)
                    break
