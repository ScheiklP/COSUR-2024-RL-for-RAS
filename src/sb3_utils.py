import os
import numpy as np
import cv2
import json

from collections import deque
from typing import Callable, Type, Tuple, Optional, Union, List
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers.time_limit import TimeLimit

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecFrameStack, VecMonitor, VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        progress_remaining will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


class EpisodeInfoLoggerCallback(BaseCallback):
    def __init__(self, verbose: int = 0, exclude_keys: List[str] = ["r", "l", "t"]):
        super(EpisodeInfoLoggerCallback, self).__init__(verbose)
        self.exclude_keys = exclude_keys

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        iteration = (self.model.num_timesteps - self.model._num_timesteps_at_start) // self.model.n_envs // self.model.n_steps

        if self.locals["log_interval"] is not None and (iteration) % self.locals["log_interval"] == 0:
            if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
                logging_keys = self.model.ep_info_buffer[0].keys()

                for key in logging_keys:
                    if key in self.exclude_keys:
                        continue
                    else:
                        try:
                            upcast_key_data = np.array([ep_info[key] for ep_info in self.model.ep_info_buffer], dtype=np.float64)
                            safe_mean = np.nan if len(upcast_key_data) == 0 else np.nanmean(upcast_key_data)
                            self.logger.record(f"trajectory/ep_{key}_mean", safe_mean)
                        except TypeError:
                            if self.verbose > 0:
                                print(f"Episode info key {key} can not be averaged by np.nanmean. Will not try to log the key in the future.")
                                self.exclude_keys.append(key)


class AdjustLoggingWindow(BaseCallback):
    def __init__(self, window_length: int, verbose=0):
        super(AdjustLoggingWindow, self).__init__(verbose)
        self.window_length = window_length

    def _on_step(self) -> bool:
        return True

    def _on_training_start(self) -> None:
        self.model.ep_info_buffer = deque(maxlen=self.window_length)
        self.model.ep_success_buffer = deque(maxlen=self.window_length)
        return super()._on_training_start()


class RenderCallback(BaseCallback):
    def _on_step(self) -> bool:
        img = self.model._vec_normalize_env.render()
        cv2.imshow("render", img[:, :, ::-1])
        cv2.waitKey(1)
        return super()._on_step()


class SaveBestModelCallback(BaseCallback):
    def __init__(self, track_best: str = "r", direction: str = "max", verbose=0):
        super(SaveBestModelCallback, self).__init__(verbose)
        self.best_mean_value = -np.inf if direction == "max" else np.inf
        self.track_best = track_best
        self.direction = direction

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        save_path = str(self.model.logger.dir) + "/best_model"
        standardization_info_path = str(self.model.logger.dir) + "/standardization_info_best_model.json"
        if len(self.model.ep_info_buffer) > 0:
            mean_value = np.mean([ep_info[self.track_best] for ep_info in self.model.ep_info_buffer])

            save = False
            if self.direction == "max" and mean_value > self.best_mean_value:
                save = True
            elif self.direction == "min" and mean_value < self.best_mean_value:
                save = True

            if save:
                self.best_mean_value = mean_value
                self.model.save(save_path)
                print(f"New best model saved with mean {self.track_best} {mean_value}.")
                norm_env = self.model.get_vec_normalize_env()

                info = {}
                if hasattr(norm_env, "obs_rms"):
                    info["obs_mean"] = norm_env.obs_rms.mean.tolist()
                    info["obs_var"] = norm_env.obs_rms.var.tolist()

                if hasattr(norm_env, "ret_rms"):
                    info["rew_mean"] = norm_env.ret_rms.mean.tolist()
                    info["rew_var"] = norm_env.ret_rms.var.tolist()

                # Save as json
                if len(info) > 0:
                    with open(standardization_info_path, "w") as f:
                        json.dump(info, f)


class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(SaveVecNormalizeCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        log_path = str(self.model.logger.dir)
        norm_env = self.model.get_vec_normalize_env()

        standardization_info = {
            "obs_mean": norm_env.obs_rms.mean.tolist(),
            "obs_var": norm_env.obs_rms.var.tolist(),
            "rew_mean": norm_env.ret_rms.mean.tolist(),
            "rew_var": norm_env.ret_rms.var.tolist(),
        }

        # Save as json
        with open(log_path + "/standardization_info.json", "w") as f:
            json.dump(standardization_info, f)

        print("Saved standardization info to", log_path + "/standardization_info.json")


def configure_make_env(env_kwargs: dict, EnvClass: Type[gym.Env], max_episode_steps: int) -> Callable:
    """Returns a make_env function that is configured with given env_kwargs."""

    def make_env() -> gym.Env:
        env = EnvClass(**env_kwargs)
        env = TimeLimit(env, max_episode_steps=max_episode_steps)

        return env

    return make_env


def configure_learning_pipeline(
    env_class: Type[gym.Env],
    env_kwargs: dict,
    pipeline_config: dict,
    monitoring_keywords: Union[Tuple[str, ...], List[str]],
    normalize_observations: bool,
    normalize_reward: bool,
    ppo_kwargs: dict,
    render: bool,
    log_dir: str = "runs/",
    extra_callbacks: Optional[List[Type[BaseCallback]]] = None,
    random_seed: Optional[int] = None,
    dummy_run: bool = False,
    use_wandb: bool = False,
    model_checkpoint_distance: Optional[int] = None,
    reward_clip: Optional[float] = None,
    obs_clip: Optional[float] = None,
):
    if use_wandb:
        import wandb

    make_env = configure_make_env(
        env_kwargs,
        EnvClass=env_class,
        max_episode_steps=pipeline_config["max_episode_steps"],
    )

    if not dummy_run:
        env = SubprocVecEnv([make_env] * pipeline_config["number_of_envs"])
    else:
        env = DummyVecEnv([make_env])

    env.seed(np.random.randint(0, 99999) if random_seed is None else random_seed)

    env = VecMonitor(
        env,
        info_keywords=monitoring_keywords,
    )

    if pipeline_config["videos_per_run"] > 0:
        # the video recorder counts steps per step_await -> additionally devide by the number_of_envs
        recorder_distance = int(np.floor(pipeline_config["total_timesteps"] / pipeline_config["videos_per_run"] / pipeline_config["number_of_envs"]))
        recorder_steps = list(range(0, int(pipeline_config["total_timesteps"] / pipeline_config["number_of_envs"]), recorder_distance))

        # if the video is longer than the time steps spent in an env for a batch, the video will be cut off -> go back until it fits
        try:
            extra_batches_necessary_to_fit_video = int(np.floor(pipeline_config["video_length"] / ppo_kwargs["n_steps"]))
        except KeyError:
            extra_batches_necessary_to_fit_video = 0

        recorder_steps[-1] = recorder_steps[-1] - extra_batches_necessary_to_fit_video

        env = VecVideoRecorder(
            venv=env,
            video_folder=str(Path(log_dir) / f"videos{(f'/{wandb.run.id}') if use_wandb else ''}"),
            record_video_trigger=lambda x: x in recorder_steps,
            video_length=pipeline_config["video_length"],
        )

    # Reward and observation normalization
    normalize_kwargs = {}
    if reward_clip is not None:
        normalize_kwargs["clip_reward"] = reward_clip

    if obs_clip is not None:
        normalize_kwargs["clip_obs"] = obs_clip

    env = VecNormalize(
        env,
        training=True,
        norm_obs=normalize_observations,
        norm_reward=normalize_reward,
        gamma=ppo_kwargs["gamma"],
        **normalize_kwargs,
    )

    env = VecFrameStack(
        venv=env,
        n_stack=pipeline_config["frame_stack"],
    )

    model = PPO(
        env=env,
        verbose=2,
        tensorboard_log=log_dir + (str(wandb.run.id) if use_wandb else ""),
        **ppo_kwargs,
    )

    callback_list = []

    callback_list.append(AdjustLoggingWindow(window_length=pipeline_config["number_of_envs"]))
    callback_list.append(EpisodeInfoLoggerCallback())
    callback_list.append(SaveBestModelCallback())

    if render:
        callback_list.append(RenderCallback())

    if use_wandb:
        from wandb.integration.sb3 import WandbCallback

        callback_list.append(
            WandbCallback(
                gradient_save_freq=10000,
                model_save_path=f"models/{wandb.run.id}",
                verbose=2,
            )
        )
        model_log_dir = os.path.join(wandb.run.dir, "logs")
    else:
        model_log_dir = log_dir

    if model_checkpoint_distance is not None:
        callback_list.append(
            CheckpointCallback(
                save_freq=max(model_checkpoint_distance // pipeline_config["number_of_envs"], 1),
                save_path=model_log_dir,
                name_prefix="rl_model",
            )
        )

    if extra_callbacks is not None:
        callback_list = callback_list + extra_callbacks

    callback = CallbackList(callback_list)

    return model, callback
