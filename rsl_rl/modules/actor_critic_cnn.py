# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import ipdb
import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal, Categorical, Independent
from typing import Any, Tuple, List

from rsl_rl.networks import CNN, MLP, EmpiricalNormalization

from .actor_critic import ActorCritic


class ActorCriticCNN(ActorCritic):
    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        num_discrete_actions: int = 0,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        actor_cnn_cfg: dict[str, dict] | dict | None = None,
        critic_cnn_cfg: dict[str, dict] | dict | None = None,
        actor_activation: str = "elu",
        actor_last_activation: tuple[str] | None = None,
        critic_activation: str = "elu",
        critic_last_activation: str | None = None,
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCriticCNN.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super(ActorCritic, self).__init__()

        # Store action dimensions
        self.num_continuous_actions = num_actions - num_discrete_actions
        self.num_discrete_actions = num_discrete_actions

        # Get the observation dimensions
        self.obs_groups = obs_groups
        num_actor_obs_1d = 0
        self.actor_obs_groups_1d = []
        actor_in_dims_2d = []
        actor_in_channels_2d = []
        self.actor_obs_groups_2d = []
        for obs_group in obs_groups["policy"]:
            if len(obs[obs_group].shape) == 4:  # B, C, H, W
                self.actor_obs_groups_2d.append(obs_group)
                actor_in_dims_2d.append(obs[obs_group].shape[2:4])
                actor_in_channels_2d.append(obs[obs_group].shape[1])
            elif len(obs[obs_group].shape) == 2:  # B, C
                self.actor_obs_groups_1d.append(obs_group)
                num_actor_obs_1d += obs[obs_group].shape[-1]
            else:
                raise ValueError(f"Invalid observation shape for {obs_group}: {obs[obs_group].shape}")
        num_critic_obs_1d = 0
        self.critic_obs_groups_1d = []
        critic_in_dims_2d = []
        critic_in_channels_2d = []
        self.critic_obs_groups_2d = []
        for obs_group in obs_groups["critic"]:
            if len(obs[obs_group].shape) == 4:  # B, C, H, W
                self.critic_obs_groups_2d.append(obs_group)
                critic_in_dims_2d.append(obs[obs_group].shape[2:4])
                critic_in_channels_2d.append(obs[obs_group].shape[1])
            elif len(obs[obs_group].shape) == 2:  # B, C
                self.critic_obs_groups_1d.append(obs_group)
                num_critic_obs_1d += obs[obs_group].shape[-1]
            else:
                raise ValueError(f"Invalid observation shape for {obs_group}: {obs[obs_group].shape}")

        # Assert that there are 2D observations
        assert self.actor_obs_groups_2d or self.critic_obs_groups_2d, (
            "No 2D observations are provided. If this is intentional, use the ActorCritic module instead."
        )

        # Actor CNN
        if self.actor_obs_groups_2d:
            # Resolve the actor CNN configuration
            assert actor_cnn_cfg is not None, "An actor CNN configuration is required for 2D actor observations."
            # If a single configuration dictionary is provided, create a dictionary for each 2D observation group
            if not all(isinstance(v, dict) for v in actor_cnn_cfg.values()):
                actor_cnn_cfg = {group: actor_cnn_cfg for group in self.actor_obs_groups_2d}
            # Check that the number of configs matches the number of observation groups
            assert len(actor_cnn_cfg) == len(self.actor_obs_groups_2d), (
                "The number of CNN configurations must match the number of 2D actor observations."
            )

            # Create CNNs for each 2D actor observation
            self.actor_cnns = nn.ModuleDict()
            encoding_dim = 0
            for idx, obs_group in enumerate(self.actor_obs_groups_2d):
                self.actor_cnns[obs_group] = CNN(
                    input_dim=actor_in_dims_2d[idx],
                    input_channels=actor_in_channels_2d[idx],
                    **actor_cnn_cfg[obs_group],
                )
                print(f"Actor CNN for {obs_group}: {self.actor_cnns[obs_group]}")
                # Get the output dimension of the CNN
                if self.actor_cnns[obs_group].output_channels is None:
                    encoding_dim += int(self.actor_cnns[obs_group].output_dim)
                else:
                    raise ValueError("The output of the actor CNN must be flattened before passing it to the MLP.")
        else:
            self.actor_cnns = None
            encoding_dim = 0

        # Actor MLP
        self.state_dependent_std = state_dependent_std
        # Actor Continuous MLP
        if self.num_continuous_actions > 0:
            if self.state_dependent_std:
                self.actor_continuous = MLP(num_actor_obs_1d + encoding_dim, [2, self.num_continuous_actions], actor_hidden_dims, actor_activation, actor_last_activation[0])
            else:
                self.actor_continuous = MLP(num_actor_obs_1d + encoding_dim, self.num_continuous_actions, actor_hidden_dims, actor_activation, actor_last_activation[0])
            print(f"Actor Continuous MLP: {self.actor_continuous}")
        else:
            self.actor_continuous = None

        # Actor Discrete MLP
        if self.num_discrete_actions > 0:
            self.actor_discrete = MLP(num_actor_obs_1d + encoding_dim, self.num_discrete_actions, actor_hidden_dims, actor_activation, actor_last_activation[1])
            print(f"Actor Discrete MLP: {self.actor_discrete}")
        else:
            self.actor_discrete = None

        # Actor observation normalization (only for 1D actor observations)
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs_1d)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        # Critic CNN
        if self.critic_obs_groups_2d:
            # Resolve the critic CNN configuration
            assert critic_cnn_cfg is not None, "A critic CNN configuration is required for 2D critic observations."
            # If a single configuration dictionary is provided, create a dictionary for each 2D observation group
            if not all(isinstance(v, dict) for v in critic_cnn_cfg.values()):
                critic_cnn_cfg = {group: critic_cnn_cfg for group in self.critic_obs_groups_2d}
            # Check that the number of configs matches the number of observation groups
            assert len(critic_cnn_cfg) == len(self.critic_obs_groups_2d), (
                "The number of CNN configurations must match the number of 2D critic observations."
            )

            # Create CNNs for each 2D critic observation
            self.critic_cnns = nn.ModuleDict()
            encoding_dim = 0
            for idx, obs_group in enumerate(self.critic_obs_groups_2d):
                self.critic_cnns[obs_group] = CNN(
                    input_dim=critic_in_dims_2d[idx],
                    input_channels=critic_in_channels_2d[idx],
                    **critic_cnn_cfg[obs_group],
                )
                print(f"Critic CNN for {obs_group}: {self.critic_cnns[obs_group]}")
                # Get the output dimension of the CNN
                if self.critic_cnns[obs_group].output_channels is None:
                    encoding_dim += int(self.critic_cnns[obs_group].output_dim)
                else:
                    raise ValueError("The output of the critic CNN must be flattened before passing it to the MLP.")
        else:
            self.critic_cnns = None
            encoding_dim = 0

        # Critic MLP
        self.critic = MLP(num_critic_obs_1d + encoding_dim, 1, critic_hidden_dims, critic_activation, critic_last_activation)
        print(f"Critic MLP: {self.critic}")

        # Critic observation normalization (only for 1D critic observations)
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs_1d)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # Action noise for continuous actions
        self.noise_std_type = noise_std_type
        if self.num_continuous_actions > 0 and self.state_dependent_std:
            torch.nn.init.zeros_(self.actor_continuous[-2].weight[self.num_continuous_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor_continuous[-2].bias[self.num_continuous_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor_continuous[-2].bias[self.num_continuous_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        elif self.num_continuous_actions > 0:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(self.num_continuous_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(self.num_continuous_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distributions
        # Note: Populated in update_distribution
        self.continuous_distribution = None
        self.discrete_distribution = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    def _update_distribution(self, mlp_obs: torch.Tensor, cnn_obs: dict[str, torch.Tensor]) -> None:
        if self.actor_cnns is not None:
            # Encode the 2D actor observations
            cnn_enc_list = [self.actor_cnns[obs_group](cnn_obs[obs_group]) for obs_group in self.actor_obs_groups_2d]
            cnn_enc = torch.cat(cnn_enc_list, dim=-1)
            # Concatenate to the MLP observations
            mlp_obs = torch.cat([mlp_obs, cnn_enc], dim=-1)

        if self.num_continuous_actions > 0:
            if self.state_dependent_std:
                # Compute mean and standard deviation
                mean_and_std = self.actor_continuous(mlp_obs)
                if self.noise_std_type == "scalar":
                    mean, std = torch.unbind(mean_and_std, dim=-2)
                elif self.noise_std_type == "log":
                    mean, log_std = torch.unbind(mean_and_std, dim=-2)
                    std = torch.exp(log_std)
                else:
                    raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
            else:
                # Compute mean
                mean = self.actor_continuous(mlp_obs)
                # Compute standard deviation
                if self.noise_std_type == "scalar":
                    std = self.std.expand_as(mean)
                elif self.noise_std_type == "log":
                    std = torch.exp(self.log_std).expand_as(mean)
                else:
                    raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
            # Create continuous distribution
            self.continuous_distribution = Normal(mean, std)

        if self.num_discrete_actions > 0:
            logits = self.actor_discrete(mlp_obs)
            self.discrete_distribution = Categorical(logits=logits)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        mlp_obs, cnn_obs = self.get_actor_obs(obs)
        mlp_obs = self.actor_obs_normalizer(mlp_obs)
        self._update_distribution(mlp_obs, cnn_obs)
        
        actions = []
        if self.num_continuous_actions > 0:
            continuous_actions = self.continuous_distribution.sample()
            actions.append(continuous_actions)
        if self.num_discrete_actions > 0:
            discrete_probs = self.discrete_distribution.probs
            actions.append(discrete_probs)

        return torch.cat(actions, dim=-1)

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        mlp_obs, cnn_obs = self.get_actor_obs(obs)
        mlp_obs = self.actor_obs_normalizer(mlp_obs)

        if self.actor_cnns is not None:
            # Encode the 2D actor observations
            cnn_enc_list = [self.actor_cnns[obs_group](cnn_obs[obs_group]) for obs_group in self.actor_obs_groups_2d]
            cnn_enc = torch.cat(cnn_enc_list, dim=-1)
            # Concatenate to the MLP observations
            mlp_obs = torch.cat([mlp_obs, cnn_enc], dim=-1)

        actions = []
        if self.num_continuous_actions > 0:
            if self.state_dependent_std:
                continuous_actions = self.actor_continuous(mlp_obs)[..., 0, :]
            else:
                continuous_actions = self.actor_continuous(mlp_obs)
            actions.append(continuous_actions)
        if self.num_discrete_actions > 0:
            logits = self.actor_discrete(mlp_obs)
            discrete_actions = Categorical(logits=logits).probs
            actions.append(discrete_actions)
        
        return torch.cat(actions, dim=-1)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        mlp_obs, cnn_obs = self.get_critic_obs(obs)
        mlp_obs = self.critic_obs_normalizer(mlp_obs)

        if self.critic_cnns is not None:
            # Encode the 2D critic observations
            cnn_enc_list = [self.critic_cnns[obs_group](cnn_obs[obs_group]) for obs_group in self.critic_obs_groups_2d]
            cnn_enc = torch.cat(cnn_enc_list, dim=-1)
            # Concatenate to the MLP observations
            mlp_obs = torch.cat([mlp_obs, cnn_enc], dim=-1)

        return self.critic(mlp_obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        log_probs = []

        continuous_actions = actions[..., :self.num_continuous_actions]
        discrete_actions = actions[..., self.num_continuous_actions:]

        if self.continuous_distribution is not None:
            log_prob_continuous = self.continuous_distribution.log_prob(continuous_actions)
            log_probs.append(log_prob_continuous)
        
        if self.discrete_distribution is not None:
            log_prob_discrete = self.discrete_distribution.log_prob(torch.argmax(discrete_actions, dim=-1)).unsqueeze(-1)
            log_probs.append(log_prob_discrete)
        
        return torch.cat(log_probs, dim=-1).sum(dim=-1)

    @property
    def entropy(self) -> torch.Tensor:
        entropies = []
        
        if self.continuous_distribution is not None:
            entropies.append(self.continuous_distribution.entropy())
        if self.discrete_distribution is not None:
            entropies.append(self.discrete_distribution.entropy().unsqueeze(-1))
        
        return torch.cat(entropies, dim=-1).sum(dim=-1)
    
    @property
    def action_mean(self) -> torch.Tensor:
        actions = []
        if self.num_continuous_actions > 0 and self.continuous_distribution is not None:
            actions.append(self.continuous_distribution.mean)
        if self.num_discrete_actions > 0 and self.discrete_distribution is not None:
            max_indices = self.discrete_distribution.probs
            actions.append(max_indices)
        
        if not actions:
            raise ValueError("No valid action distributions found")
        
        return torch.cat(actions, dim=-1)

    @property
    def action_std(self) -> torch.Tensor:
        if self.num_continuous_actions > 0 and self.continuous_distribution is not None:
            std = self.continuous_distribution.stddev
            if self.num_discrete_actions > 0:
                # 为离散部分添加0方差（确定性）
                zeros = torch.zeros(*std.shape[:-1], self.num_discrete_actions, device=std.device)
                std = torch.cat([std, zeros], dim=-1)
            return std
        return torch.zeros(*self.action_mean.shape, device=self.action_mean.device)

    def get_actor_obs(self, obs: TensorDict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        obs_list_1d = [obs[obs_group] for obs_group in self.actor_obs_groups_1d]
        obs_dict_2d = {}
        for obs_group in self.actor_obs_groups_2d:
            obs_dict_2d[obs_group] = obs[obs_group]
        return torch.cat(obs_list_1d, dim=-1), obs_dict_2d

    def get_critic_obs(self, obs: TensorDict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        obs_list_1d = [obs[obs_group] for obs_group in self.critic_obs_groups_1d]
        obs_dict_2d = {}
        for obs_group in self.critic_obs_groups_2d:
            obs_dict_2d[obs_group] = obs[obs_group]
        return torch.cat(obs_list_1d, dim=-1), obs_dict_2d

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            actor_obs, _ = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs, _ = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)
