import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mve import REGISTRY as mve_REGISTRY
import torch as th
import torch.nn.functional as F
from torch.optim import Adam
from components.standarize_stream import RunningMeanStd
import os

class MASIALearner:
    def __init__(self, mac, latent_model, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.latent_model = latent_model
        self.logger = logger

        if not self.args.rl_signal:
            assert 0, "Must use rl signal in this method !!!"
            self.params = list(mac.rl_parameters())
        else:
            self.params = list(mac.parameters())
        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        if self.args.use_latent_model:
            # use_latent_model means use_spr
            self.params += list(latent_model.parameters())

        self.optimiser = Adam(params=self.params, lr=args.lr)
                
        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.training_steps = 0
        self.last_target_update_step = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        # Three-phase training tracking
        self.current_phase = getattr(args, 'training_phase', 1)
        self.phase_start_step = 0
        self.phase_start_t_env = 0  # Track t_env at phase start
        self.q_base_saved = False

        # Phase 2: Message Value Estimator (MVE)
        self.mve = None
        self.mve_optimiser = None
        self.frozen_mac = None  # Frozen Q_base for MVE training
        self.frozen_mixer = None

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=device)

    def repr_train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # states.shape: [batch_size, seq_len, state_dim]
        states = batch["state"]
        # actions.shape: [batch_size, seq_len, n_agents, 1]
        actions_onehot = batch["actions_onehot"]
        rewards = batch["reward"]
        terminated = batch["terminated"].float()
        mask = batch["filled"].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        # go through vae
        recons, z = [], []
        self.mac.init_hidden(batch.batch_size)  # useless in current version
        for t in range(batch.max_seq_length):
            recons_t, _, z_t = self.mac.vae_forward(batch, t)
            recons.append(recons_t)
            z.append(z_t)
        # recons.shape: [batch_size, seq_len, state_repre_dim]
        recons = th.stack(recons, dim=1)  # Concat over time
        z = th.stack(z, dim=1)

        bs, seq_len  = states.shape[0], states.shape[1]
        loss_dict = self.mac.agent.encoder.loss_function(recons.reshape(bs*seq_len, -1), states.reshape(bs*seq_len, -1))
        vae_loss = loss_dict["loss"].reshape(bs, seq_len, 1)
        mask = mask.expand_as(vae_loss)
        masked_vae_loss = (vae_loss * mask).sum() / mask.sum()

        if self.args.use_latent_model:
            # Compute target z first
            target_projected = []
            with th.no_grad():
                self.mac.init_hidden(batch.batch_size)
                for t in range(batch.max_seq_length):
                    target_projected_t = self.mac.target_transform(batch, t)
                    target_projected.append(target_projected_t)
            target_projected = th.stack(target_projected, dim=1)  # Concat over time, shape: [bs, seq_len, spr_dim]

            curr_z = z
            # Do final vector prediction
            predicted_f = self.mac.agent.online_projection(curr_z)   # [bs, seq_len, spr_dim]
            tot_spr_loss = self.compute_spr_loss(predicted_f, target_projected, mask)
            if self.args.use_rew_pred:
                predicted_rew = self.latent_model.predict_reward(curr_z)   # [bs, seq_len, 1]
                tot_rew_loss = self.compute_rew_loss(predicted_rew, rewards, mask)
            for t in range(self.args.pred_len):
                # do transition model forward
                curr_z = self.latent_model(curr_z, actions_onehot[:, t:])[:, :-1]
                # Do final vector prediction
                predicted_f = self.mac.agent.online_projection(curr_z)  # [bs, seq_len, spr_dim]
                tot_spr_loss += self.compute_spr_loss(predicted_f, target_projected[:, t+1:], mask[:, t+1:])
                if self.args.use_rew_pred:
                    predicted_rew = self.latent_model.predict_reward(curr_z)
                    tot_rew_loss += self.compute_rew_loss(predicted_rew, rewards[:, t+1:], mask[:, t+1:])
            
            if self.args.use_rew_pred:
                repr_loss = masked_vae_loss + self.args.spr_coef * tot_spr_loss + self.args.rew_pred_coef * tot_rew_loss
            else:
                repr_loss = masked_vae_loss + self.args.spr_coef * tot_spr_loss
        else:
            repr_loss = masked_vae_loss

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("repr_loss", repr_loss.item(), t_env)
            self.logger.log_stat("vae_loss", masked_vae_loss.item(), t_env)
            if self.args.use_latent_model:
                self.logger.log_stat("model_loss", tot_spr_loss.item(), t_env)
                if self.args.use_rew_pred:
                    self.logger.log_stat("rew_pred_loss", tot_rew_loss.item(), t_env)

        return repr_loss
    
    def compute_rew_loss(self, pred_rew, env_rew, mask):
        # pred_rew.shape: [bs, seq_len, 1]
        # mask.shape: [bs, seq_len, 1]
        mask = mask.squeeze(-1)
        rew_loss = F.mse_loss(pred_rew, env_rew, reduction="none").sum(-1)
        masked_rew_loss = (rew_loss * mask).sum() / mask.sum()
        return masked_rew_loss

    def compute_spr_loss(self, pred_f, target_f, mask):
        # pred_f.shape: [bs, seq_len, spr_dim]
        # mask.shape: [bs, seq_len, 1]
        mask = mask.squeeze(-1)
        spr_loss = F.mse_loss(pred_f, target_f, reduction="none").sum(-1)
        mask_spr_loss = (spr_loss * mask).sum() / mask.sum()
        return mask_spr_loss

    def rl_train(self, batch: EpisodeBatch, t_env: int, episode_num: int, repr_loss):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            state_repr_t = self.mac.enc_forward(batch, t=t)
            if not self.args.rl_signal:
                state_repr_t = state_repr_t.detach()
            agent_outs = self.mac.rl_forward(batch, state_repr_t, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            state_repr_t = self.target_mac.enc_forward(batch, t=t)
            target_agent_outs = self.target_mac.rl_forward(batch, state_repr_t, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        if self.args.standardise_returns:
            target_max_qvals = target_max_qvals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals.detach()

        if self.args.standardise_returns:
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        rl_loss = (masked_td_error ** 2).sum() / mask.sum() 
        # Compute tot loss
        tot_loss = rl_loss + self.args.repr_coef * repr_loss

        # Optimise
        self.optimiser.zero_grad()
        tot_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.training_steps += 1
        if self.args.target_update_interval_or_tau > 1 and (self.training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.mac.agent.momentum_update()
            self.last_target_update_step = self.training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)
            self.mac.agent.momentum_update()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("rl_loss", rl_loss.item(), t_env)
            self.logger.log_stat("tot_loss", tot_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)

            # Three-phase training logging
            self.logger.log_stat("training_phase", self.current_phase, t_env)
            self.logger.log_stat("phase_t_env", t_env - self.phase_start_t_env, t_env)
            self.logger.log_stat("phase_steps", self.training_steps - self.phase_start_step, t_env)
            if hasattr(self.args, 'message_dropout_rate'):
                self.logger.log_stat("message_dropout_rate", self.args.message_dropout_rate, t_env)

            self.log_stats_t = t_env

    def mve_train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        """
        Phase 2: Train Context-Aware Message Value Estimator (MVE).

        For each agent i, compute:
            y_i = Q_context - Q_neg_i

        where:
            Q_context = Q_base(s, u, m)  # All agents' messages present
            Q_neg_i = Q_base(s, u, m_{-i})  # Agent i silenced BEFORE obs_embed→encoder

        Then train MVE to predict these values: v_hat = V_phi(messages)

        Note: Messages are the embedded observations (output of obs_embed), not encoded states.
        """
        if self.mve is None or self.frozen_mac is None:
            # MVE not initialized yet
            return

        # Get relevant data from batch
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        bs = batch.batch_size
        max_seq_length = batch.max_seq_length
        n_agents = self.args.n_agents

        # Step 1: Extract messages (embedded observations) from all agents
        messages_list = []
        for t in range(max_seq_length - 1):
            messages_t = self.frozen_mac.get_messages(batch, t=t, silence_mask=None)
            messages_list.append(messages_t)

        # Stack over time: [bs, seq_len-1, n_agents, input_shape]
        messages = th.stack(messages_list, dim=1)
        message_dim = messages.shape[-1]

        # Step 2: Compute Q_context (Q-values with all agents communicating)
        with th.no_grad():
            mac_out_context = []
            self.frozen_mac.init_hidden(bs)
            for t in range(max_seq_length - 1):
                # Full forward pass (no silencing)
                state_repr_t = self.frozen_mac.enc_forward(batch, t=t, silence_mask=None)
                agent_outs = self.frozen_mac.rl_forward(batch, state_repr_t, t=t)
                mac_out_context.append(agent_outs)

            mac_out_context = th.stack(mac_out_context, dim=1)  # [bs, seq_len-1, n_agents, n_actions]
            chosen_qvals_context = th.gather(mac_out_context, dim=3, index=actions).squeeze(3)  # [bs, seq_len-1, n_agents]

            # Mix to get joint Q-value
            if self.frozen_mixer is not None:
                q_context = self.frozen_mixer(chosen_qvals_context, batch["state"][:, :-1]).squeeze(-1)
            else:
                q_context = chosen_qvals_context.sum(dim=2)

        # Step 3: For each agent i, compute Q_neg_i (agent i silenced BEFORE obs_embed→encoder)
        labels = th.zeros(bs, max_seq_length - 1, n_agents, device=messages.device)

        for i in range(n_agents):
            with th.no_grad():
                # Create silence mask: [bs, n_agents] where 1=silence, 0=keep
                silence_mask = th.zeros(bs, n_agents, device=messages.device)
                silence_mask[:, i] = 1.0  # Silence agent i

                # Forward pass with agent i silenced
                mac_out_neg_i = []
                self.frozen_mac.init_hidden(bs)
                for t in range(max_seq_length - 1):
                    # Encode with agent i silenced (happens in obs_embed → encoder)
                    state_repr_t = self.frozen_mac.enc_forward(batch, t=t, silence_mask=silence_mask)
                    agent_outs = self.frozen_mac.rl_forward(batch, state_repr_t, t=t)
                    mac_out_neg_i.append(agent_outs)

                mac_out_neg_i = th.stack(mac_out_neg_i, dim=1)
                chosen_qvals_neg_i = th.gather(mac_out_neg_i, dim=3, index=actions).squeeze(3)

                # Mix
                if self.frozen_mixer is not None:
                    q_neg_i = self.frozen_mixer(chosen_qvals_neg_i, batch["state"][:, :-1]).squeeze(-1)
                else:
                    q_neg_i = chosen_qvals_neg_i.sum(dim=2)

                # Label: y_i = Q_context - Q_neg_i (NO ReLU)
                # Positive = helpful message, Negative = harmful message
                labels[:, :, i] = (q_context - q_neg_i).detach()

        # Step 4: Train MVE to predict message values
        messages_train = messages.view(-1, n_agents, message_dim)  # [bs*(seq_len-1), n_agents, message_dim]
        labels_train = labels.view(-1, n_agents)  # [bs*(seq_len-1), n_agents]
        mask_train = mask.view(-1, 1).expand(-1, n_agents)  # [bs*(seq_len-1), n_agents]

        # Predict using MVE
        predicted_values = self.mve(messages_train)  # [bs*(seq_len-1), n_agents]

        # Compute loss
        mve_loss = self.mve.compute_loss(predicted_values, labels_train, agent_mask=mask_train)

        # Optimize
        self.mve_optimiser.zero_grad()
        mve_loss.backward()
        mve_grad_norm = th.nn.utils.clip_grad_norm_(self.mve.parameters(), self.args.grad_norm_clip)
        self.mve_optimiser.step()

        # Logging
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("mve_loss", mve_loss.item(), t_env)
            self.logger.log_stat("mve_grad_norm", mve_grad_norm.item(), t_env)

            # Log statistics
            if mask_train.sum() > 0:
                avg_predicted = (predicted_values * mask_train).sum() / mask_train.sum()
                avg_label = (labels_train * mask_train).sum() / mask_train.sum()
                self.logger.log_stat("mve_pred_mean", avg_predicted.item(), t_env)
                self.logger.log_stat("mve_label_mean", avg_label.item(), t_env)

                # Per-agent statistics (first 4 agents to avoid clutter)
                for i in range(min(n_agents, 4)):
                    agent_mask = mask_train[:, i] > 0
                    if agent_mask.sum() > 0:
                        avg_value_i = labels_train[agent_mask, i].mean()
                        self.logger.log_stat(f"mve_value_agent_{i}", avg_value_i.item(), t_env)

            self.log_stats_t = t_env

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Check for phase transitions (automatic three-phase training)
        self._check_phase_transition(t_env)

        # Phase-specific training
        if self.current_phase == 1:
            # Phase 1: Robust Pre-training with message dropout
            repr_loss = self.repr_train(batch, t_env, episode_num)
            self.rl_train(batch, t_env, episode_num, repr_loss)

        elif self.current_phase == 2:
            # Phase 2: Context-Aware MVE Training
            # Q_base is frozen - only train MVE to predict message values
            self.mve_train(batch, t_env, episode_num)

        elif self.current_phase == 3:
            # Phase 3: Value-Conditional Unlearning (TODO: implement)
            # Resume Q-network training with unlearning objective
            repr_loss = self.repr_train(batch, t_env, episode_num)
            self.rl_train(batch, t_env, episode_num, repr_loss)
            # TODO: Add unlearning logic here

        else:
            # Default: same as Phase 1
            repr_loss = self.repr_train(batch, t_env, episode_num)
            self.rl_train(batch, t_env, episode_num, repr_loss)

    def _check_phase_transition(self, t_env):
        """Check if we should transition to the next phase based on environment timesteps."""
        t_env_in_phase = t_env - self.phase_start_t_env

        # Phase 1 -> Phase 2 transition
        if self.current_phase == 1 and t_env_in_phase >= getattr(self.args, 'phase1_steps', float('inf')):
            self._transition_to_phase2(t_env)

        # Phase 2 -> Phase 3 transition
        elif self.current_phase == 2 and t_env_in_phase >= getattr(self.args, 'phase2_steps', float('inf')):
            self._transition_to_phase3(t_env)

    def _transition_to_phase2(self, t_env):
        """Transition from Phase 1 (Robust Pre-training) to Phase 2 (Context-Aware MVE)."""
        self.logger.console_logger.info(f"\n{'='*60}")
        self.logger.console_logger.info(f"PHASE TRANSITION: Phase 1 -> Phase 2 at t_env {t_env} (training_steps {self.training_steps})")
        self.logger.console_logger.info(f"{'='*60}\n")

        # Save Q_base (frozen Q-network for Phase 2 use)
        if getattr(self.args, 'save_phase1_qbase', True):
            self._save_q_base()

        # Create frozen copies of MAC and mixer for MVE training
        # Avoid deepcopy issues with non-leaf tensors by manually creating new instances
        # Safe copy approach: create new MAC and load state_dict
        # First, get a clean state dict with all tensors detached
        mac_state = {k: v.clone().detach() for k, v in self.mac.state_dict().items()}

        # Create new MAC instance (same class as self.mac)
        self.frozen_mac = type(self.mac)(self.mac.scheme, self.mac.groups, self.mac.args)
        self.frozen_mac.load_state_dict(mac_state)
        # Move frozen_mac to correct device
        if self.args.use_cuda:
            self.frozen_mac.cuda()

        if self.mixer is not None:
            # Same approach for mixer
            mixer_state = {k: v.clone().detach() for k, v in self.mixer.state_dict().items()}
            # VDNMixer takes no args, QMixer takes args
            if self.args.mixer == "vdn":
                self.frozen_mixer = type(self.mixer)()
            else:  # qmix or others
                self.frozen_mixer = type(self.mixer)(self.args)
            self.frozen_mixer.load_state_dict(mixer_state)
            # Move frozen_mixer to correct device
            if self.args.use_cuda:
                self.frozen_mixer.cuda()

        # Freeze parameters (no gradients)
        for param in self.frozen_mac.parameters():
            param.requires_grad = False
        if self.frozen_mixer is not None:
            for param in self.frozen_mixer.parameters():
                param.requires_grad = False

        # Initialize Message Value Estimator (MVE)
        # Get message dimension from agent's input shape
        message_dim = self.mac.agent.obs_embed.in_features

        self.mve = mve_REGISTRY["attention"](
            message_dim=message_dim,
            n_agents=self.args.n_agents,
            hidden_dim=getattr(self.args, 'mve_hidden_dim', 64),
            num_heads=getattr(self.args, 'mve_num_heads', 4),
            dropout=0.1
        )

        # Move MVE to device
        if self.args.use_cuda:
            self.mve.cuda()

        # Create separate optimizer for MVE
        mve_lr = getattr(self.args, 'mve_lr', 0.0005)
        self.mve_optimiser = Adam(params=self.mve.parameters(), lr=mve_lr)

        self.logger.console_logger.info(f"Initialized MVE with message_dim={message_dim}, hidden_dim={getattr(self.args, 'mve_hidden_dim', 64)}")
        self.logger.console_logger.info(f"MVE optimizer lr={mve_lr}")

        # Update phase tracking
        self.current_phase = 2
        self.phase_start_step = self.training_steps
        self.phase_start_t_env = t_env

        # Disable message dropout for Phase 2
        self.args.message_dropout_rate = 0.0

        # Log transition
        self.logger.log_stat("training_phase", self.current_phase, t_env)

    def _transition_to_phase3(self, t_env):
        """Transition from Phase 2 (Context-Aware MVE) to Phase 3 (Value-Conditional Unlearning)."""
        self.logger.console_logger.info(f"\n{'='*60}")
        self.logger.console_logger.info(f"PHASE TRANSITION: Phase 2 -> Phase 3 at t_env {t_env} (training_steps {self.training_steps})")
        self.logger.console_logger.info(f"{'='*60}\n")

        # Update phase tracking
        self.current_phase = 3
        self.phase_start_step = self.training_steps
        self.phase_start_t_env = t_env

        # Log transition
        self.logger.log_stat("training_phase", self.current_phase, t_env)

    def _save_q_base(self):
        """Save frozen Q-network (Q_base) at the end of Phase 1 for use in Phase 2."""
        # Create directory for Q_base
        save_path = getattr(self.args, 'local_results_path', './results')
        q_base_path = os.path.join(save_path, "phase1_qbase")
        os.makedirs(q_base_path, exist_ok=True)

        # Save agent (contains Q-network: gate, fc1, rnn, fc2) and encoder
        th.save(self.mac.agent.state_dict(), f"{q_base_path}/agent.th")

        # Save mixer
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), f"{q_base_path}/mixer.th")

        # Save optimizer state for potential resumption
        th.save(self.optimiser.state_dict(), f"{q_base_path}/opt.th")

        self.logger.console_logger.info(f"Saved Q_base to: {q_base_path}")
        self.q_base_saved = True

    def test_encoder(self, batch: EpisodeBatch):
        # states.shape: [batch_size, seq_len, state_dim]
        states = batch["state"]
        terminated = batch["terminated"].float()
        mask = batch["filled"].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        # go through vae
        recons, z = [], []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            recons_t, _, z_t = self.mac.vae_forward(batch, t)
            recons.append(recons_t)
            z.append(z_t)
        # recons.shape: [batch_size, seq_len, state_repre_dim]
        recons = th.stack(recons, dim=1)
        z = th.stack(z, dim=1)

        encoder_result = {
            "recons": recons,
            "z": z,
            "states": states,
            "mask": mask,
        }
        th.save(encoder_result, os.path.join(self.args.encoder_result_direc, "result.pth"))

    def _update_targets_hard(self):
        # not quite good, but don't have bad effect
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _update_targets_soft(self, tau):
        # not quite good, but don't have bad effect
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        if self.mixer is not None:
            for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.latent_model.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
        # Phase 2 & 3: MVE and frozen models
        if self.mve is not None:
            self.mve.cuda()
        if self.frozen_mac is not None:
            self.frozen_mac.cuda()
        if self.frozen_mixer is not None:
            self.frozen_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.latent_model.state_dict(), "{}/latent_model.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

        # Save phase-specific models
        th.save({
            'current_phase': self.current_phase,
            'phase_start_step': self.phase_start_step,
            'phase_start_t_env': self.phase_start_t_env
        }, "{}/phase_info.th".format(path))

        # Phase 2 & 3: Save MVE
        if self.mve is not None:
            th.save(self.mve.state_dict(), "{}/mve.th".format(path))
            th.save(self.mve_optimiser.state_dict(), "{}/mve_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        self.latent_model.load_state_dict(th.load("{}/latent_model.th".format(path), map_location=lambda storage, loc: storage))

        # Load phase info if available
        import os.path as osp
        phase_info_path = "{}/phase_info.th".format(path)
        if osp.exists(phase_info_path):
            phase_info = th.load(phase_info_path, map_location=lambda storage, loc: storage)
            self.current_phase = phase_info['current_phase']
            self.phase_start_step = phase_info['phase_start_step']
            self.phase_start_t_env = phase_info.get('phase_start_t_env', 0)

        # Load MVE if in Phase 2 or 3
        mve_path = "{}/mve.th".format(path)
        if osp.exists(mve_path) and self.mve is not None:
            self.mve.load_state_dict(th.load(mve_path, map_location=lambda storage, loc: storage))
            self.mve_optimiser.load_state_dict(th.load("{}/mve_opt.th".format(path), map_location=lambda storage, loc: storage))