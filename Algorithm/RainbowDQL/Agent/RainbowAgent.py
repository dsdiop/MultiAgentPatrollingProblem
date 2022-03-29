from typing import Dict, List, Tuple
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from ReplayBuffers import PrioritizedReplayBuffer, ReplayBuffer
from Networks.network import VisualNetwork, NoisyVisualNetwork


class DQNAgent:
	"""DQN Agent interacting with environment.

	Attribute:
		env (gym.Env): openAI Gym environment
		memory (PrioritizedReplayBuffer): replay memory to store transitions
		batch_size (int): batch size for sampling
		target_update (int): period for target model's hard update
		gamma (float): discount factor
		dqn (Network): model to train and select actions
		dqn_target (Network): target model to update
		optimizer (torch.optim): optimizer for training dqn
		transition (list): transition information including state, action, reward, next_state, done
		v_min (float): min value of support
		v_max (float): max value of support
		atom_size (int): the unit number of support
		support (torch.Tensor): support for categorical dqn
		use_n_step (bool): whether to use n_step memory
		n_step (int): step number to calculate n-step td error
		memory_n (ReplayBuffer): n-step replay buffer
	"""

	def __init__(
			self,
			env: gym.Env,
			memory_size: int,
			batch_size: int,
			target_update: int,
			gamma: float = 0.99,
			lr: float = 1e-4,
			noisy_layers: bool = True,
			# PER parameters
			alpha: float = 0.2,
			beta: float = 0.6,
			prior_eps: float = 1e-6,
			# Categorical DQN parameters
			v_min: float = 0.0,
			v_max: float = 200.0,
			atom_size: int = 51,
			# N-step Learning
			n_step: int = 3,
			logdir=None,
			log_name="Experiment"
	):
		"""Initialization.

		Args:
			env (gym.Env): openAI Gym environment
			memory_size (int): length of memory
			batch_size (int): batch size for sampling
			target_update (int): period for target model's hard update
			lr (float): learning rate
			gamma (float): discount factor
			alpha (float): determines how much prioritization is used
			beta (float): determines how much importance sampling is used
			prior_eps (float): guarantees every transition can be sampled
			v_min (float): min value of support
			v_max (float): max value of support
			atom_size (int): the unit number of support
			n_step (int): step number to calculate n-step td error
		"""

		""" Logging parameters """
		self.logdir = logdir
		self.experiment_name = log_name
		self.writer = None

		""" Observation space dimensions """
		obs_dim = env.observation_space.shape
		action_dim = env.action_space.n

		""" Agent embeds the environment """
		self.env = env
		self.batch_size = batch_size
		self.target_update = target_update
		self.gamma = gamma
		self.learning_rate = lr
		self.noisy_layers = noisy_layers
		self.epsilon = 1.0

		""" Automatic selection of the device """
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		print("Selected device: ", self.device)

		""" Prioritized Experience Replay """
		self.beta = beta
		self.prior_eps = prior_eps
		self.memory = PrioritizedReplayBuffer(obs_dim, memory_size, batch_size, alpha=alpha)

		""" Prioritized Experience Replay for n-step estimation. In this implementation, both methods (1step and n-step
		is used to lower the variance caused by the n-step updated. """

		self.use_n_step = True if n_step > 1 else False
		if self.use_n_step:
			self.n_step = n_step
			self.memory_n = ReplayBuffer(obs_dim, memory_size, batch_size, n_step=n_step, gamma=gamma)

		""" Distributional DQN parameters """
		self.v_min = v_min  # Max-Min values for the accumulated reward
		self.v_max = v_max
		self.atom_size = atom_size  # Number of quantiles
		self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(self.device)  # Tensor of quantiles

		""" Create the DQN and the DQN-Target """
		number_of_features = 256
		if self.noisy_layers:
			self.dqn = NoisyVisualNetwork(obs_dim, action_dim, self.atom_size, 256, self.support).to(self.device)
			self.dqn_target = NoisyVisualNetwork(obs_dim, action_dim, self.atom_size, 256, self.support).to(self.device)
		else:
			self.dqn = VisualNetwork(obs_dim, action_dim, self.atom_size, 256, self.support).to(self.device)
			self.dqn_target = VisualNetwork(obs_dim, action_dim, self.atom_size, 256, self.support).to(self.device)

		self.dqn_target.load_state_dict(self.dqn.state_dict())
		self.dqn_target.eval()

		""" Optimizer """
		self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.learning_rate)

		""" Actual list of transitions """
		self.transition = list()

		""" Evaluation flag """
		self.is_eval = False

		""" Data for logging """
		self.episodic_reward = []
		self.episodic_loss = []
		self.episodic_length = []
		self.episode = 0
		# TODO: Implement an annealed Learning Rate (see:
		#  https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau)


	def select_action(self, state: np.ndarray) -> np.ndarray:
		"""Select an action from the input state. If deterministic, no noise is applied. """

		# TODO: Implement DETERMINISTIC action selection. Noise should be removed when evaluating.

		if self.noisy_layers:
			selected_action = self.dqn(torch.FloatTensor(state).unsqueeze(0).to(self.device)).argmax()
			selected_action = selected_action.detach().cpu().numpy()
		else:
			# Epsilon greedy policy
			if self.epsilon > np.random.rand():
				selected_action = self.env.action_space.sample()
			else:
				selected_action = self.dqn(torch.FloatTensor(state).unsqueeze(0).to(self.device)).argmax()
				selected_action = selected_action.detach().cpu().numpy()

		if not self.is_eval:
			self.transition = [state, selected_action]

		return selected_action

	def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
		"""Take an action and return the response of the env."""

		""" Perform a step """
		next_state, reward, done, _ = self.env.step(action)

		if not self.is_eval:

			""" Store transition """
			self.transition += [reward, next_state, done]

			if self.use_n_step:
				""" N-step transition """
				one_step_transition = self.memory_n.store(*self.transition)

			else:
				""" 1-step transition """
				one_step_transition = self.transition

			""" Store the transition into the Exp. Replay Buffer """
			if one_step_transition:
				self.memory.store(*one_step_transition)

		return next_state, reward, done

	def update_model(self) -> torch.Tensor:
		""" Update the model by taking a gradient descent step. """

		# PER needs beta to calculate weights
		samples = self.memory.sample_batch(self.beta)
		weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
		indices = samples["indices"]

		# 1-step Learning loss
		elementwise_loss = self._compute_dqn_loss(samples, self.gamma)

		# PER: importance sampling before average
		loss = torch.mean(elementwise_loss * weights)

		# N-step Learning loss
		# we are gonna combine 1-step loss and n-step loss so as to
		# prevent high-variance. The original rainbow employs n-step loss only.
		if self.use_n_step:

			gamma = self.gamma ** self.n_step
			samples = self.memory_n.sample_batch_from_idxs(indices)
			elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
			elementwise_loss += elementwise_loss_n_loss

			# PER: importance sampling before average
			loss = torch.mean(elementwise_loss * weights)

		self.optimizer.zero_grad()
		loss.backward()
		clip_grad_norm_(self.dqn.parameters(), 10.0)
		self.optimizer.step()

		# PER: update priorities
		loss_for_prior = elementwise_loss.detach().cpu().numpy()
		new_priorities = loss_for_prior + self.prior_eps
		self.memory.update_priorities(indices, new_priorities)

		# NoisyNet: reset noise
		if self.noisy_layers:
			self.dqn.reset_noise()
			self.dqn_target.reset_noise()

		return loss.item()

	@staticmethod
	def anneal_epsilon(p, p_init = 0.1, p_fin = 0.9, e_min=0.1):

		if p < p_init:
			return 1.0
		elif p > p_fin:
			return e_min
		else:
			return (e_min - 1)/(p_fin-p_init) * (p - p_init) + 1.0

	@staticmethod
	def anneal_beta(p, p_init=0.1, p_fin=0.9, b_init=0.4, b_end=1.0):

		if p < p_init:
			return b_init
		elif p > p_fin:
			return b_end
		else:
			return (b_end - b_init) / (p_fin - p_init) * (p - p_init) + b_init

	def train(self, episodes):
		""" Train the agent. """

		# Create train logger #
		if self.writer is None:
			self.writer = SummaryWriter(log_dir=self.logdir, filename_suffix=self.experiment_name)

		# Agent in training mode #
		self.is_eval = False
		# Reset episode count #
		self.episode = 1
		# Reset metrics #
		episodic_reward_vector = []
		record = -np.inf
		self.epsilon = 1.0

		for episode in range(1, int(episodes) + 1):

			done = False
			state = self.env.reset()
			score = 0
			length = 0
			losses = []

			# PER: Increase beta temperature
			self.beta = self.anneal_beta(p=episode/episodes, p_init=0, p_fin=0.9, b_init=0.4, b_end=1.0)

			# Epsilon greedy annealing
			if not self.noisy_layers:
				self.epsilon = self.anneal_epsilon(p=episode/episodes, p_init=0.1, p_fin=0.5, e_min=0.1)

			while not done:

				action = self.select_action(state)
				next_state, reward, done = self.step(action)

				state = next_state
				score += reward
				length += 1

				# if episode ends
				if done:

					# Append loss metric #
					if losses:
						self.episodic_loss = np.mean(losses)

					# Compute average metrics #
					self.episodic_reward = score
					self.episodic_length = length
					episodic_reward_vector.append(self.episodic_reward)
					self.episode += 1

					# Log progress
					self.log_data()

					# Save policy if is better on average
					mean_episodic_reward = np.mean(episodic_reward_vector[-50:])
					if mean_episodic_reward > record:
						print(f"New best policy with mean reward of {mean_episodic_reward}")
						print("Saving model in " + self.writer.log_dir)
						record = mean_episodic_reward
						self.save_model()

				# if training is ready
				if len(self.memory) >= self.batch_size:

					loss = self.update_model()
					losses.append(loss)

					# if hard update is needed
					if episode % self.target_update == 0 and done:
						self._target_hard_update()

	def evaluate_policy(self, episodes=1, render = False):
		"""Evaluate the current policy."""

		self.is_eval = True

		scores = []

		for e in range(episodes):

			state = self.env.reset()

			if render:
				self.env.render()

			done = False
			score = 0

			while not done:

				action = self.select_action(state)
				next_state, reward, done = self.step(action)
				state = next_state
				score += reward

				if render:
					self.env.render()

			print(f"Episode {e} total score: {score}")
			scores.append(score)

		print(f"Mean Reward: {np.mean(scores)} +- {np.std(scores)}")

		self.is_eval = False

	def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
		"""Return categorical dqn loss."""
		device = self.device  # for shortening the following lines
		state = torch.FloatTensor(samples["obs"]).to(device)
		next_state = torch.FloatTensor(samples["next_obs"]).to(device)
		action = torch.LongTensor(samples["acts"]).to(device)
		reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
		done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

		# Categorical DQN algorithm
		delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

		with torch.no_grad():
			# Double DQN
			next_action = self.dqn(next_state).argmax(1)
			next_dist = self.dqn_target.dist(next_state)
			next_dist = next_dist[range(self.batch_size), next_action]

			t_z = reward + (1 - done) * gamma * self.support
			t_z = t_z.clamp(min=self.v_min, max=self.v_max)
			b = (t_z - self.v_min) / delta_z
			l = b.floor().long()
			u = b.ceil().long()

			offset = (
				torch.linspace(0, (self.batch_size - 1) * self.atom_size, self.batch_size).long().unsqueeze(1).expand(self.batch_size, self.atom_size).to(self.device)
			)

			proj_dist = torch.zeros(next_dist.size(), device=self.device)
			proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
			proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

		dist = self.dqn.dist(state)
		log_p = torch.log(dist[range(self.batch_size), action])
		elementwise_loss = -(proj_dist * log_p).sum(1)

		return elementwise_loss

	def _target_hard_update(self):
		"""Hard update: target <- local."""
		print(f"Hard update performed at episode {self.episode}!")
		self.dqn_target.load_state_dict(self.dqn.state_dict())

	def log_data(self):

		if self.episodic_loss:
			self.writer.add_scalar('train/loss', self.episodic_loss, self.episode)

		if not self.noisy_layers:
			self.writer.add_scalar('train/epsilon', self.epsilon, self.episode)
		else:
			self.writer.add_scalar('train/beta', self.beta, self.episode)

		self.writer.add_scalar('train/accumulated_reward', self.episodic_reward, self.episode)
		self.writer.add_scalar('train/accumulated_length', self.episodic_length, self.episode)

		self.writer.flush()

	def load_model(self, path_to_file):

		self.dqn.load_state_dict(torch.load(path_to_file, map_location=self.device))

	def save_model(self):

		torch.save(self.dqn.state_dict(), self.writer.log_dir + '/experiment.pth')

