import torch
import torch.optim as optim

from MPN_project.pdqn.algo_mpdqn import MultiPassQActor
from MPN_project.pdqn.algo_pdqn_nstep import PDQNNStepAgent
from MPN_project.pdqn.utils import hard_update_target_network



class MultiPassPDQNNStepAgent(PDQNNStepAgent):
    NAME = "Multi-Pass P-DQN N-Step Agent"

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        device = kwargs['device']
        self.actor = MultiPassQActor(self.observation_space.shape[0], self.num_actions, self.action_parameter_sizes,
                                     **kwargs['actor_kwargs']).to(device)
        self.actor_target = MultiPassQActor(self.observation_space.shape[0], self.num_actions, self.action_parameter_sizes,
                                            **kwargs['actor_kwargs']).to(device)
        hard_update_target_network(self.actor, self.actor_target)
        self.actor_target.eval()
        self.actor_optimiser = optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor)
