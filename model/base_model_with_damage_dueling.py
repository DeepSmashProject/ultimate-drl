
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_torch
import torch.nn.functional as F
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
torch, nn = try_import_torch()

class BaseModel(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("device: ", self.device)
        # (N, C, H, W)
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 32, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(48672, 128)
        self.fc2 = nn.Linear(130, 64)
        self.fcV1 = nn.Linear(64, 32)
        self.fcA1 = nn.Linear(64, 32)
        self.fcV2 = nn.Linear(32, 1)
        self.fcA2 = nn.Linear(32, action_space.n)
        #self.fc3 = nn.Linear(34, action_space.n)


    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"]["observation"]
        damage = input_dict["obs"]["damage"] # torch.Size([32, 2])
        x = x.float().to(self.device)
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = torch.cat([x, damage], dim=1) # torch.Size([32, 130])
        x = self.dropout2(x)
        x = self.fc2(x)

        V = self.fcV2(self.fcV1(x))
        A = self.fcA2(self.fcA1(x))
        averageA = A.mean(1).unsqueeze(1)
        output = V.expand(-1, self.num_actions) + (A - averageA.expand(-1, self.num_actions))
        #output = F.log_softmax(x, dim=1)
        # TODO: dueling net should be written here?
        return output, []

Model = BaseModel