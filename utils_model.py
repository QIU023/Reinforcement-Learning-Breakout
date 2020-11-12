import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, action_dim, device):
        super(DQN, self).__init__()
        self.__conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False)
        self.__conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.__conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        self.__fc1 = nn.Linear(64*7*7, 512)
        self.__fc2 = nn.Linear(512, action_dim)
        self.__device = device

    def DQN_Conv_forward(self, x):
        x = x / 255.
        x = F.relu(self.__conv1(x))
        x = F.relu(self.__conv2(x))
        x = F.relu(self.__conv3(x))
        x = F.relu(self.__fc1(x.view(x.size(0), -1)))
        return x
    
    def forward(self, x):
        x = x / 255.
        x = F.relu(self.__conv1(x))
        x = F.relu(self.__conv2(x))
        x = F.relu(self.__conv3(x))
        x = F.relu(self.__fc1(x.view(x.size(0), -1)))
        return self.__fc2(x)

    def Convs_load(self, weights):
        self.load_state_dict(torch.load(weights))
        torch.nn.init.kaiming_normal_(self.__fc2.weight, nonlinearity="relu")
        #self.__fc2.bias.data.fill_(0.0)
    
    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")

class DuelingDQN(nn.Module):

    def __init__(self, action_dim, device):
        super(DuelingDQN, self).__init__()
        self.DQN = DQN(action_dim, device)
        self.__fc_vstate = nn.Linear(512, 1)
        self.__fc_adv_action = nn.Linear(512, action_dim)
        self.__device = device
        self.action_dim = action_dim

    def Convs_load(self, weights):
        self.DQN.load_state_dict(torch.load(weights))
        torch.nn.init.kaiming_normal_(self.__fc_vstate.weight, nonlinearity="relu")
        torch.nn.init.kaiming_normal_(self.__fc_adv_action.weight, nonlinearity="relu")
        #torch.full(self.__fc_vstate.bias, 0.0)
        #torch.full(self.__adv_action.bias, 0.0)
        
    def forward(self, x):
        x = self.DQN.DQN_Conv_forward(x)
        value = F.relu(self.__fc_vstate(x))
        adv_action = F.relu(self.__fc_adv_action(x))
        #import ipdb;ipdb.set_trace()
        j = torch.mean(adv_action, dim=1).unsqueeze(1)
        #print(value.shape, adv_action.shape, j.shape)
        return (value-j).repeat(1,self.action_dim)+adv_action