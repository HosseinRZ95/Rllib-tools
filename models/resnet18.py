import torch
from torch import nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret
        # self.addcoords = AddCoordsTh(x_dim=x_dim, y_dim=y_dim, with_r=with_r)


class resblock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)
class ResNet18( TorchModelV2,nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
    # def __init__(self, in_channels, resblock, outputs=11):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                      model_config, name)
        nn.Module.__init__(self)
        with_r=model_config['custom_model_config']["with_r"]
        in_channels=model_config['custom_model_config']["in_channels"]
        resblock=model_config['custom_model_config']["resblock"]
        mlp_size = model_config['custom_model_config']["mlp_size"]        
        self.addcoords = AddCoords(with_r=with_r)
        in_channels = in_channels + 2
        if with_r:
            in_channels += 1
        # print(obs_space)
        # ( h, in_channels) = obs_space.shape
        self.in_channels = in_channels
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels ,64 , kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer0x0 = nn.Sequential(
            nn.Conv2d(1 ,64 , kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True),
            resblock(256, 256, downsample=False)
        )


        self.layer4 = nn.Sequential(
            resblock(256, 512, downsample=True),
            resblock(512, 512, downsample=False)
        )

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.flatt = torch.nn.Flatten()
        self.fc = torch.nn.Linear(512, num_outputs)
        self.fcv = torch.nn.Linear(512, 1)
    def forward(
        self,
        input ,
        state,
        seq_lens):
        # print((input["obs"]))
        # X = torch.unsqueeze(input["obs"].float(), 0)
        # input = X.permute(1,0,2,3)
        # print('input',input)
        print('seq',seq_lens)
        input = input["obs"].float()
        input = self.addcoords(input)
        input = self.layer0(input)
      
        
        # print(type(1))
        # # print(input["obs"].float().size(dim=2))
        # input = torch.unsqueeze(input["obs"].float(), 0)
        # if input["obs"].size(dim=0) == 32:
        #   input = self.layer0(input)
        # else:
        #   input = self.layer0x0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        print(input.size())
        input = self.gap(input)
        input = self.flatt(input)
        # print(input.size())
        # input_last = torch.flatten(input)
        # print(input_last.size())
        output = self.fc(input)
        self.value_f = self.fcv(input)
        # output =torch.reshape(output, [-1])  
        # output = torch.unsqueeze(output,0)
        print(self.value_f.size())
        print('output',output.size())

        return output,state

    def value_function(self):
        assert self.value_f is not None, "must call forward() first"
        print("1stvalue",self.value_f)
        self.value_f = torch.reshape(self.value_f , [-1])
        print('secondvalue',self.value_f)
        return self.value_f      
