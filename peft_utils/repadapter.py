import torch
import torch.nn as nn
import torch.nn.functional as F

class RepAdapterLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features):
        super(RepAdapterLinear, self).__init__(in_features=in_features, out_features=out_features,)
        self.adapter = RepAdapter(in_features=in_features)
    
    def forward(self, input):
        input = self.adapter(input) + input
        return F.linear(input, self.weight, self.bias)

    @staticmethod
    def from_linear(linear_module):
        new_linear = RepAdapterLinear(linear_module.in_features, linear_module.out_features)
        new_linear.weight = linear_module.weight
        new_linear.bias = linear_module.bias
        return new_linear
    
    @staticmethod
    def to_linear(repadapter_module):
        repadapter_module.reparameterize()
        new_linear = nn.Linear(repadapter_module.in_features, repadapter_module.out_features)
        new_linear.weight = repadapter_module.weight
        new_linear.bias = repadapter_module.bias
        return new_linear
    
    def reparameterize(self,do_residual=False):
        Wa = self.adapter.conv_A.weight
        Wb = self.adapter.conv_B.weight
        Ba = self.adapter.conv_A.bias
        Bb = self.adapter.conv_B.bias
        print(f'shape of Ba {Ba.shape}')
        print(f'shape of Wb {Wb.shape}')
        scale = self.adapter.scale
        bias = None
        id_tensor = None
        if Ba is not None and Wb is not None:
            bias = (Ba[:, None] * Wb.squeeze(-1)).sum(dim=0)
        if Bb is not None:
            if bias is not None:
                bias = bias + Bb
            else:
                bias = Bb.clone()

        if do_residual:
            id_tensor = torch.eye(Wa.shape[0], Wb.shape[1]).to(Wa.device)
        weight = Wa @ Wb * scale + id_tensor
        self.weight = weight.T
        self.bias = bias * scale if isinstance(bias, torch.Tensor) else None


class RepadpterModuleInjection:
    @staticmethod
    def make_scalable(linear_module):
        """Make a (linear) layer adding repadapter.
        :param linear_module: A Linear module
        :return: a repadapter linear layer
        """
        new_linear = RepAdapterLinear.from_linear(linear_module)
        return new_linear
    
class RepadpterReparam:
    @staticmethod
    def reparam(repadapter_module):
        """reparam a repadapter layer to a linear layer.
        :param repadapter_module: A repadapter module 
        :return: a linear layer
        """
        new_linear = RepAdapterLinear.to_linear(repadapter_module)
        return new_linear
    
    
class RepAdapter(nn.Module):
    """
    Pytorch Implemention of RepAdapter for 1d tensor
    copy from https://github.com/luogen1996/RepAdapter/blob/main/repadapter.py
    """

    def __init__(
            self,
            in_features=768,
            hidden_dim=8,
            groups=2,
            scale=10
    ):
        super().__init__()
        self.conv_A = nn.Conv1d(in_features,hidden_dim,1,groups=1,bias=True)
        self.conv_B = nn.Conv1d(hidden_dim, in_features, 1, groups=groups, bias=True)

        self.dropout=nn.Dropout(0.1)
        self.groups=groups
        self.scale=scale

        nn.init.xavier_uniform_(self.conv_A.weight)
        nn.init.zeros_(self.conv_A.bias)
        nn.init.zeros_(self.conv_B.weight)
        nn.init.zeros_(self.conv_B.bias)

    def forward(self, x,weights=None):
        x=x.transpose(1,2)
        x=self.conv_B(self.dropout(self.conv_A(x)))
        x=x.transpose(1,2).contiguous()
        return x
    

def set_repadapter(model):
    layers = []
    for name, l in model.named_modules():
        if isinstance(l, nn.Linear):
            tokens = name.strip().split('.')
            layer = model
            for t in tokens[:-1]:
                if not t.isnumeric():
                    layer = getattr(layer, t)
                else:
                    layer = layer[int(t)]
            layers.append([layer, tokens[-1]])
    for parent_layer, last_token in layers:
        if not 'head' in last_token:
            setattr(parent_layer, last_token, RepadpterModuleInjection.make_scalable(getattr(parent_layer, last_token)))

@torch.no_grad()
def save_repadapter(save_path, model):
    model.eval()
    model = model.cpu()
    trainable = {}
    for n, p in model.named_parameters():
        if any([x in n for x in ['adapter']]):
            trainable[n] = p.data
    torch.save(trainable, save_path)
    
def load_repadapter(load_path, model):
    weights = torch.load(load_path)
    loaded = 0
    for n, p in model.named_parameters():
        if any([x in n for x in ['adapter']]):
            p.data = weights[n]
            loaded +=1
    print(f'successfully loaded {loaded} trained parameter tensors')
    return model


if __name__ == '__main__':
    input_tensor = torch.randn(1, 1, 20)
    repadpter_linear = RepAdapterLinear(20, 10)
    nn.init.normal_(repadpter_linear.adapter.conv_A.weight)
    nn.init.normal_(repadpter_linear.adapter.conv_B.weight)
    output_repadpter = repadpter_linear(input_tensor)
    print("RepAdapterLinear 输出结果：", output_repadpter)
    repadpter_linear.reparameterize()
    linear_module = RepAdapterLinear.to_linear(repadpter_linear)
    output_linear = linear_module(input_tensor)
    # print("RepadpterLinear 输出结果：", output_repadpter)
    print("Linear 输出结果：", output_linear)