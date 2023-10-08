import torch
import torch.nn as nn
import torch.nn.functional as F

def set_repadapter(model, use_router):
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
            setattr(parent_layer, last_token, RepadpterModuleInjection.make_scalable(getattr(parent_layer, last_token), use_router))

def load_repadapter(args, model):
    weights = torch.load(args.load_path + args.dataset + '.pt')
    loaded = 0
    for n, p in model.named_parameters():
        if any([x in n for x in ['A', 'B', 'C', 'D', 'E', 'head']]):
            p.data = weights[n]
            loaded +=1
    print(f'successfully loaded {loaded} trained parameter tensors')
    return model
            
            
class RepadpterLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features,use_router=False):
        super(RepadpterLinear, self).__init__(in_features=in_features, out_features=out_features,)
        if use_router:
            self.adapter = RepAdapter_Router(in_features=in_features)
        else :
            self.adapter = RepAdapter(in_features=in_features)
    
    def forward(self, input):
        input = self.adapter(input)
        return F.linear(input, self.weight, self.bias)

    @staticmethod
    def from_linear(linear_module, use_router):
        new_linear = RepadpterLinear(linear_module.in_features, linear_module.out_features,use_router=use_router)
        new_linear.weight = linear_module.weight
        new_linear.bias = linear_module.bias
        return new_linear

class RepadpterModuleInjection:
    @staticmethod
    def make_scalable(linear_module, use_router=False):
        """Make a (linear) layer super scalable.
        :param linear_module: A Linear module
        :return: a suepr linear that can be trained to
        """
        new_linear = RepadpterLinear.from_linear(linear_module, use_router)
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
            scale=1
    ):
        super().__init__()
        self.conv_A=nn.Conv1d(in_features,hidden_dim,1,groups=1,bias=True)
        self.conv_B = nn.Conv1d(hidden_dim, in_features, 1, groups=groups, bias=True)

        self.dropout=nn.Dropout(0.1)
        self.groups=groups
        self.scale=scale

        nn.init.xavier_uniform_( self.conv_A.weight)
        nn.init.zeros_(self.conv_A.bias)
        nn.init.zeros_(self.conv_B.weight)
        nn.init.zeros_(self.conv_B.bias)

    def forward(self, x,weights=None):
        x=x.transpose(1,2)
        x=self.conv_B(self.dropout(self.conv_A(x)))
        x=x.transpose(1,2).contiguous()
        return x
    
class RepAdapter_Router(nn.Module):
    """ Pytorch Implemention of RepAdapter for 1d tensor"""

    def __init__(
            self,
            in_features=768,
            hidden_dim=8,
            groups=2,
            scale=1,
            t=10.
    ):
        super().__init__()
        self.conv_A=nn.Conv1d(in_features,hidden_dim,1,groups=1,bias=True)
        self.conv_B = nn.Conv1d(hidden_dim, in_features, 1, groups=groups, bias=True)

        self.conv_D = nn.Conv1d(hidden_dim, in_features, 1, groups=groups, bias=True)

        self.expert_weights=nn.Linear(in_features,2)

        self.dropout=nn.Dropout(0.1)
        self.groups=groups
        self.scale=scale
        self.t=t

        nn.init.xavier_uniform_( self.conv_A.weight)
        nn.init.zeros_(self.conv_A.bias)
        nn.init.zeros_(self.conv_B.weight)
        nn.init.zeros_(self.conv_B.bias)


        nn.init.zeros_(self.conv_D.weight)
        nn.init.zeros_(self.conv_D.bias)

    def forward(self, x,weights=None):
        if weights is None:
            weights=torch.softmax(self.expert_weights(x[:,0])/self.t,-1)
        x=x.transpose(1,2)
        x_=self.dropout(self.conv_A(x))
        x=self.conv_B(x_)*self.scale*weights[:,0,None,None]+self.conv_D(x_)*self.scale*weights[:,1,None,None]+x
        x=x.transpose(1,2).contiguous()
        return x