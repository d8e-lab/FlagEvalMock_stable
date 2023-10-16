import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class ScalableLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, rank):
        super(ScalableLinear, self).__init__(in_features=in_features, out_features=out_features)
        config = [f'LoRA_{rank}', 'none']
        self.configs = [config]
        self.Bd, self.Bu = self.make_param((out_features, in_features), f'LoRA_{rank}')
        self.eval_config = None
        nn.init.xavier_uniform_(self.Bu)
    
    def prepare_path(self, config, Xd, Xu=None):
        gate_LoRA = 1 if 'LoRA' in config else 0
        X_LoRA = torch.matmul(Xd, Xu) * gate_LoRA
        X = X_LoRA 
        return X
    
    def make_param(self, shape, config=None):
        if 'LoRA' in config:
            out_feature = shape[0]
            in_feature = shape[1]
            try:
                rank = int(config.split('_')[1])
            except:
                rank = 4
            return nn.Parameter(torch.zeros(out_feature, rank)), nn.Parameter(torch.zeros(rank, in_feature))
        return nn.Parameter(torch.zeros(*shape))
        
    def forward(self, input):
        if self.eval_config is not None:
            path_config = self.eval_config
        else:
            path_config = random.choice(self.configs)
        B = self.prepare_path(path_config['B'], self.Bd, self.Bu)
        optimal_weight = self.weight + B
        if torch.is_tensor(self.bias):
            optimal_bias = self.bias 
        else:
            optimal_bias =0
        return F.linear(input, optimal_weight, optimal_bias)

    @staticmethod
    def from_linear(linear_module, rank):
        new_linear = ScalableLinear(linear_module.in_features, linear_module.out_features, rank)
        new_linear.weight = linear_module.weight
        new_linear.bias = linear_module.bias
        return new_linear

class XloraModuleInjection:
    @staticmethod
    def make_scalable(linear_module, rank=4):
        """Make a (linear) layer super scalable.
        :param linear_module: A Linear module
        :return: a suepr linear that can be trained to
        """
        new_linear = ScalableLinear.from_linear(linear_module, rank)
        return new_linear
    
def set_xlora(model, lora_rank):
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
            setattr(parent_layer, last_token, XloraModuleInjection.make_scalable(getattr(parent_layer, last_token), lora_rank))

@torch.no_grad()
def save_xlora(save_path, model):
    model.eval()
    model = model.cpu()
    trainable = {}
    for n, p in model.named_parameters():
        if any([x in n for x in ['A', 'B', 'C', 'D', 'E']]):
            trainable[n] = p.data
    torch.save(trainable, save_path )
    
def load_xlora(load_path, model):
    weights = torch.load(load_path)
    loaded = 0
    for n, p in model.named_parameters():
        if any([x in n for x in ['A', 'B', 'C', 'D', 'E']]):
            p.data = weights[n]
            loaded +=1
    print(f'successfully loaded {loaded} trained parameter tensors')
    return model