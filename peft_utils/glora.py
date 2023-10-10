import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class SuperScalableLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, rank):
        super(SuperScalableLinear, self).__init__(in_features=in_features, out_features=out_features)
        config_A_B = [f'LoRA_{rank}', 'vector', 'constant']
        config_C = [f'LoRA_{rank}', 'vector',]
        config_D_E = ['constant', 'vector']
        self.configs = []
        for A in config_A_B:
            for B in config_A_B:
                for C in config_C:
                    for D in config_D_E:
                        for E in config_D_E:
                            config = {'A':A,'B':B,'C':C,'D':D,'E':E}
                            self.configs.append(config)

        self.Ad, self.Au = self.make_param((out_features, in_features), f'LoRA_{rank}')
        self.Bd, self.Bu = self.make_param((out_features, in_features), f'LoRA_{rank}')
        self.Cd, self.Cu = self.make_param((in_features, 1), f'LoRA_{rank}')
        self.D = nn.Parameter(torch.zeros(out_features))
        self.E = nn.Parameter(torch.zeros(out_features))
        self.eval_config = None
        nn.init.xavier_uniform_(self.Au)
        nn.init.xavier_uniform_(self.Bu)
        nn.init.xavier_uniform_(self.Cu)
    
    def prepare_path(self, config, Xd, Xu=None):
        if Xu is not None:
            if 'LoRA' in config:
                rank = int(config.split('_')[1])
                X = torch.matmul(Xd[:,:rank], Xu[:rank, :])
            elif 'vector' in config:
                X = Xd[:,0].unsqueeze(1)
            elif 'constant' in config:
                X = Xd[0,0]
            else:
                raise ValueError
        else:
            if 'vector' in config:
                X = Xd
            elif 'constant' in config:
                X = Xd[0]
            else:
                raise ValueError
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
        A = self.prepare_path(path_config['A'], self.Ad, self.Au)
        B = self.prepare_path(path_config['B'], self.Bd, self.Bu)
        C = self.prepare_path(path_config['C'], self.Cd, self.Cu)
        D = self.prepare_path(path_config['D'], self.D)
        E = self.prepare_path(path_config['E'], self.E)
        optimal_weight = self.weight + self.weight*A + B
        if torch.is_tensor(self.bias):
            optimal_bias = self.bias + self.bias*D + E
        else:
            optimal_bias = E
        optimal_prompt = torch.matmul(self.weight, C).squeeze()
        return F.linear(input, optimal_weight, optimal_bias+optimal_prompt)

    @staticmethod
    def from_linear(linear_module, rank):
        new_linear = SuperScalableLinear(linear_module.in_features, linear_module.out_features, rank)
        new_linear.weight = linear_module.weight
        new_linear.bias = linear_module.bias
        return new_linear
    
class GloraModuleInjection:
    @staticmethod
    def make_scalable(linear_module, rank=4, eval_config=None):
        """Make a (linear) layer super scalable.
        :param linear_module: A Linear module
        :return: a suepr linear that can be trained to
        """
        new_linear = SuperScalableLinear.from_linear(linear_module, rank)
        if eval_config is not None:
            new_linear.eval_config = eval_config
        return new_linear
    
def set_glora(model, rank):
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
            setattr(parent_layer, last_token, GloraModuleInjection.make_scalable(getattr(parent_layer, last_token), rank))

@torch.no_grad()
def save_glora(save_path, model):
    model.eval()
    model = model.cpu()
    trainable = {}
    for n, p in model.named_parameters():
        if any([x in n for x in ['A', 'B', 'C', 'D', 'E', 'head']]):
            trainable[n] = p.data
    torch.save(trainable, save_path )
    
def load_glora(load_path, model):
    weights = torch.load(load_path)
    weights = {k.replace('module.', '') if k.startswith('module.') else k: v for k, v in weights.items()}
    loaded = 0
    for n, p in model.named_parameters():
        if any([x in n for x in ['A', 'B', 'C', 'D', 'E', 'head']]):
            p.data = weights[n]
            loaded +=1
    print(f'successfully loaded {loaded} trained parameter tensors')
    return model

def set_glora_with_config(model, lora_rank, configs,set_head=False,use_print=False):
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
            layers.append([layer, tokens[-1],name])
    count=0
    for parent_layer, last_token, name in layers:
        config = configs[count]
        if set_head:
            setattr(parent_layer, last_token, GloraModuleInjection.make_scalable(getattr(parent_layer, last_token, config), lora_rank))
            if use_print:
                print(f'{name} config:{config}')
        else:
            if not 'head' in last_token:
                setattr(parent_layer, last_token, GloraModuleInjection.make_scalable(getattr(parent_layer, last_token, config), lora_rank))
                if use_print:
                    print(f'{name} config:{config}')
        count+=1
            