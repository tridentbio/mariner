from torch import nn

def build_model_from_yaml(yamlstr: str) -> nn.Module:
    return nn.Sequential(nn.Linear(100, 1))
