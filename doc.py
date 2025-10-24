import torch

model_data = torch.load("model.pt", map_location="cpu")

# nếu file chứa state_dict:
if isinstance(model_data, dict):
    print(model_data.keys())  # xem có gì bên trong

# nếu là model
try:
    for name, param in model_data.items():
        print(name, param.shape)
except:
    print(model_data)
