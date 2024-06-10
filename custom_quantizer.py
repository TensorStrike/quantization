import torch
import torch.nn as nn
import torch.nn.functional as F



random_int8 = torch.randint(-128,128,(32,16)).to(torch.int8)
print(random_int8)
random_hs = torch.randn((1,16), dtype=torch.bfloat16)
print(random_hs)
scales = torch.randn((1,32), dtype=torch.bfloat16)