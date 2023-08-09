"""
File to train and test the model
"""

import torch
import torch.nn as nn
from Transformer import Transformer
from GPTWrapper import WrappedModel
from Optimizer import OptimizerWrapper
from torch import tensor

# get the training data and split it into batches & pad it
# batch of sentences will be 2d tensor and then embedding will extend it to 3d tensor
...

# now we got a model
d_model, d_vocab = 512, 10
transformer_model = Transformer(dim_model=d_model)
model = WrappedModel(transformer_model, d_model, d_vocab)

# initialize it with some params from harvard, ty harvard <3
for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

# sanity check
def sanity():
    input_token = torch.rand(1, 4).to(torch.int32)
    output_token = torch.rand(1, 4).to(torch.int32)
    transformer_model = Transformer(num_enc_blocks=6, num_dec_blocks=6, out_distribution_size=100)
    model = WrappedModel(transformer_model, 512, 10)
    out = model(input_token, output_token)
    print("out: ", out.shape)
# sanity()

# some dude named adam says he can optimize our model
wrapped_optimizer = OptimizerWrapper(d_model, 2, 4000, 
                                     torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

# these labels are rough, lets smooth em
...

# ok nice now lets run epochs and compute loss then backprop it [reverse moonwalk in ML]
# loop thru epochs
# compute loss
# backward() to calculate gradient
# optim.step to calculate new params
# reset grad before next iter

# ok how'd we do :D