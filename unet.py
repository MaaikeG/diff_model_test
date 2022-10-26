import math
import torch
import torch.nn as nn



class AttentionBlock(nn.Module):
    def __init__(self, in_size, attention_size=16):
        super().__init__()

        self.query_layer = torch.nn.Conv2d(kernel_size=1, in_channels=in_size, out_channels=attention_size)
        self.key_layer= torch.nn.Conv2d(kernel_size=1, in_channels=in_size, out_channels=attention_size)
        self.value_layer= torch.nn.Conv2d(kernel_size=1, in_channels=in_size, out_channels=in_size)

        self.recip_sqrt_size = 1. / math.sqrt(attention_size)

        self.softmax = torch.nn.Softmax(-1) # query dim

    def forward(self, x):
        queries = self.query_layer(x)
        keys = self.key_layer(x)
        values = self.value_layer(x)

        # dot all queries with keys
        attn = torch.einsum('bqwh, bkWH -> bwhWH', queries, keys) 
        b, w, h, _, _ = attn.shape
        attn = self.softmax(attn.view(b, w, h, -1) * self.recip_sqrt_size) # softmax the attention scores over the pixels
        attn = attn.view(b, w, h, w, h)

        out = torch.einsum('bwhWH, bvWH -> bvwh', attn, values)

        return out 


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_dim, up=True, add_attention=False):
        super().__init__()

        if up:
            in_ch *= 2 # allow for concatenation with redidual values
            self.sample_layer = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=2, stride=2, padding=0)
        else:
            self.sample_layer = nn.Conv2d(out_ch, out_ch, kernel_size=2, stride=2, padding=0)

        groupnorm_size = max(1, in_ch // 4)
        self.groupnorm = torch.nn.GroupNorm(num_groups=groupnorm_size, num_channels=in_ch)

        self.conv_1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()    
        self.time_mlp = nn.Linear(time_embed_dim, out_ch)

        self.attention = None
        if add_attention:
            self.attention = AttentionBlock(out_ch)


    def forward(self, x, t):
        x = self.groupnorm(x)

        x = self.relu(self.conv_1(x))
        t = self.relu(self.time_mlp(t))

        x = x + t[..., None, None]
        x = self.relu(self.conv_2(x))

        if self.attention is not None:
            x = x + self.attention(x)

        return self.sample_layer(x)


def construct_conv_blocks(intermediate_channels, time_embed_dim, attention_resolutions, up=False):
    blocks = nn.ModuleList()
    for i, (c_in, c_out) in enumerate(zip(intermediate_channels[:-1], intermediate_channels[1:])):
        add_attention = c_out in attention_resolutions
        blocks.append(ResBlock(c_in, c_out, time_embed_dim, up, add_attention))
    return blocks


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim, attention_resolutions=[128]):
        super(UNet, self).__init__()

        intermediate_channels = [64, 128, 256]

        self.input_conv = nn.Conv2d(in_channels, intermediate_channels[0], kernel_size=3, stride=1, padding=1)

        self.down_blocks = construct_conv_blocks(intermediate_channels, time_embed_dim, attention_resolutions, up=False)
        # up channels go in reversed order
        intermediate_channels.reverse()
        self.up_blocks = construct_conv_blocks(intermediate_channels, time_embed_dim, attention_resolutions, up=True)
        
        self.attention_resolutions = attention_resolutions

        self.output_layer = nn.Conv2d(intermediate_channels[-1], out_channels, kernel_size=1, stride=1)

            
    def forward(self, x, t):
        intermediate_outputs = [] # each intermediate output is passed on 
                                  # to the corresponding upsampling layer

        x = self.input_conv(x)

        # encoder
        for layer in self.down_blocks:
             x = layer(x, t)
             intermediate_outputs.append(x)
         #    print(x.shape)
             
        intermediate_outputs.reverse()

        # decoder
        for layer, old_x in zip(self.up_blocks, intermediate_outputs):
            x = torch.cat([x, old_x], 1)
            x = layer(x, t)
          #  print(x.shape)
           
        return self.output_layer(x)
