import torch
import torch.nn as nn

class ConvGRUCell(nn.Module):
    def __init__(self, n_attrs, in_dim, out_dim, kernel_size=3,deconv=False):
        super(ConvGRUCell, self).__init__()
        self.n_attrs = n_attrs
        self.deconv = deconv

        if self.deconv:
            self.upsample = nn.ConvTranspose2d(
            in_dim * 2 + n_attrs, out_dim, 4, 2, 1, bias=False)

        else:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_dim * 2 + n_attrs, out_dim, 3, 1, 1, bias=False))

        self.reset_gate = nn.Sequential(
            nn.Conv2d(in_dim + out_dim, out_dim, kernel_size,
                      1, (kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Sigmoid()
        )

        self.update_gate = nn.Sequential(
            nn.Conv2d(in_dim + out_dim, out_dim, kernel_size,
                      1, (kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Sigmoid()
        )
        self.hidden = nn.Sequential(
            nn.Conv2d(in_dim + out_dim, out_dim, kernel_size,
                      1, (kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Tanh()
        )

    def forward(self, input, old_state, attr):

        n, _, h, w = old_state.size()
        attr = attr.view((n, self.n_attrs, 1, 1)).expand(
            (n, self.n_attrs, h, w))

        state_hat = self.upsample(torch.cat([old_state, attr], 1))
        r = self.reset_gate(torch.cat([input, state_hat], dim=1))
        z = self.update_gate(torch.cat([input, state_hat], dim=1))
        new_state = r * state_hat
        hidden_info = self.hidden(torch.cat([input, new_state], dim=1))
        output = (1-z) * state_hat + z * hidden_info
        return output, new_state


class Generator(nn.Module):

    def __init__(self, attr_dim, conv_dim=64, n_layers=5, shortcut_layers=2, stu_kernel_size=3, use_stu=True, one_more_conv=True,deconv= False):
        super(Generator, self).__init__()
        self.n_attrs = attr_dim
        self.n_layers = n_layers
        self.shortcut_layers = min(shortcut_layers, n_layers - 1)
        self.use_stu = use_stu
        self.deconv = deconv
        in_channels = 3

        
        if not self.deconv:
            self.upsample = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False)
        self.encoder = nn.ModuleList()
        
        for i in range(self.n_layers):
            self.encoder.append(nn.Sequential(
                nn.Conv2d(in_channels, conv_dim * 2 ** i, 4, 2, 1, bias=False),
                nn.BatchNorm2d(conv_dim * 2 ** i),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ))
            in_channels = conv_dim * 2 ** i

        if use_stu:
            
            self.stu = nn.ModuleList()
            for i in reversed(range(self.n_layers - 1 - self.shortcut_layers, self.n_layers - 1)):
                self.stu.append(ConvGRUCell(
                    self.n_attrs, conv_dim * 2 ** i, conv_dim * 2 ** i, stu_kernel_size,deconv=deconv))

        self.decoder = nn.ModuleList()
        for i in range(self.n_layers):
            if i < self.n_layers - 1:
                if i == 0:
                    if self.deconv:
                        self.decoder.append(nn.Sequential(
                        nn.ConvTranspose2d(conv_dim * 2 ** (self.n_layers - 1) + attr_dim,
                                           conv_dim * 2 ** (self.n_layers - 1), 4, 2, 1, bias=False),
                        nn.BatchNorm2d(in_channels),
                        nn.ReLU(inplace=True)
                        ))
                    else:
                        self.decoder.append(nn.Sequential(
                            nn.Conv2d(conv_dim * 2 ** (self.n_layers - 1) + attr_dim,
                                    conv_dim * 2 ** (self.n_layers - 1), 3, 1, 1, bias=False),
                            nn.BatchNorm2d(in_channels),
                            nn.ReLU(inplace=True)
                        ))
                elif i <= self.shortcut_layers:     # not <
                    if self.deconv:

                        self.decoder.append(nn.Sequential(
                            nn.ConvTranspose2d(conv_dim * 3 * 2 ** (self.n_layers - 1 - i),
                                            conv_dim * 2 ** (self.n_layers - 1 - i), 4, 2, 1, bias=False),
                            nn.BatchNorm2d(
                                conv_dim * 2 ** (self.n_layers - 1 - i)),
                            nn.ReLU(inplace=True)
                        ))
                    
                    else:
                        self.decoder.append(nn.Sequential(
                            nn.Conv2d(conv_dim * 3 * 2 ** (self.n_layers - 1 - i),
                                    conv_dim * 2 ** (self.n_layers - 1 - i),  3, 1, 1, bias=False),
                            nn.BatchNorm2d(
                                conv_dim * 2 ** (self.n_layers - 1 - i)),
                            nn.ReLU(inplace=True)
                        ))
                else:

                    if self.deconv:
                        self.decoder.append(nn.Sequential(
                            nn.ConvTranspose2d(conv_dim * 2 ** (self.n_layers - i),
                                            conv_dim * 2 ** (self.n_layers - 1 - i), 4, 2, 1, bias=False),
                            nn.BatchNorm2d(
                                conv_dim * 2 ** (self.n_layers - 1 - i)),
                            nn.ReLU(inplace=True)
                        ))

                    else:
                        self.decoder.append(nn.Sequential(
                            nn.Conv2d(conv_dim * 2 ** (self.n_layers - i),
                                    conv_dim * 2 ** (self.n_layers - 1 - i),  3, 1, 1, bias=False),
                            nn.BatchNorm2d(
                                conv_dim * 2 ** (self.n_layers - 1 - i)),
                            nn.ReLU(inplace=True)
                        ))
            else:

                in_dim = conv_dim * 3 if self.shortcut_layers == self.n_layers - 1 else conv_dim * 2
                if one_more_conv:
                    if self.deconv:
                        self.decoder.append(nn.Sequential(
                            nn.ConvTranspose2d(in_dim, conv_dim // 4, 4, 2, 1, bias=False),
                            nn.BatchNorm2d(conv_dim // 4),
                            nn.ReLU(inplace=True),
                            nn.ConvTranspose2d(conv_dim // 4, 3, 3, 1, 1, bias=False),
                            nn.Tanh()
                        ))

                    else:
                        self.decoder.append(nn.Sequential(
                            nn.Conv2d(
                                in_dim, conv_dim // 4, 3, 1, 1, bias=False),
                            nn.BatchNorm2d(conv_dim // 4),
                            nn.ReLU(inplace=True),

                            nn.Conv2d(
                                conv_dim // 4, 3, 3, 1, 1, bias=False),
                            nn.Tanh()
                        ))
                else:
                    if self.deconv:
                        self.decoder.append(nn.Sequential(
                            nn.ConvTranspose2d(in_dim, 3, 4, 2, 1, bias=False),
                            nn.Tanh()
                        ))

                    else:
                        self.decoder.append(nn.Sequential(
                            nn.Conv2d(in_dim, 3, 3, 1, 1, bias=False),
                            nn.Tanh()
                        ))

    def encode(self, x):
        # propagate encoder layers
        y = []
        x_ = x
        for layer in self.encoder:
            x_ = layer(x_)
            y.append(x_)

        return y[-1], y
                
    def forward(self, x, a):

        out,y = self.encode(x)
        n, _, h, w = out.size()
        attr = a.view((n, self.n_attrs, 1, 1)).expand((n, self.n_attrs, h, w))

        if self.deconv:
            out = self.decoder[0](torch.cat([out, attr], dim=1))
        else:
            out = self.decoder[0](self.upsample(torch.cat([out, attr], dim=1)))
        stu_state = y[-1]

        # propagate shortcut layers
        for i in range(1, self.shortcut_layers + 1):
            if self.use_stu:
                stu_out, stu_state = self.stu[i-1](y[-(i+1)], stu_state, a)

                if self.deconv:
                    out = self.decoder[i](torch.cat([out, stu_out], dim=1))
                else:
                    out = self.decoder[i](self.upsample(
                        torch.cat([out, stu_out], dim=1)))
            else:
                if self.deconv:
                    out = self.decoder[i](torch.cat([out, y[-(i+1)]], dim=1))
                else:
                    out = self.decoder[i](self.upsample(
                    torch.cat([out, y[-(i+1)]], dim=1)))

        # propagate non-shortcut layers
        for i in range(self.shortcut_layers + 1, self.n_layers):

            if self.deconv:
                out = self.decoder[i](out)
            else:
                out = self.decoder[i](self.upsample(out))
        
        return out

class Discriminator(nn.Module):
    def __init__(self, image_size=128, attr_dim=10, conv_dim=64, fc_dim=1024, n_layers=5):
        super(Discriminator, self).__init__()
        layers = []
        in_channels = 3

        for i in range(n_layers):
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, conv_dim * 2 ** i, 4, 2, 1),
                nn.InstanceNorm2d(conv_dim * 2 ** i,
                                  affine=True, track_running_stats=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ))
            in_channels = conv_dim * 2 ** i
        self.conv = nn.Sequential(*layers)
                    
        feature_size = image_size // 2**n_layers
        self.fc_adv = nn.Sequential(
            nn.Linear(conv_dim * 2 ** (n_layers - 1)
                      * feature_size ** 2, fc_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(fc_dim, 1)
        )
        self.fc_att = nn.Sequential(
            nn.Linear(conv_dim * 2 ** (n_layers - 1)
                      * feature_size ** 2, fc_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(fc_dim, attr_dim),
        )

        self.fc_att.add_module('Sigmoid',nn.Sigmoid())

    def forward(self, x):

        y = self.conv(x)
        y = y.view(y.size()[0], -1)
        logit_adv = self.fc_adv(y)
        logit_att = self.fc_att(y)

        return logit_adv, logit_att
