import math
import numpy as np
import ot
import torch
import torch.nn as nn
import torch.nn.functional as F
# from geomloss import SamplesLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 功率约束
def PowerNormalize(z):
    z_square = torch.mul(z, z)
    power = torch.mean(z_square).sqrt()
    if power > 1:
        z = torch.div(z, power)
    return z

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):  # 128 128 32
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), x.size(1))

class Resblock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 1, bias=False)
        )

    def forward(self, x):
        return x + self.model(x)


class Resblock_down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 2, bias=False)
        )

    def forward(self, x):
        return self.downsample(x) + self.model(x)


class Encoder(nn.Module):
    def __init__(self, output_channel, psnr):
        super(Encoder, self).__init__()
        self.output_channel = output_channel
        self.psnr = psnr
        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            Resblock(64)
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            Resblock(128)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            Resblock(256)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            Resblock(512)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        # self.linear = nn.Sequential(
        #     nn.Linear(2048, 1024),
        #     nn.ReLU()
        # )
    def forward(self, x0):
        x = self.prep(x0)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = torch.reshape(x, (x.size()[0], 128 * 4 * 4))
        # x = self.linear(x)

        return x

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings  # 16
        self.embedding_dim = embedding_dim     # 512
        self.commitment_cost = commitment_cost
        self.alpha = 0.5

        self.epsilon = 0.05
        self.dual_steps = 10
        self.dual_lr = 0.5
        self.hist_temperature = 0.5
        self.ot_weight = 1.0
        self.gaussian_mean = None
        self.gaussian_std = None

        # self.codebook = nn.Parameter(torch.randn(self.num_embeddings, self.embedding_dim)) # 初始化为正态分布
        self.codebook = nn.Parameter(torch.empty(self.num_embeddings, self.embedding_dim).data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)) # 初始化为均匀分布

    def _normalize_prob(self, p, eps=1e-12):
        p = p.clamp_min(eps)
        return p / p.sum()

    def _dual_transport_objective(self, phi, src_w, tgt_w, cost):
        src_w = self._normalize_prob(src_w)
        tgt_w = self._normalize_prob(tgt_w)

        log_tgt = torch.log(tgt_w.clamp_min(1e-12)).unsqueeze(0)
        exp_term = (-cost + phi.unsqueeze(0)) / self.epsilon
        logsumexp = torch.logsumexp(log_tgt + exp_term, dim=1)

        obj = torch.sum(src_w * (-self.epsilon * logsumexp)) + torch.sum(tgt_w * phi)
        return obj

    def _dual_ot_loss(self, src_w, tgt_w, cost):
        src_det = src_w.detach()
        tgt_det = tgt_w.detach()
        cost_det = cost.detach()

        phi = torch.zeros_like(tgt_det, requires_grad=True)

        for _ in range(self.dual_steps):
            obj = self._dual_transport_objective(phi, src_det, tgt_det, cost_det)
            grad_phi = torch.autograd.grad(obj, phi, create_graph=False)[0]
            phi = (phi + self.dual_lr * grad_phi).detach().requires_grad_(True)

        phi_star = phi.detach()
        loss = self._dual_transport_objective(phi_star, src_w, tgt_w, cost)
        return loss

    def mod_channel_demod(self, mod, x, device):
        """信道调制和解调"""
        X = mod.modulate(x)
        X = mod.awgn(X)  # 添加 AWGN 噪声
        return mod.demodulate(X).to(device)  # 解调并返回

    def construct_noise(self, encodings, mod, device):
        """构造噪声用于扰动编码"""
        x = torch.argmax(encodings, dim=-1)  # 获取编码的索引
        x_tilde = self.mod_channel_demod(mod, x, device)  # 调制并解调
        noise = F.one_hot(x_tilde, num_classes=self.num_embeddings).float() - \
               F.one_hot(x, num_classes=self.num_embeddings).float()  # 计算噪声
        return noise

    def recover(self, encodings):
        """通过编码还原数据"""
        out = torch.matmul(encodings, self.codebook)
        return out

    def generate_awgn_gaussian(self, device, dtype):
        indices = torch.arange(self.num_embeddings, device=device, dtype=dtype)
        mean = self.gaussian_mean
        if mean is None:
            mean = (self.num_embeddings - 1) / 2
        std = self.gaussian_std
        if std is None:
            std = self.num_embeddings / 6
        gaussian = torch.exp(-0.5 * ((indices - mean) / std) ** 2)
        return gaussian / gaussian.sum()

    def forward(self, inputs, mod):
        """前向传播，进行量化并计算损失"""
        inputs = inputs.permute(0, 2, 3, 1).contiguous()  # Convert BCHW -> BHWC
        inputs_shape = inputs.shape

        flat_input = inputs.view(-1, self.embedding_dim).to(inputs.device)  # Flatten inputs for distance calculation
        d = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.codebook ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.codebook.t()))
        # 找到最近的编码
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.num_embeddings).to(inputs.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)
        min_encodings = PowerNormalize(min_encodings)
        # 调制与解调
        noise = self.construct_noise(min_encodings, mod=mod, device=inputs.device)
        min_encodings_noise = min_encodings + noise
        quantized = self.recover(min_encodings_noise)
        # quantized = PowerNormalize(quantized)

        if self.training:
            hard_hist = torch.mean(min_encodings, dim=0)
            # 硬分配得到的码字激活频率

            soft_assign = F.softmax(-d / self.hist_temperature, dim=1)
            # 基于距离的软分配概率

            soft_hist = torch.mean(soft_assign, dim=0)
            # 软分配得到的码字激活频率

            # codeword_weight = hard_hist + (soft_hist - soft_hist.detach())
            # 前向看硬分布，反向走软分布梯度
            codeword_weight = self._normalize_prob(soft_hist)

            codeword_weight = self._normalize_prob(codeword_weight)
            # 归一化成合法概率分布

            uniform_target = torch.ones(
                self.num_embeddings,
                device=inputs.device,
                dtype=inputs.dtype,
            ) / self.num_embeddings
            # 均匀目标分布

            gaussian_target = self.generate_awgn_gaussian(inputs.device, inputs.dtype)
            gaussian_target = self._normalize_prob(gaussian_target)
            # 高斯目标分布

            mixed_target = self.alpha * uniform_target + (1.0 - self.alpha) * gaussian_target
            mixed_target = self._normalize_prob(mixed_target)
            # 混合目标分布

            codeword_support = torch.arange(
                self.num_embeddings,
                device=inputs.device,
                dtype=inputs.dtype,
            ).view(-1, 1)
            # 码字索引 support

            distances = torch.cdist(codeword_support, codeword_support, p=2)
            # OT 代价矩阵

            loss = self.ot_weight * self._dual_ot_loss(
                codeword_weight, mixed_target, distances
            )
            # 当前码字分布 vs 混合目标分布 的半对偶 OT 损失

            print(f"dual_ot_Loss: {loss.item():.4f}")
        else:
            loss = 0.0
            
        quantized = quantized.view(inputs_shape)
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        return quantized, loss, perplexity, min_encodings, min_encodings_noise

class Classifier(nn.Module):
    def __init__(self, psnr):
        super(Classifier, self).__init__()
        self.psnr = psnr
        # self.fc = nn.Sequential(
        #     nn.Linear(1024, 1024),
        #     nn.Tanh()
        # )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Resblock(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            Resblock(256)
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Resblock(128),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.classifier1 = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            Flatten()
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(64, 10),
        )

    def forward(self, x):
        # x = self.fc(x)
        # x = torch.reshape(x, (-1, 64, 4, 4))
        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.classifier1(x)
        output = self.classifier2(x)
        return output

class classification_Model(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, output_channel, psnr, mod=None):
        super(classification_Model, self).__init__()
        self.Encoder = Encoder(output_channel, psnr)
        self.vq_vae = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.Classifier = Classifier(psnr)
        self.mod = mod  # 存储 mod 为成员变量

    def forward(self, x, mod=None):
        if mod is None:
            mod = self.mod
        if mod is None:
            raise ValueError("mod 是 None，请在初始化或 forward 调用时显式传入 mod")
        z = self.Encoder(x)
        quantized, loss, perplexity = self.vq_vae(z, mod=mod)
        y_hat = self.Classifier(quantized)

        return loss, y_hat, perplexity

