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
        self.alpha = 1

        # self.codebook = nn.Parameter(torch.randn(self.num_embeddings, self.embedding_dim)) # 初始化为正态分布
        self.codebook = nn.Parameter(torch.empty(self.num_embeddings, self.embedding_dim).data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)) # 初始化为均匀分布

        # self.codebook = nn.Embedding(self.num_embeddings, self.embedding_dim)  # 16X64
        # self.codebook.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)


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
        # out = torch.matmul(encodings, self.codebook.weight)
        out = torch.matmul(encodings, self.codebook)
        return out

    def generate_awgn_gaussian(self, device):
        indices = torch.arange(self.num_embeddings, device=device).float()
        mean = self.num_embeddings / 2
        # std = self.sigma_awgn
        std = self.num_embeddings / 6  # 可以调整 std，控制目标分布宽度
        gaussian = torch.exp(-0.5 * ((indices - mean) / std) ** 2)
        return gaussian / gaussian.sum()

    def forward(self, inputs, mod):
        """前向传播，进行量化并计算损失"""
        inputs = inputs.permute(0, 2, 3, 1).contiguous()  # Convert BCHW -> BHWC
        inputs_shape = inputs.shape

        flat_input = inputs.view(-1, self.embedding_dim).to(inputs.device)  # Flatten inputs for distance calculation

        # 计算样本与码字之间的距离（L2距离）
        # d = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
        #              + torch.sum(self.codebook.weight ** 2, dim=1)
        #              - 2 * torch.matmul(flat_input, self.codebook.weight.t()))
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
            # # ------------------------------------------------------------------------------------------------------
            # size_batch = flat_input.shape[0]
            # sample_weight = torch.ones(size_batch).to(inputs.device) / size_batch
            # codeword_weight = torch.ones(self.num_embeddings).to(inputs.device) / self.num_embeddings
            #
            # centroids = torch.matmul(torch.diag(torch.ones(self.num_embeddings)).to(inputs.device), self.codebook.weight)
            # loss_WS = ot.emd2(codeword_weight, sample_weight, d.t(), numItermax=500000)
            #
            # # --- 1. 获取当前平均码字使用概率（latent分布）
            # avg_probs = torch.mean(min_encodings, dim=0) + 1e-6
            # avg_probs = avg_probs / avg_probs.sum()
            #
            # # --- 2. 获取 AWGN 高斯目标分布
            # gaussian_target = self.generate_awgn_gaussian(inputs.device)
            #
            # # --- 3. 计算 pairwise 欧氏距离 (1D, 所以等价于距离差绝对值)
            # codebook_support = torch.arange(self.num_embeddings, device=inputs.device).float().view(-1, 1)
            # distances_gauss = torch.cdist(codebook_support, codebook_support, p=2)  # shape [K, K]
            #
            # # --- 5. Wasserstein 距离（POT）
            # loss_AWGN = ot.emd2(avg_probs, gaussian_target, distances_gauss, numItermax=50000)
            # loss = self.Lamda_WS * loss_WS + self.Lamda_GUASS * loss_AWGN
            # # ------------------------------------------------------------------------------------------------------
            # # --- Sinkhorn 损失
            # sinkhorn_loss = SamplesLoss("sinkhorn", p=2, blur=0.01)
            # # --- Sinkhorn loss 1: 样本到码字 ---
            # size_batch = flat_input.shape[0]
            # alpha = torch.ones(self.num_embeddings, device=inputs.device) / self.num_embeddings
            # beta = torch.ones(size_batch, device=inputs.device) / size_batch
            #
            # # 特征点
            # x_support = self.codebook
            # y_support = flat_input
            # loss_WS = sinkhorn_loss(alpha, x_support, beta, y_support)
            #
            # # --- Sinkhorn loss 2: latent 分布约束为 Gaussian ---
            # codebook_support = torch.linspace(0, 1, self.num_embeddings, device=inputs.device).unsqueeze(1)
            # # codebook_support = torch.arange(self.num_embeddings, device=inputs.device).float().view(-1, 1)
            #
            # # --- Soft assignment for gradient flow ---
            # temperature = 0.1  # 这个可调
            # soft_assign = F.softmax(-d / temperature, dim=1)
            # avg_probs = torch.mean(soft_assign, dim=0) + 1e-6
            # avg_probs = avg_probs / avg_probs.sum()
            # # 2. Ensure probs shape is [K, 1]
            # avg_probs = avg_probs.unsqueeze(1)
            #
            # gaussian_target = self.generate_awgn_gaussian(inputs.device)
            # gaussian_target = gaussian_target.unsqueeze(1)
            #
            # # 3. Stable Sinkhorn loss
            # sinkhorn_lossAWGN = SamplesLoss("sinkhorn", p=2, blur=0.01)
            # loss_AWGN = sinkhorn_lossAWGN(codebook_support, codebook_support, avg_probs, gaussian_target)
            #
            # # --- 总损失
            # loss = self.Lamda_WS * loss_WS + self.Lamda_GUASS * loss_AWGN
            # # ------------------------------------------------------------------------------------------------------方法一
            # codeword_weight = torch.ones(self.num_embeddings).to(inputs.device) / self.num_embeddings
            codeword_weight = torch.mean(min_encodings, dim=0)
            codeword_weight = codeword_weight / (codeword_weight.sum() + 1e-12)
            uniform_target = torch.ones(self.num_embeddings, device=inputs.device) / self.num_embeddings
            gaussian_target = self.generate_awgn_gaussian(inputs.device)
            mixed_target = self.alpha * uniform_target + (1 - self.alpha) * gaussian_target
            mixed_target = mixed_target / mixed_target.sum()  # 保证和为 1

            codeword_support = torch.arange(self.num_embeddings, device=inputs.device).float()  # 码字支持
            distances = torch.cdist(codeword_support.view(-1, 1), codeword_support.view(-1, 1), p=2)  # 计算欧氏距离

            # 距离矩阵（这里假设你之前还是用 d.t() 作为 cost matrix）
            loss = ot.emd2(codeword_weight, mixed_target, distances, numItermax=500000)
            # # ------------------------------------------------------------------------------------------------------方法二
            # --- 1. 计算当前激活概率（avg_probs）
            # temperature = 0.1  # 这个可调
            # soft_assign = F.softmax(-d / temperature, dim=1)  # 使用当前距离 d 计算 soft assignment
            # avg_probs = torch.mean(soft_assign, dim=0) + 1e-6  # 计算每个码字的平均激活概率
            # avg_probs = avg_probs / avg_probs.sum()  # 保证概率和为1
            # avg_probs = avg_probs.unsqueeze(1)
            #
            # # --- 2. 生成目标分布
            # uniform_target = torch.ones(self.num_embeddings, device=inputs.device) / self.num_embeddings
            # gaussian_target = self.generate_awgn_gaussian(inputs.device)
            # mixed_target = self.alpha * uniform_target + (1 - self.alpha) * gaussian_target
            # mixed_target = mixed_target / mixed_target.sum()  # 保证目标和为1
            # mixed_target = mixed_target.unsqueeze(1)  # 将目标分布也变为 [1, 256]
            #
            # # --- 3. 计算 Wasserstein 距离来优化激活概率
            # loss = ot.emd2(avg_probs, mixed_target, d, numItermax=500000)

            print(f"emd_Loss: {loss.item():.4f}")

        else:
            loss = 0.0

        quantized = quantized.view(inputs_shape)
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        return quantized, loss, perplexity

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

