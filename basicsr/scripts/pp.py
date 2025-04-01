import torch
import matplotlib.pyplot as plt

# 假设 feature_maps 是一个 PyTorch Tensor，形状为 (b, c, h, w)
tensor = torch.randn(3,5, 60, 60)

# # 对每个特征图应用二维傅里叶变换
# # 这里假设处理单个特征图，因此我们选取第一个批次，第一个通道
# sample_feature_map = feature_maps[0, 0, :, :]
# f_transform = torch.fft.fftn(sample_feature_map)
#
# # 计算功率谱密度
# magnitude_spectrum = torch.log(torch.abs(f_transform)**2)
#
# # 可视化
# # 因为 PyTorch Tensor 不是直接可视化的，我们需要先将其转换为 NumPy 数组
# magnitude_spectrum_np = magnitude_spectrum.numpy()
# plt.imshow(magnitude_spectrum_np, cmap='gray')
# plt.title('Log Amplitude Power Spectrum')
# plt.colorbar()
# plt.show()
# plt.show()


# 假设我们有一个特征图 tensor
# feature_map = ...

# 对特征图进行傅里叶变换
import torch
import numpy as np
import matplotlib.pyplot as plt

# 假设我们有一个形状为 (b, c, h, w) 的 tensor
# tensor = ...

# 初始化用于存储每个通道径向平均的列表
radial_means = []
def radial_profile(data):
    center = tuple(np.array(data.shape) // 2)
    y, x = np.ogrid[:data.shape[0], :data.shape[1]]
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile
# 对每个通道计算平均特征图并进行傅里叶变换
for channel in range(tensor.shape[1]):
    # 计算该通道的平均特征图
    avg_feature_map = torch.mean(tensor[:, channel, :, :], dim=0)

    # 进行傅里叶变换
    f_transform = torch.fft.fftn(avg_feature_map)
    f_shift = torch.fft.fftshift(f_transform)

    # 计算功率谱并取对数
    log_power = torch.log(torch.abs(f_shift)**2 + 1)

    # 计算径向平均功率
    radial_mean = radial_profile(log_power.numpy())
    radial_means.append(radial_mean)

radial_means = np.array(radial_means)

# 计算所有通道的平均功率谱
avg_radial_means = np.mean(radial_means, axis=0)
# 绘制曲线图
freqs = np.fft.rfftfreq(tensor.shape[2]) * np.pi
frequencies = freqs

# 绘制曲线图
for channel, radial_mean in enumerate(radial_means):
    plt.plot(frequencies, radial_mean[:len(frequencies)], label=f'Channel {channel}')

# plt.plot(frequencies, avg_radial_means[:len(frequencies)], label=f'Channel {channel}')

plt.xlabel('Frequency')
plt.ylabel('Log Amplitude Power')
plt.title('Log Amplitude Power vs Frequency for each Channel')
plt.legend()
plt.show()
#下面这段加在basiclayer
# avg_feature_map = torch.mean(xx, dim=1).squeeze(0)
#
# # print(avg_feature_map.shape)
# f_transform = torch.fft.fftn(avg_feature_map)
# f_shift = torch.fft.fftshift(f_transform)
# log_power = torch.log(torch.abs(f_shift) ** 2 + 1)
# radial_mean = radial_profile(log_power.cpu().detach().numpy())
# avg_power_spectrum.append(radial_mean)
# # print(avg_power_spectrum)
# # 计算所有通道的平均功率谱
# # print(len(avg_power_spectrum))
# # avg_power_spectrum = np.mean(avg_power_spectrum, axis=0)
# # print(avg_power_spectrum.shape)
# # color = plt.cm.Blues(1 - i / len(self.blocks))
# freqs = np.fft.rfftfreq(xx.shape[2]) * np.pi
# frequencies = freqs
# # 绘制当前块后的平均功率谱
# # print("Shape of frequencies:", frequencies.shape)
# # print("Shape of radial_mean:", avg_power_spectrum.shape)
# for channel, radial_mean in enumerate(avg_power_spectrum):
#     # plt.plot(frequencies, radial_mean[:len(frequencies)], label=f'Channel {channel}')
#     plt.plot(frequencies, radial_mean[:len(frequencies)], label=f'Block {channel}')
# plt.xlabel('Frequency')
# plt.ylabel('Log Amplitude Power')
# plt.title('Log Amplitude Power vs Frequency for each Channel')
# plt.legend()
# plt.savefig('average_power_spectrum.png')
# plt.show()