import torch
import math

# 两个向量的普通余弦相似度
a = torch.tensor([1, 2, 3], dtype=torch.float32)
b = torch.tensor([1, 2, 3], dtype=torch.float32)
cos_alpha = a@b / (torch.sqrt(torch.sum(torch.pow(a, 2))) * torch.sqrt(torch.sum(torch.pow(b, 2))))

print(torch.pow(a, 2))  # tensor([1., 4., 9.])
print(torch.sum(torch.pow(a, 2)))  # tensor(14.))
print(torch.sqrt(torch.sum(torch.pow(a, 2))))  # tensor(3.7417)
print(a @ b)  # tensor(14.)
print(cos_alpha)  # tensor(1.0000)  -- 相似度值
print(torch.acos(cos_alpha))  # tensor(0.0003)  -- 转弧度
print(math.degrees(torch.acos(cos_alpha)))  # 0.01978234059262607  -- 转角度
print("=============")

# 两个向量均值化后的余弦相似度
a_b = torch.cat((a, b), dim=0)  # tensor([1., 2., 3., 1., 2., 3.])
print(a_b)
min_value = torch.min(a_b)
max_value = torch.max(a_b)
print(min_value)  # tensor(1.)
print(max_value)  # tensor(3.)

mean_value = (max_value + min_value) / 2
print(mean_value)  # tensor(2.)

c = (a - mean_value)  # tensor([-1.,  0.,  1.])
d = (b - mean_value)  # tensor([-1.,  0.,  1.])
print(c)
print(d)

cos_beta = c@d / (torch.sqrt(torch.sum(torch.pow(c, 2))) * torch.sqrt(torch.sum(torch.pow(d, 2))))
print(c@d)  # tensor(2.)
print(cos_beta)  # tensor(1.0000)

cos_beta = torch.floor(cos_beta) if cos_beta >= 1 else cos_beta
print(cos_beta)  # tensor(1.)
print(torch.acos(torch.tensor(1, dtype=torch.float32)))  # tensor(0.)
print(torch.acos(cos_beta))  # tensor(0.)
print(math.degrees(torch.acos(cos_beta)))  # 0.0



