import torch
import torchvision.models as models

def count_operations(model, input_size=(3, 224, 224)):
    # 将模型设置为评估模式
    model.eval()
    
    # 生成随机输入数据
    inputs = torch.randn(1, *input_size)
    
    # 将模型移到GPU（如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    inputs = inputs.to(device)
    
    # 定义计数器
    total_mults = 0
    total_adds = 0
    
    # 遍历模型的每一层
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            # 计算卷积层的乘法和加法数量
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size[0] * module.kernel_size[1]
            out_size = inputs.size(2) * inputs.size(3)
            total_mults += in_channels * out_channels * kernel_size * out_size
            total_adds += (in_channels * out_channels * kernel_size - 1) * out_size
        elif isinstance(module, torch.nn.Linear):
            # 计算全连接层的乘法和加法数量
            in_features = module.in_features
            out_features = module.out_features
            total_mults += in_features * out_features
            total_adds += (in_features * out_features - 1)
    
    return total_mults, total_adds

# 加载AlexNet模型
alexnet = models.alexnet()

# 计算操作数量
mults, adds = count_operations(alexnet)
print("乘法数量:", mults)
print("加法数量:", adds)


















