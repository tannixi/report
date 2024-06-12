
import torch
# Create your views here.
from PIL import Image
from modules.tokenizers import Tokenizer
from models.models import BaseCMNModel
from torchvision import transforms
import os
import numpy as np
def handle_uploaded_file(f):
    image = Image.open(f)
    return image.size  # 返回图片大小（宽，高）
def model_normal(images):

    torch.manual_seed(9233)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(9233)
    # create tokenizer
    tokenizer = Tokenizer()


    model = BaseCMNModel(tokenizer).to('cuda')
    state_dict = torch.load(os.path.join('models/normlo_model.pth'))['state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # 将图片列表转换成张量
    images_tensor = torch.stack([transform(img.convert('RGB')) for img in images]).cuda()  # 这将创建一个维度为 (2, 3, 224, 224) 的张量

    # 现在 images_tensor 就是你想要的形状
    print(images_tensor.shape)  # 应该打印出 torch.Size([2, 3, 224, 224])
    images_tensor_unsqueezed = images_tensor.unsqueeze(0).cuda()
    print(images_tensor_unsqueezed.shape)  # 应该打印出 torch.Size([2, 3, 224, 224])

    output, _ = model(images_tensor_unsqueezed, mode='sample')
    reports = model.tokenizer.decode_batch(output.cpu().numpy())
    print(reports)
    return reports
