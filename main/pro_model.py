import argparse
import numpy as np
import torch
import random
from models1.models import XProNet
from modules1.dataloaders import R2DataLoader
from modules1.loss import compute_loss
from modules1.metrics import compute_scores
from modules1.optimizers import build_optimizer, build_lr_scheduler
from modules1.tokenizers import Tokenizer
from modules1.trainer import Trainer
import os
from modules1.logger import create_logger
import argparse
from torchvision import transforms
from PIL import Image



def model_pro(images,selected_values):

    # fix random seeds
    seed = 7580
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    save_dir = "test_results"
    os.makedirs(save_dir, exist_ok=True)
    logger = create_logger(output_dir=save_dir, name='XPRONet')

    tokenizer = Tokenizer()


    # build model architecture
    model = XProNet(tokenizer)
    state_dict = torch.load(os.path.join("./models1/model_pro.pth"))['state_dict']
    model.load_state_dict(state_dict)
    model = model.to(device)

    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    # 将图片列表转换成张量
    images_tensor = torch.stack([transform(img.convert('RGB')) for img in images]).cuda()  # 这将创建一个维度为 (2, 3, 224, 224) 的张量

    print(selected_values)

    # 现在 images_tensor 就是你想要的形状
    print(images_tensor.shape)  # 应该打印出 torch.Size([2, 3, 224, 224])
    images_tensor_unsqueezed = images_tensor.unsqueeze(0).cuda()
    print(images_tensor_unsqueezed.shape)  # 应该打印出 torch.Size([2, 3, 224, 224])
    if isinstance(selected_values, str):
        # Assuming selected_values is a string of numbers separated by commas
        selected_values = list(map(int, selected_values.split(',')))

    # Initialize labels tensor with zeros and update based on selected_values
    labels = torch.zeros(1, 14, dtype=torch.int32).cuda()  # Create a tensor for labels
    labels[0, torch.tensor(selected_values) - 1] = 1
    print(labels)
    output, _ = model(images_tensor_unsqueezed, labels=labels, mode='sample')
    reports = model.tokenizer.decode_batch(output.cpu().numpy())
    print(reports)
    return reports