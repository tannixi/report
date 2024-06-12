from django.shortcuts import render
from PIL import Image
from main.normol import model_normal
from main.pro_model import model_pro
from main.three import three

def upload_images(request):
    context = {}
    if request.method == 'POST':
        model_choice = request.POST.get('model_choice')


        if model_choice == 'model_normal':
            # 取两张图片
            image_files = [request.FILES.get('image1'), request.FILES.get('image2')]
            images = [Image.open(img_file) for img_file in image_files]
            # 假设 model_normal 是处理图片并返回报告的函数
            reports = model_normal(images)
            context['reports'] = reports
        elif model_choice == 'model_pro':
            # 取两张图片
            image_files = [request.FILES.get('image1'), request.FILES.get('image2')]
            images = [Image.open(img_file) for img_file in image_files]
            selected_values = request.POST.get('selected_values', '')
            # 调用 model_pro 处理图片和选择的复选框值
            # 假设 model_pro 是处理专业模式的函数
            reports = model_pro(images, selected_values)
            context['reports'] = reports
        elif model_choice == 'model_third':
            # 只取第一个上传的图片
            image_file = request.FILES.get('image1')
            image = Image.open(image_file)
            reports=three(image)

            context['image_size'] = reports  # 将图片大小添加到上下文中

    return render(request, 'upload.html', context)