from medclip import MedCLIPModel, MedCLIPVisionModelViT
from medclip import MedCLIPProcessor
from medclip import PromptClassifier


def three(image):
    processor = MedCLIPProcessor() #预处理输入数据
    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)  #视觉模型部分
    model.from_pretrained()   #加载预训练的模型权重。
    clf = PromptClassifier(model, ensemble=True)
    clf.cuda()

    # prepare input image
    from PIL import Image
    inputs = processor(images=image, return_tensors="pt")
    from medclip.prompts import generate_chexpert_class_prompts, process_class_prompts

    cls_prompts = process_class_prompts(generate_chexpert_class_prompts(n=5))
    inputs['prompt_inputs'] = cls_prompts

    output = clf(**inputs)
    # Assuming 'output' is your dictionary containing logits and class_names
    logits_tensor = output['logits'][0]  # Accessing the nested tensor and getting the first (and only) element
    class_names = output['class_names']

    # Pair each logit with its corresponding class name
    logit_class_pairs = zip(logits_tensor.tolist(), class_names)

    # # Print each pair
    # for logit, class_name in logit_class_pairs:
    #     print(f"{class_name}: {logit}")

    final_output = ""
    for logit, class_name in logit_class_pairs:
        final_output = f"{class_name}: {logit}"

    return final_output
