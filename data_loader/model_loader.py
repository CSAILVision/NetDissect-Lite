import settings
import torch
import torchvision

def loadmodel(hook_fn):
    if settings.MODEL_FILE is None:
        model = torchvision.models.__dict__[settings.MODEL](pretrained=True)
    else:
        model = torchvision.models.__dict__[settings.MODEL](num_classes=settings.NUM_CLASSES)
        checkpoint = torch.load(settings.MODEL_FILE)
        model.load_state_dict(checkpoint)
    for name in settings.FEATURE_NAMES:
        model._modules.get(name).register_forward_hook(hook_fn)
    if settings.GPU:
        model.cuda()
    model.eval()
    return model