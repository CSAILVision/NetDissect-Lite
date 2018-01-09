import settings
import torch
import torchvision

def loadmodel(hook_fn):
    def register_hook_model(model=None,feature_names=None):
        if feature_names is not None:
            for name in feature_names:
                model._modules.get(name).register_forward_hook(hook_fn)

    model = torchvision.models.__dict__[settings.MODEL](num_classes=settings.NUM_CLASSES)
    checkpoint = torch.load(settings.MODEL_FILE)
    model.load_state_dict(checkpoint)
    register_hook_model(model, settings.FEATURE_NAMES)
    model.eval()
    return model