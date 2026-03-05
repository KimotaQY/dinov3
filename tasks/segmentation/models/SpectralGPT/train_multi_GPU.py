import torch

from .src.models_vit_tensor_CD_2 import vit_base_patch8
from .utils.pos_embed import interpolate_pos_embed


def create_model(nb_classes, weight_path, pretrain=False):
    model = vit_base_patch8(num_classes=nb_classes)
    # model = vit_large_patch8(num_classes=nb_classes)

    if pretrain:
        checkpoint = torch.load(weight_path, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % weight_path)
        checkpoint_model = checkpoint['model']
        # checkpoint_model = checkpoint
        state_dict = model.state_dict()
        # for k in ['pos_embed','patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias']:
        #     if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        #         print(f"Removing key {k} from pretrained checkpoint")
        #         del checkpoint_model[k]
        for k in [
                'pos_embed', 'patch_embed.proj.weight',
                'patch_embed.proj.bias', 'head.weight', 'head.bias',
                'pos_embed_spatial'
        ]:
            if k in checkpoint_model and checkpoint_model[
                    k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        # model.load_state_dict(checkpoint_model, strict=False)
        # print(model)
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
    return model
