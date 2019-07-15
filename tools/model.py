import torch
from torch import nn

from models import r2plus1d, i3d_nl

def generate_model(opt):
    assert opt.model in [
        'r2plus1d', 'i3d_nl'
    ]
    
    if opt.model == 'r2plus1d':
        assert opt.model_depth in [18, 34]
        
        from models.r2plus1d import get_fine_tuning_parameters
        
        model = r2plus1d.create_model(
                    model_depth=opt.model_depth,
                    sample_duration=opt.sample_duration,
                    freeze_bn=opt.freeze_bn,
                    num_classes=opt.n_classes)
    
    if opt.model == 'i3d_nl':
        assert opt.model_depth in [50]
        
        from models.i3d_nl import get_fine_tuning_parameters
        
        model = i3d_nl.create_model(
                    model_depth=opt.model_depth,
                    sample_duration=opt.sample_duration,
                    freeze_bn=opt.freeze_bn,
                    num_classes=opt.n_classes)
            
    if not opt.no_cuda:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None).cuda()

        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)
            assert opt.arch == pretrain['arch']

            model.load_state_dict(pretrain['state_dict'], strict=False)
            model.module.fc = nn.Linear(model.module.fc.in_features,
                                        opt.n_finetune_classes)
            model.module.fc = model.module.fc.cuda()

            parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            return model, parameters
    else:
        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)
            assert opt.arch == pretrain['arch']

            model.load_state_dict(pretrain['state_dict'])
            model.fc = nn.Linear(model.fc.in_features,
                                        opt.n_finetune_classes)

            parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            return model, parameters

    return model, model.parameters()
