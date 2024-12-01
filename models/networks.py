import torch
import models.archs.DMFourLLIE as DMFourLLIE

# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'DMFourLLIE':
        netG = DMFourLLIE.DMFourLLIE(y_nf=opt_net['latt_nf'], f_nf=opt_net['first_stage_nf'], s_nf=opt_net['second_stage_nf'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG

