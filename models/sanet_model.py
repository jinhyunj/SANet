import torch
from .base_model import BaseModel
from . import networks, SegNet
from util.image_pool import ImagePool
import itertools
import torch.nn.functional as F


class SANETModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        if is_train:
            parser.add_argument('--lambda_Syn', type=float, default=10.0, help='weight for Syn loss')
            parser.add_argument('--lambda_Fea', type=float, default=2.0, help='weight for Fea loss')
            parser.add_argument('--lambda_Sem', type=float, default=2.0, help='weight for Sem loss')
            parser.add_argument('--lambda_AE', type=float, default=5.0, help='weight for AE loss')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['G_Adv', 'G_Syn', 'G_Fea', 'G_Sem', 'G_AE', 'D_Adv']
        self.visual_names = ['I_g', 'I_g_fake', 'I_a']

        if self.isTrain:
            self.model_names = ['E_a', 'T_ag', 'G_g', 'E_g', 'D']
        else:
            self.model_names = ['E_a', 'T_ag', 'G_g']

        self.netE_a = networks.define_Enc('SAnet', opt.ngf, 4, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netT_ag = networks.define_Trans('SAnet', opt.ngf, 4, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_g = networks.define_Dec('SAnet', opt.ngf, 4, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.netE_g = networks.define_Enc('SAnet', opt.ngf, 4, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD = networks.define_D(3, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netS_seg = SegNet.SegNet(3, 5).to(self.device)
            self.netS_seg.load_state_dict(torch.load('./segnet.pth'))
            self.netS_seg.eval()

        if self.isTrain:
            self.I_g_fake_pool = ImagePool(opt.pool_size)
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netE_a.parameters(), self.netT_ag.parameters(), self.netG_g.parameters(), self.netE_g.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.I_a = input['A'].to(self.device)
        self.I_g = input['G'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        self.f_a = self.netE_a(self.I_a)
        self.f_ag_1, self.f_ag_2, self.f_ag_3, self.f_ag_4 = self.netT_ag(self.f_a)
        self.f_ag = self.f_ag_1 + self.f_ag_2 + self.f_ag_3 + self.f_ag_4
        self.I_g_fake = self.netG_g(self.f_ag)

        if self.isTrain:
            self.f_g = self.netE_g(self.I_g)
            self.I_g_rec = self.netG_g(self.f_g)

            self.S_g_pred, self.S_g_pred_softmaxed = self.netS_seg(self.I_g)
            self.S_g_fake_pred, _ = self.netS_seg(self.I_g_fake)

    def semantic_aware_syn_loss(self):
        b, _, h, w = self.I_g_fake.size()
        S_g = self.S_g_pred_softmaxed.argmax(axis=1)

        self.S_g_1 = torch.zeros((b, h, w)).to(self.device)
        self.S_g_2 = torch.zeros((b, h, w)).to(self.device)
        self.S_g_3 = torch.zeros((b, h, w)).to(self.device)
        self.S_g_4 = torch.zeros((b, h, w)).to(self.device)

        self.S_g_1[S_g == 0] = 1
        self.S_g_2[S_g == 1] = 1
        self.S_g_3[S_g == 2] = 1
        self.S_g_4[S_g == 3] = 1

        N_1 = torch.count_nonzero(self.S_g_1)
        N_2 = torch.count_nonzero(self.S_g_2)
        N_3 = torch.count_nonzero(self.S_g_3)
        N_4 = torch.count_nonzero(self.S_g_4)

        diff = torch.abs(self.I_g_fake-self.I_g)

        loss_1 = torch.sum(diff*self.S_g_1.unsqueeze(1))/(N_1+1)
        loss_2 = torch.sum(diff*self.S_g_2.unsqueeze(1))/(N_2+1)
        loss_3 = torch.sum(diff*self.S_g_3.unsqueeze(1))/(N_3+1)
        loss_4 = torch.sum(diff*self.S_g_4.unsqueeze(1))/(N_4+1)

        return 0.5*loss_1 + 2*loss_2 + loss_3 + loss_4

    def semantic_aware_fea_loss(self):
        b, _, h, w = self.f_g.size()

        S_g_1_ds = F.interpolate(self.S_g_1.unsqueeze(1), size=(h,w))
        S_g_2_ds = F.interpolate(self.S_g_2.unsqueeze(1), size=(h,w))
        S_g_3_ds = F.interpolate(self.S_g_3.unsqueeze(1), size=(h,w))
        S_g_4_ds = F.interpolate(self.S_g_4.unsqueeze(1), size=(h,w))

        N_1 = torch.count_nonzero(S_g_1_ds)
        N_2 = torch.count_nonzero(S_g_2_ds)
        N_3 = torch.count_nonzero(S_g_3_ds)
        N_4 = torch.count_nonzero(S_g_4_ds)

        loss_1 = torch.sum(torch.abs(self.f_ag_1-self.f_g)*S_g_1_ds)/(N_1+1)
        loss_2 = torch.sum(torch.abs(self.f_ag_2-self.f_g)*S_g_2_ds)/(N_2+1)
        loss_3 = torch.sum(torch.abs(self.f_ag_3-self.f_g)*S_g_3_ds)/(N_3+1)
        loss_4 = torch.sum(torch.abs(self.f_ag_4-self.f_g)*S_g_4_ds)/(N_4+1)

        return 0.5*loss_1 + 2*loss_2 + loss_3 + loss_4

    def backward_D(self):
        # Real
        pred_real = self.netD(self.I_g)
        loss_D_real = self.criterionGAN(pred_real, True)

        # Fake
        I_g_fake = self.I_g_fake_pool.query(self.I_g_fake)
        pred_fake = self.netD(I_g_fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        self.loss_D_Adv = (loss_D_fake + loss_D_real) * 0.5
        self.loss_D_Adv.backward()

    def backward_G(self):
        lambda_Syn = self.opt.lambda_Syn
        lambda_Fea = self.opt.lambda_Fea
        lambda_Sem = self.opt.lambda_Sem
        lambda_AE = self.opt.lambda_AE

        # loss for images
        self.loss_G_Adv = self.criterionGAN(self.netD(self.I_g_fake), True)
        self.loss_G_Syn = self.semantic_aware_syn_loss()*lambda_Syn
        self.loss_G_AE = self.criterionL1(self.I_g_rec, self.I_g) * lambda_AE

        # loss for segmentations
        self.loss_G_Sem = self.criterionL1(self.S_g_fake_pred, self.S_g_pred) * lambda_Sem

        # loss for features
        self.loss_G_Fea = self.semantic_aware_fea_loss()*lambda_Fea

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_Adv + self.loss_G_Syn + self.loss_G_AE + self.loss_G_Sem + self.loss_G_Fea
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
