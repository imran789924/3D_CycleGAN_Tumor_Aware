import os
import sys
import torch
import itertools
import random
from .base_model import BaseModel
from . import networks3D


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images


class CycleGANModel(BaseModel):
    def name(self):
        return 'CycleGANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0,
                                help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of '
                                                                                   'scaling the weight of the identity mapping loss. For example, if the weight of the'
                                                                                   ' identity loss should be 10 times smaller than the weight of the reconstruction loss, '
                                                                                   'please set lambda_identity = 0.1')
            '''
            adjust the weight of correlation coefficient loss
            '''
            parser.add_argument('--lambda_co_A', type=float, default=2,
                                help='weight for correlation coefficient loss (A -> B)')
            parser.add_argument('--lambda_co_B', type=float, default=2,
                                help='weight for correlation coefficient loss (B -> A )')
            parser.add_argument('--use_attention', action='store_true', help='use mask attention in generator (input must have 2 channels: image, mask)')
            parser.add_argument('--attention_strength', type=float, default=1.0, help='strength of mask attention modulation when use_attention is set')
            parser.add_argument('--lambda_tumor', type=float, default=0.0, help='weight for tumor prediction loss (UNet on fake_B vs ground-truth mask). Requires mask_dir and unet_checkpoint.')
            parser.add_argument('--lambda_bg', type=float, default=0.0, help='weight for background-preserving loss: L1(fake_B, real_A) in background (1-mask). Requires mask_dir. Use to avoid generator saturating background to white.')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        if getattr(opt, 'lambda_tumor', 0.0) > 0:
            self.loss_names.append('tumor')
        if getattr(opt, 'lambda_bg', 0.0) > 0:
            self.loss_names.append('bg')
        # self.loss_names = ['D_A', 'G_A', 'cycle_A', 'cor_coe_GA', 'D_B', 'G_B', 'cycle_B', 'cor_coe_GB']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_A')
            visual_names_B.append('idt_B')

        # Optional: pretrained frozen UNet for tumor prediction on fake_B
        self.netSeg = None
        if getattr(opt, 'unet_checkpoint', None) and os.path.isfile(opt.unet_checkpoint):
            _proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if _proj_root not in sys.path:
                sys.path.insert(0, _proj_root)
            from unet_tumor_segmentation.model import build_unet
            ckpt = torch.load(opt.unet_checkpoint, map_location=self.device)
            patch_size = ckpt.get('patch_size', getattr(opt, 'patch_size', [128, 128, 64]))
            patch_size = list(map(int, patch_size))
            self.netSeg = build_unet(patch_size=patch_size)
            state = ckpt.get('state_dict', ckpt)
            if list(state.keys())[0].startswith('module.'):
                from torch.nn import DataParallel
                self.netSeg = DataParallel(self.netSeg)
            self.netSeg.load_state_dict(state)
            self.netSeg = self.netSeg.to(self.device)
            if hasattr(self.netSeg, 'module'):
                self.netSeg = self.netSeg.module
            for p in self.netSeg.parameters():
                p.requires_grad = False
            self.netSeg.eval()
            visual_names_A.append('tumor_pred_B')
            visual_names_A.append('mask_B')  # ground-truth mask for comparison with tumor_pred_B
            print('Loaded frozen tumor UNet from', opt.unet_checkpoint)

        self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        use_att = getattr(opt, 'use_attention', False)
        att_strength = getattr(opt, 'attention_strength', 1.0)
        self.netG_A = networks3D.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                        use_attention=use_att, attention_strength=att_strength)
        self.netG_B = networks3D.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                        use_attention=use_att, attention_strength=att_strength)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks3D.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks3D.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks3D.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            # mixed precision (AMP) for faster training when CUDA is available
            self.use_amp = getattr(opt, 'use_amp', False) and torch.cuda.is_available()
            if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
                self.scaler_G = torch.amp.GradScaler('cuda', enabled=self.use_amp)
                self.scaler_D = torch.amp.GradScaler('cuda', enabled=self.use_amp)
            else:
                self.scaler_G = torch.cuda.amp.GradScaler(enabled=self.use_amp)
                self.scaler_D = torch.cuda.amp.GradScaler(enabled=self.use_amp)
            if getattr(opt, 'use_amp', False) and not torch.cuda.is_available():
                print('Warning: --use_amp ignored (CUDA not available)')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input[0 if AtoB else 1].to(self.device)
        self.real_B = input[1 if AtoB else 0].to(self.device)
        # Optional ground-truth tumor mask (same index as real_B; used for tumor loss when lambda_tumor > 0)
        self.mask_B = None
        if len(input) >= 3 and isinstance(input[2], torch.Tensor):
            self.mask_B = input[2].to(self.device)

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A = self.netG_B(self.fake_B)

        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)

        # Frozen UNet: predict tumor mask on fake_B (for monitoring and optional tumor loss)
        use_tumor_loss = getattr(self.opt, 'lambda_tumor', 0.0) > 0 and self.mask_B is not None
        if self.netSeg is not None:
            if use_tumor_loss:
                self.tumor_pred_B = self.netSeg(self.fake_B)  # keep grad for backward_G
            else:
                with torch.no_grad():
                    self.tumor_pred_B = self.netSeg(self.fake_B)

    def backward_D_basic(self, netD, real, fake, do_backward=True):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        if do_backward:
            loss_D.backward()
        return loss_D

    def backward_D_A(self, do_backward=True):
        fake_B = self.fake_B_pool.query(self.fake_B)
        if getattr(self, 'use_amp', False):
            fake_B = fake_B.float()
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B, do_backward=do_backward)

    def backward_D_B(self, do_backward=True):
        fake_A = self.fake_A_pool.query(self.fake_A)
        if getattr(self, 'use_amp', False):
            fake_A = fake_A.float()
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A, do_backward=do_backward)

    def backward_G(self):
        # When AMP is on, forward ran in autocast so generator outputs are float16; backward_G runs
        # with autocast disabled (for BCE). Cast to float32 so D and loss ops get matching dtypes.
        if getattr(self, 'use_amp', False):
            self.fake_B = self.fake_B.float()
            self.fake_A = self.fake_A.float()
            self.rec_A = self.rec_A.float()
            self.rec_B = self.rec_B.float()
            if hasattr(self, 'tumor_pred_B') and self.tumor_pred_B is not None:
                self.tumor_pred_B = self.tumor_pred_B.float()

        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        '''
        lambda_coA & lambda_coB
        '''
        lambda_co_A = self.opt.lambda_co_A
        lambda_co_B = self.opt.lambda_co_B

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)

        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        # Forward cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A

        # Backward cycle loss
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        '''
        self.cor_coeLoss
        '''
        self.loss_cor_coe_GA = networks3D.Cor_CoeLoss(self.fake_B,
                                                    self.real_A) * lambda_co_A  # fake ct & real mr; Evaluate the Generator of ct(G_A)
        self.loss_cor_coe_GB = networks3D.Cor_CoeLoss(self.fake_A,
                                                    self.real_B) * lambda_co_B  # fake mr & real ct; Evaluate the Generator of mr(G_B)

        # Tumor prediction loss (frozen UNet on fake_B vs ground-truth mask)
        lambda_tumor = getattr(self.opt, 'lambda_tumor', 0.0)
        if lambda_tumor > 0 and self.netSeg is not None and self.mask_B is not None:
            self.loss_tumor = torch.nn.functional.binary_cross_entropy(self.tumor_pred_B, self.mask_B) * lambda_tumor
        else:
            self.loss_tumor = 0.0

        # Background-preserving loss: in background (1 - mask), keep fake_B close to real_A
        lambda_bg = getattr(self.opt, 'lambda_bg', 0.0)
        if lambda_bg > 0 and self.mask_B is not None:
            w = 1.0 - self.mask_B  # background weight
            diff = torch.abs(self.fake_B - self.real_A)
            w_sum = w.sum().clamp(min=1e-6)
            self.loss_bg = (w * diff).sum() / w_sum * lambda_bg
        else:
            self.loss_bg = 0.0

        # combined loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        if isinstance(self.loss_tumor, torch.Tensor):
            self.loss_G = self.loss_G + self.loss_tumor
        if isinstance(self.loss_bg, torch.Tensor):
            self.loss_G = self.loss_G + self.loss_bg
        # self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_cor_coe_GA + self.loss_cor_coe_GB
        if not getattr(self, 'use_amp', False):
            self.loss_G.backward()

    def optimize_parameters(self):
        use_amp = getattr(self, 'use_amp', False)
        # Prefer torch.amp (PyTorch 2.1+); fallback to torch.cuda.amp
        if use_amp and hasattr(torch, 'amp'):
            _autocast = lambda enabled=True: torch.amp.autocast('cuda', enabled=enabled)
        else:
            _autocast = lambda enabled=True: torch.cuda.amp.autocast(enabled=enabled)
        # forward (autocast for speed; G/D/UNet in mixed precision)
        if use_amp:
            with _autocast(enabled=True):
                self.forward()
        else:
            self.forward()
        # G_A and G_B — backward_G outside autocast (BCE/BCELoss unsafe in autocast)
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        if use_amp:
            with _autocast(enabled=False):
                self.backward_G()
            self.scaler_G.scale(self.loss_G).backward()
            self.scaler_G.step(self.optimizer_G)
            self.scaler_G.update()
        else:
            self.backward_G()
            self.optimizer_G.step()
        # D_A and D_B — D backwards outside autocast (BCE unsafe in autocast)
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        if use_amp:
            with _autocast(enabled=False):
                self.backward_D_A(do_backward=False)
                self.backward_D_B(do_backward=False)
            self.scaler_D.scale(self.loss_D_A + self.loss_D_B).backward()
            self.scaler_D.step(self.optimizer_D)
            self.scaler_D.update()
        else:
            self.backward_D_A()
            self.backward_D_B()
            self.optimizer_D.step()
