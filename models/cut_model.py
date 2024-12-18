import numpy as np
import torch

from models.contextual import ContextualBilateralLoss, ContextualLoss
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util


class CUTModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        if hasattr(self.opt, 'use_perceptual_loss') and self.opt.use_perceptual_loss:
            self.loss_names += ['perceptual_content', 'perceptual_style']
        if hasattr(self.opt, "use_contextual_loss") and self.opt.use_contextual_loss:
            self.loss_names += ['contextual']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            if hasattr(self.opt, 'use_perceptual_loss') and self.opt.use_perceptual_loss:
                self.vgg = networks.VGGPerceptualLoss().to(self.device)
            if hasattr(self.opt, "use_contextual_loss") and self.opt.use_contextual_loss:
                # self.contextual = ContextualBilateralLoss(use_vgg=True, vgg_layers=["relu1_2", "relu2_2", "relu3_4", "relu4_4", "relu5_4"]).to(self.device)
                self.contextual = ContextualLoss(use_vgg=True, vgg_layers=["relu3_4", "relu4_4", "relu5_4"]).to(self.device)
                # self.contextual = ContextualBilateralLoss(use_vgg=True, vgg_layers=["relu3_4", "relu4_4", "relu5_4"]).to(self.device)

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])
        # import pdb
        # pdb.set_trace()
        self.fake = self.netG(self.real)

        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        self.loss_G = self.loss_G_GAN + loss_NCE_both
        if hasattr(self.opt, 'use_perceptual_loss') and self.opt.use_perceptual_loss:
            vgg_losses = self.compute_vgg_losses()
            self.loss_perceptual_content = vgg_losses["content"]
            self.loss_perceptual_style = vgg_losses["style"]
            self.loss_G += (self.loss_perceptual_content + self.loss_perceptual_style)
        if hasattr(self.opt, "use_contextual_loss") and self.opt.use_contextual_loss:
            contextual_loss = self.contextual(self.fake_B, self.real_A) + self.contextual(self.fake_B, self.real_B)
            # contextual_loss = self.contextual(self.fake_B, self.real_A)
            self.loss_contextual = self.opt.lambda_contextual * contextual_loss
            self.loss_G += self.loss_contextual
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        # import pdb
        # pdb.set_trace()
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]
        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers


    def compute_style_loss(self, feat_fake: torch.Tensor, feat_real: torch.Tensor):
        """Compute style loss using Gram matrices"""
        b, ch, h, w = feat_fake.size()
        feat_fake = feat_fake.view(b, ch, -1)
        feat_real = feat_real.view(b, ch, -1)

        # Compute Gram matrices
        gram_fake = torch.bmm(feat_fake, feat_fake.transpose(1, 2)) / (ch * h * w)
        gram_real = torch.bmm(feat_real, feat_real.transpose(1, 2)) / (ch * h * w)

        return torch.nn.functional.l1_loss(gram_fake, gram_real)

    def compute_vgg_losses(self):
        """Compute both content and style losses using VGG19 features"""
        losses = {}

        # Normalize inputs
        real_A = (self.real_A - self.vgg.mean.to(self.real_A.device)) / self.vgg.std.to(self.real_A.device)
        fake_B = (self.fake_B - self.vgg.mean.to(self.fake_B.device)) / self.vgg.std.to(self.fake_B.device)
        real_B = (self.real_B - self.vgg.mean.to(self.real_B.device)) / self.vgg.std.to(self.real_B.device)

        # VGG19-specific layer selection
        # Block indices in VGG19:
        # conv1_1(0-2), conv1_2(2-4)
        # conv2_1(4-7), conv2_2(7-9)
        # conv3_1(9-12), conv3_2(12-14), conv3_3(14-16), conv3_4(16-18)
        # conv4_1(18-21), conv4_2(21-23), conv4_3(23-25), conv4_4(25-27)
        # conv5_1(27-30), conv5_2(30-32), conv5_3(32-34), conv5_4(34-36)

        content_layers = [1, 2]  # conv4_2 for content preservation
        style_layers = [0, 1, 2, 3, 4]  # Use all blocks for style

        x_content = real_A
        x_fake = fake_B
        x_style = real_B

        content_loss = 0
        style_loss = 0

        for i, block in enumerate(self.vgg.blocks):
            x_content = block(x_content)
            x_fake = block(x_fake)
            x_style = block(x_style)

            # Content loss between real_A and fake_B
            if i in content_layers:
                content_loss += torch.nn.functional.l1_loss(x_fake, x_content)

            # Style loss between fake_B and real_B
            if i in style_layers:
                style_loss += self.compute_style_loss(x_fake, x_style)

        losses['content'] = content_loss * self.opt.lambda_perceptual_content
        losses['style'] = style_loss * self.opt.lambda_perceptual_style

        return losses
