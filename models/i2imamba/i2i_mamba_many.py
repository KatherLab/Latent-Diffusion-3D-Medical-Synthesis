import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from .networks import define_D, define_G, GANLoss

class I2IMamba_model(torch.nn.Module):
    def __init__(self, isTrain=True, gpu_ids=[0]) -> None:
        super().__init__()
        self.isTrain = isTrain
        self.gpu_ids = gpu_ids

        # Fixed parameters
        self.input_nc = 3
        self.output_nc = 3
        self.ngf = 64
        self.ndf = 64
        self.which_model_netG = 'i2i_mamba'
        self.which_model_netD = 'basic'
        self.n_layers_D = 3
        self.norm = 'instance'
        self.no_dropout = False
        self.init_type = 'normal'
        self.vit_name = 'I2IMamba-B_16'
        self.fineSize = 256
        self.pre_trained_path = './checkpoints/T1_T2_PD_IXI/latest_net_G.pth'
        self.pre_trained_transformer = 0
        self.pre_trained_resnet = 0
        self.lambda_f = 10.0
        self.lambda_adv = 1.0
        self.lambda_A = 10.0
        self.no_lsgan = False
        self.pool_size = 50
        self.lr = 0.0002
        self.beta1 = 0.5
        self.which_direction = "AtoB"

        # Define networks
        self.netG = define_G(
            2, self.output_nc, self.ngf, self.which_model_netG, self.vit_name,
            self.fineSize, self.pre_trained_path, self.norm, not self.no_dropout,
            self.init_type, self.gpu_ids,
            pre_trained_trans=self.pre_trained_transformer,
            pre_trained_resnet=self.pre_trained_resnet
        )

        if self.isTrain:
            use_sigmoid = self.no_lsgan
            self.netD = define_D(
                self.input_nc + self.output_nc - 1, self.ndf, self.which_model_netD, self.vit_name,
                self.fineSize, self.n_layers_D, self.norm, use_sigmoid, self.init_type, self.gpu_ids
            )
            self.fake_AB_pool = ImagePool(self.pool_size)
            self.criterionGAN = GANLoss(use_lsgan=not self.no_lsgan, tensor=torch.FloatTensor)
            self.criterionL1 = torch.nn.L1Loss()

            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.which_direction == "AtoB"
        input_A = input["A" if AtoB else "B"]
        input_B = input["B" if AtoB else "A"]
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], non_blocking=True)
            input_B = input_B.cuda(self.gpu_ids[0], non_blocking=True)
        self.input_A = input_A
        self.input_B = input_B
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG(self.real_A[:, 0:2, :, :])
        self.real_B = Variable(self.input_B)

    def test(self):
        with torch.no_grad():
            self.real_A = Variable(self.input_A)
            self.fake_B = self.netG(self.real_A[:, 0:2, :, :])
            self.real_B = Variable(self.input_B)

    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A[:, 0:2, :, :], self.fake_B), 1).data)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        real_AB = torch.cat((self.real_A[:, 0:2, :, :], self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 * self.lambda_adv
        self.loss_D.backward()

    def backward_G(self):
        fake_AB = torch.cat((self.real_A[:, 0:2, :, :], self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * self.lambda_adv

        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.lambda_A
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([
            ("G_GAN", self.loss_G_GAN.item()),
            ("G_L1", self.loss_G_L1.item()),
            ("D_real", self.loss_D_real.item()),
            ("D_fake", self.loss_D_fake.item()),
        ])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([("real_A", real_A), ("fake_B", fake_B), ("real_B", real_B)])

    def save(self, label):
        self.save_network(self.netG, "G", label, self.gpu_ids)
        self.save_network(self.netD, "D", label, self.gpu_ids)