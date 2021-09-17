import torch
import torch.nn as nn

from ..attack import Attack


class IPGD2(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]
    
    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (DEFALUT: 0.3)
        alpha (float): step size. (DEFALUT: 2/255)
        steps (int): number of steps. (DEFALUT: 40)
        random_start (bool): using random initialization of delta. (DEFAULT: False)
        
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
          
    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=False)
        >>> adv_images = attack(images, labels)
        
    """
    def __init__(self, model, mean, std, eps=0.3, alpha=2/255, steps=40, alpha2=2/255, steps2=40, random_start=False):
        super(IPGD2, self).__init__("IPGD2", model)
        
        # PGD Attack Parameter normalization
        self.lower_limit = (0.0 - mean) / std
        self.upper_limit = (1.0 - mean) / std
        self.eps = eps / std
        self.alpha = alpha / std
        self.alpha2 = alpha2 / std
        
        self.steps = steps
        self.steps2 = steps2
        self.random_start = random_start

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        labels = self._transform_label(images, labels)
        
        loss = nn.CrossEntropyLoss()

        images2 = torch.nn.functional.interpolate(images, size=(16, 16), mode='bilinear', align_corners=False)
        adv_images = images2.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=self.lower_limit, max=self.upper_limit).detach()

        for i in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            cost = self._targeted*loss(outputs, labels)

            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() - self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images2, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images2 + delta, min=self.lower_limit, max=self.upper_limit).detach()

        adv_images2 = images.clone().detach()
        
        for i in range(self.steps2):
            adv_images3 = torch.nn.functional.interpolate(adv_images2, size=(16, 16), mode='bilinear', align_corners=False)
            spert = self.alpha2*(adv_images - adv_images3).sign()
            spert = torch.nn.functional.interpolate(spert, scale_factor=2, mode='nearest')
            delta = torch.clamp(adv_images2 - images + spert, min=-self.eps, max=self.eps)
            adv_images2 = torch.clamp(images + delta, min=self.lower_limit, max=self.upper_limit).detach()
        
        return adv_images2
