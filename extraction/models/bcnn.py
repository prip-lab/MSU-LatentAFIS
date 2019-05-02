import os
import torch
import torchvision

class BCNN(torch.nn.Module):
    """B-CNN for CUB200.
    The B-CNN model is illustrated as follows.
    conv1^2 (64) -> pool1 -> conv2^2 (128) -> pool2 -> conv3^3 (256) -> pool3
    -> conv4^3 (512) -> pool4 -> conv5^3 (512) -> bilinear pooling
    -> sqrt-normalize -> L2-normalize -> fc (200).
    The network accepts a 3*448*448 input, and the pool5 activation has shape
    512*28*28 since we down-sample 5 times.
    Attributes:
        features, torch.nn.Module: Convolution and pooling layers.
        fc, torch.nn.Module: 200.
    """
    def __init__(self, nclasses=0):
        """Declare all needed layers."""
        torch.nn.Module.__init__(self)
        # Convolution and pooling layers of VGG-16.
        self.features = torchvision.models.vgg16(pretrained=False).features
        self.features = torch.nn.Sequential(*list(self.features.children())
                                            [:-1])  # Remove pool5.

        self.nclasses = nclasses
        # Linear classifier.
        self.fc = torch.nn.Linear(512**2, nclasses)

    def forward(self, X, features=False):
        """Forward pass of the network.
        Args:
            X, torch.autograd.Variable of shape N*3*448*448.
        Returns:
            Score, torch.autograd.Variable of shape N*200.
        """
        N = X.size()[0]
        assert X.size() == (N, 3, 448, 448)
        X = self.features(X)
        assert X.size() == (N, 512, 28, 28)
        X = X.view(N, 512, 28**2)
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (28**2)  # Bilinear
        assert X.size() == (N, 512, 512)
        X = X.view(N, 512**2)
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)
        if self.nclasses > 0:
            if features is True:
                return [X]
            else:
                Y = self.fc(X)
                assert Y.size() == (N, self.nclasses)
                return [X, Y]
        else:
            return [X]

def bcnn(pretrained=False, **kwargs):
    """Constructs a BCNN model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BCNN(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['bcnn']))
    return model