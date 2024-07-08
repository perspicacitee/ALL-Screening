import torch
import torch.nn as nn
import torchvision.models as models
from timm import create_model
import torch.nn.functional as F


class ResNet(nn.Module):
    """ResNet18 or ResNet34"""

    def __init__(self, model_depth, hidden_dims: list, dropout_rate=0.5):
        super(ResNet, self).__init__()

        if model_depth == 18:
            self.resnet = create_model("resnet18", pretrained=True)
        elif model_depth == 34:
            self.resnet = create_model("resnet34", pretrained=True)
        else:
            raise ValueError("model depth only support 18 & 34")

        self.in_feature = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.in_feature, hidden_dims[0])

        # process the end fc layer
        layers = []
        for in_dim, out_dim in zip(hidden_dims, hidden_dims[1:]):
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(num_features=out_dim))
        self.fc = nn.Sequential(*layers)

    def forward(self, img, tabular_data=None):
        img_repr = self.resnet(img)
        logit = self.fc(img_repr)

        return logit


def get_model(model_name: str, output_dim=512) -> nn.Module:
    """根据模型名字返回模型"""
    if model_name == "VGG16":
        model = create_model("vgg16", pretrained=True)
        model.Classifier = nn.Linear(model.Classifier.in_features, output_dim)
    elif model_name == "Xception":
        model = create_model("xception", pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, output_dim)
    elif model_name == "resnet18":
        model = create_model("resnet18", pretrained=True)
        model.fc = nn.Identity()
    elif model_name == "resnet34":
        model = create_model("resnet34", pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, output_dim)
    elif model_name == "resnet50":
        model = create_model("resnet50", pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, output_dim)
    elif model_name == "vit_base":
        model = create_model("vit_base_patch16_224", pretrained=True)
        model.head = nn.Linear(model.head.in_features, output_dim)
    elif model_name == 'efficientnet_b4':
        model = create_model("efficientnet_b4", pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, output_dim)
    elif model_name == "dense121":
        model = create_model("densenet121", pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, output_dim)
    elif model_name == "efficientnet_b0":
        model = create_model("efficientnet_b0", pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, output_dim)
    else:
        raise ValueError("model name only support ResNet18 & ResNet34")


class Acute(nn.Module):
    """同时输入图像和血常规数据"""
    def __init__(self, model_name: str, hidden_dims: list, dropout_rate=0.5):
        super().__init__()
        hidden_num = 512
        self.img_extractor = create_model(model_name, pretrained=True, num_classes=hidden_num)

        self.fc = nn.Linear(512 + 128, 3)

        self.mlp2 = nn.Sequential(
            nn.Linear(25, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 128)
        )
        self.fc1 = nn.Linear(512, 3)
        self.fc2 = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

        self.mlp_atten = nn.Sequential(
            nn.Linear(25, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 512)
        )

        self.fc_atten = nn.Linear(512, 3)
        self.weight = nn.Parameter(torch.tensor([0.5], requires_grad=True))



    def forward_concat(self, img, tabular):
        img_feature = self.img_extractor(img)
        tab_feature = self.mlp2(tabular)

        feature = torch.cat([img_feature, tab_feature], dim=1)

        return self.fc(feature)


    def forward_attention(self, img, tabular):
        img_feature = self.img_extractor(img)
        tab_feature = self.mlp_atten(tabular)

        attention = torch.sum(img_feature * tab_feature, dim=1, keepdim=True)
        attention /= torch.norm(img_feature, dim=1, keepdim=True) * torch.norm(tab_feature, dim=1, keepdim=True)
        attention = torch.sigmoid(attention)

        feature = attention * img_feature + (1 - attention) * tab_feature

        return self.fc_atten(feature)

    def forward_add(self, img, tabular):
        img_feature = self.img_extractor(img)
        img_logit = self.fc1(img_feature)

        tab_feature = self.mlp2(tabular)
        tab_logit = self.fc2(tab_feature)

        return 0.5 * img_logit + 0.5 * tab_logit

    def forward_gate(self, img, tabular):
        img_feature = self.img_extractor(img)
        img_logit = self.fc1(img_feature)

        tab_feature = self.mlp_atten(tabular)
        tab_logit = self.fc_atten(tab_feature)

        zeros = torch.zeros_like(img_logit)
        img_score = F.softmax(img_logit, dim=1)
        # tab_score = F.softmax(tab_logit, dim=1)
        mask = torch.all(img_score < 0.5, dim=1, keepdim=True).to(torch.float32)

        feature = img_feature + mask * tab_feature
        logit = self.fc_atten(feature)

        return logit

    def forward(self, img, tabular):
        # return self.forward_concat(img, tabular)
        return self.forward_add(img, tabular)
        # return self.forward_attention(img, tabular)
        # return self.forward_gate(img, tabular)


class ImageModel(nn.Module):
    """仅使用图像信息，但输出两个模型"""
    def __init__(self, model_name: str, hidden_dims: list, dropout_rate=0.5):
        super().__init__()
        hidden_num = 512

        self.img_extractor = create_model(model_name, pretrained=True, num_classes=hidden_dims[-1])

    def forward(self, img, tabular=None):
        return self.img_extractor(img)


class TabularModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(25, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        self.fc = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, img, tabular):
        return self.fc(self.mlp(tabular))



if __name__ == '__main__':
    model = Acute("ResNet18", [3]).cuda()
    img_sample = torch.randn(8, 3, 224, 224).cuda()
    tab_sample = torch.randn(8, 57).cuda()

    result = model(img_sample, tab_sample)

    loss = torch.sum(result - 0)
    loss.backward()
    print(result.shape)



