from loss import *
from torch import nn


# Base model.
class SCMM(nn.Module):
    def __init__(self, configs):
        super(SCMM, self).__init__()

        # 3-layer 1-D CNN encoder.
        self.encoder = nn.Sequential(
            # Conv1D layer_1.
            nn.Conv1d(configs.in_channels, configs.encoder_out_dim // 4, kernel_size=8, bias=False, padding=4),
            nn.BatchNorm1d(configs.encoder_out_dim // 4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),

            # Conv1D layer_2.
            nn.Conv1d(configs.encoder_out_dim // 4, configs.encoder_out_dim // 2, kernel_size=8, bias=False, padding=4),
            nn.BatchNorm1d(configs.encoder_out_dim // 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),

            # Conv1D layer_3.
            nn.Conv1d(configs.encoder_out_dim // 2, configs.encoder_out_dim, kernel_size=8, bias=False, padding=4),
            nn.BatchNorm1d(configs.encoder_out_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        # 2-layer MLP projector.
        self.projector = nn.Sequential(
            nn.Linear(configs.encoder_out_dim * (configs.fea_length + 1) // 2, configs.encoder_out_dim * 2),
            nn.BatchNorm1d(configs.encoder_out_dim * 2),
            nn.ReLU(),
            nn.Linear(configs.encoder_out_dim * 2, configs.encoder_out_dim)
        )

        self.awl = AutoWeightLoss(loss_num=2)
        self.contrastive = SoftContrastiveLoss(configs)
        self.aggregation = AggregateReconstruct(configs)

        # single-layer MLP decoder.
        self.decoder = nn.Linear(configs.encoder_out_dim * (configs.fea_length + 1) // 2,
                                 configs.in_channels * configs.fea_length)
        self.mse = ReconstructiveLoss()

    def forward(self, x_in, train_mode="pre_train"):
        if train_mode == "pre_train":
            x = self.encoder(x_in)
            h = x.reshape(x.shape[0], -1)
            z = self.projector(h)

            sim_matrix, loss_c = self.contrastive(x_in, z)
            reconstruct_weight_matrix, reconstruct_bath_emb = self.aggregation(sim_matrix, x)
            reconstruct_x = self.decoder(reconstruct_bath_emb.reshape(reconstruct_bath_emb.size(0), -1))
            loss_r = self.mse(x_in, reconstruct_x)
            loss = self.awl(loss_c, loss_r)

            return loss, loss_c, loss_r
        else:
            # Only train the emotion classifier in the fine-tuning stage.
            x = self.encoder(x_in)
            h = x.reshape(x.shape[0], -1)

            return h


# Emotion classifier.
class target_classifier(nn.Module):
    def __init__(self, configs):
        super(target_classifier, self).__init__()
        self.fc = nn.Linear(configs.encoder_out_dim * (configs.fea_length + 1) // 2, configs.out_dim)
        self.classifier = nn.Linear(configs.out_dim, configs.target_num_classes)

    def forward(self, h):
        # 2-layer MLP for classifier.
        h_nonlinear = torch.sigmoid(self.fc(h))
        pred = self.classifier(h_nonlinear)
        return pred
