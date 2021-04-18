import numpy as np
import torch
import torch.nn as nn

from networks.common.basic_blocks import PaddedConv3x3Block, nearest_upsample, conv1x1, FSEModule
from networks.predictor.utils import create_multiscale_predictor


class HRDepthDecoder(nn.Module):
    def __init__(self, chans_enc, scales=4, mobile_encoder=False, predictor=None):
        super(HRDepthDecoder, self).__init__()

        self.scales = scales
        self.mobile_encoder = mobile_encoder
        if mobile_encoder:
            self.num_ch_dec = np.array([4, 12, 20, 40, 80])
        else:
            self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.all_position = ["01", "11", "21", "31", "02", "12", "22", "03", "13", "04"]
        self.attention_position = ["31", "22", "13", "04"]
        self.non_attention_position = ["01", "11", "21", "02", "12", "03"]

        self.convs = nn.ModuleDict()
        for j in range(5):
            for i in range(5 - j):
                # upconv 0
                num_ch_in = chans_enc[i]
                if i == 0 and j != 0:
                    num_ch_in = num_ch_in // 2
                num_ch_out = num_ch_in // 2
                self.convs[f"X_{i}{j}_Conv_0"] = PaddedConv3x3Block(num_ch_in, num_ch_out)

                # X_04 upconv 1, only add X_04 convolution
                if i == 0 and j == 4:
                    num_ch_in = num_ch_out
                    num_ch_out = self.num_ch_dec[i]
                    self.convs[f"X_{i}{j}_Conv_1"] = PaddedConv3x3Block(num_ch_in, num_ch_out)

        # declare FSEModule and original module
        for index in self.attention_position:
            row = int(index[0])
            col = int(index[1])
            if mobile_encoder:
                self.convs["X_" + index + "_attention"] = FSEModule(chans_enc[row + 1] // 2, chans_enc[row]
                                                                    + self.num_ch_dec[row] * 2 * (col - 1),
                                                                    output_channel=self.num_ch_dec[row] * 2)
            else:
                self.convs["X_" + index + "_attention"] = FSEModule(chans_enc[row + 1] // 2, chans_enc[row]
                                                                    + self.num_ch_dec[row + 1] * (col - 1))
        for index in self.non_attention_position:
            row = int(index[0])
            col = int(index[1])
            if mobile_encoder:
                self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = PaddedConv3x3Block(
                    chans_enc[row] + chans_enc[row + 1] // 2 +
                    self.num_ch_dec[row] * 2 * (col - 1), self.num_ch_dec[row] * 2)
            else:
                if col == 1:
                    self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = PaddedConv3x3Block(chans_enc[row + 1] // 2 +
                                                                                     chans_enc[row],
                                                                                     self.num_ch_dec[row + 1])
                else:
                    self.convs["X_" + index + "_downsample"] = conv1x1(chans_enc[row + 1] // 2 + chans_enc[row]
                                                                       + self.num_ch_dec[row + 1] * (col - 1),
                                                                       self.num_ch_dec[row + 1] * 2)
                    self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = PaddedConv3x3Block(self.num_ch_dec[row + 1] * 2,
                                                                                     self.num_ch_dec[row + 1])

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

        self.predictor = None
        if predictor is not None:
            self.predictor = create_multiscale_predictor(predictor, self.scales,
                                                         in_chans=self.num_ch_dec[:self.scales])

    def nestConv(self, conv, high_feature, low_features):
        conv_0 = conv[0]
        conv_1 = conv[1]
        assert isinstance(low_features, list)
        high_features = [nearest_upsample(conv_0(high_feature))]
        for feature in low_features:
            high_features.append(feature)
        high_features = torch.cat(high_features, 1)
        if len(conv) == 3:
            high_features = conv[2](high_features)
        return conv_1(high_features)

    def forward(self, input_features,**kwargs):
        features = {}
        for i in range(5):
            features[f"X_{i}0"] = input_features[i]
        # Network architecture
        for index in self.all_position:
            row = int(index[0])
            col = int(index[1])

            low_features = []
            for i in range(col):
                low_features.append(features[f"X_{row}{i}"])

            # add fSE block to decoder
            if index in self.attention_position:
                features["X_" + index] = self.convs["X_" + index + "_attention"](
                    self.convs[f"X_{row + 1}{col - 1}_Conv_0"](features[f"X_{row + 1}{col - 1}"]),
                    low_features)
            elif index in self.non_attention_position:
                conv = [self.convs[f"X_{row + 1}{col - 1}_Conv_0"],
                        self.convs[f"X_{row + 1}{col - 1}_Conv_1"]]
                if col != 1 and not self.mobile_encoder:
                    conv.append(self.convs["X_" + index + "_downsample"])
                features["X_" + index] = self.nestConv(conv, features[f"X_{row + 1}{col - 1}"], low_features)

        x = features["X_04"]
        x = self.convs["X_04_Conv_0"](x)
        x = self.convs["X_04_Conv_1"](nearest_upsample(x))

        features = [x, features["X_04"], features["X_13"], features["X_22"]]

        if self.predictor is None:
            return features

        for i, f in enumerate(features):
            self.predictor(f, i, **kwargs)

        return self.predictor.compile_predictions()