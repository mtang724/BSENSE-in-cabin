import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models


def weight_init(m):
    """Initialize weights of a model.
        Borrowed from https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


class ResNet18(nn.Module):
    """Model to predict x and y flow from radar heatmaps.
    Based on ResNet18 "Deep Residual Learning for Image Recognition" https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, num_channels):
        super(ResNet18, self).__init__()

        # self.range_flag = range_flag

        # CNN encoder for heatmaps
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.conv1 = nn.Conv2d(
            num_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        self.resnet18.fc = nn.Linear(512, 2)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            m.apply(weight_init)

    def forward(self, input):
        # ranges   = input['range']
        # heatmaps = input['radar1']
        input = input.unsqueeze(1)
        # print(input.shape)
        out = self.resnet18(input)
        # out = torch.sigmoid(out)
        # if self.training and self.range_flag:
        #     flow_x = torch.arctan(heatmaps_enc[:,0] / ranges[:,0].clamp(0.1, ))
        #     flow_y = torch.arctan(heatmaps_enc[:,1] / ranges[:,0].clamp(0.1, ))
        #     flow = torch.stack((flow_x, flow_y), -1)
        # else:
        #     flow = heatmaps_enc
        return out


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet3D(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet3D, self).__init__()
        self.in_planes = 64

        # Adjusted for 1 input channel
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.linear_res = nn.Linear(128 * block.expansion, num_classes)
        self.linear = nn.Linear(128 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, mask):
        mask_adjusted = (
            mask.unsqueeze(0).unsqueeze(0).unsqueeze(2)
        )  # Now [1, 1, 1, 12, 12]
        mask_adjusted = torch.sigmoid(mask_adjusted)
        # Apply the mask
        x = x * mask_adjusted

        # Proceed with the model's computations
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out1 = self.linear(out)
        out2 = self.linear_res(out)
        return out, out1, out2


# Example of building a ResNet-18 like 3D model
def ResNet18_3D():
    return ResNet3D(BasicBlock3D, [2, 2, 2, 2], num_classes=2)


class AoA_AoD_Model(nn.Module):
    def __init__(self):
        super(AoA_AoD_Model, self).__init__()
        # Convolutional layers for spatial feature extraction
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1
        )

        # LSTM layer for temporal features
        self.lstm = nn.LSTM(
            input_size=64 * 16 * 16, hidden_size=128, num_layers=1, batch_first=True
        )

        # Dense layers for prediction
        self.fc1 = nn.Linear(128, 128)  # Adjust the input size accordingly
        self.fc2 = nn.Linear(128, 1)  # Assuming a single regression output

    def forward(self, x, mask):
        mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        # print("mask:", mask.shape)
        # print(x.shape)
        # mask = torch.sigmoid(mask)
        x = x * mask
        # print(x.shape)
        batch_size, _, H, W, T = x.size()

        # Initialize a placeholder for the convolved features
        conv_features = torch.zeros(batch_size, T, 64 * 16 * 16).to(x.device)

        # Process each temporal step
        for t in range(T):
            # Extract spatial features for each temporal slice
            xt = F.relu(self.conv1(x[:, :, :, :, t]))
            xt = F.relu(self.conv2(xt))
            xt = xt.view(batch_size, -1)  # Flatten
            conv_features[:, t, :] = xt

        # Temporal processing
        lstm_out, _ = self.lstm(conv_features)

        # Fully connected layers
        out = F.relu(
            self.fc1(lstm_out[:, -1, :])
        )  # Use the output of the last time step
        # print("out:", out.shape)
        x = self.fc2(out)

        return out, x


class AttentionMechanism(nn.Module):
    def __init__(self, feature_dim, intermediate_dim=64):
        super(AttentionMechanism, self).__init__()
        self.attention_fc1 = nn.Linear(feature_dim, intermediate_dim)
        self.attention_fc2 = nn.Linear(intermediate_dim, 1)

    def forward(self, x1, x2):
        # Combine x1 and x2 in some way - here we simply concatenate them
        combined = torch.cat((x1, x2), dim=1)

        # Pass through attention network
        attention_weights = F.relu(self.attention_fc1(combined))
        attention_weights = torch.sigmoid(self.attention_fc2(attention_weights))

        # Normalize weights
        attention_weights = F.softmax(attention_weights, dim=1)

        # Apply attention weights
        attention_applied = x1 * attention_weights + x2 * (1 - attention_weights)

        return attention_applied


class CombinedModelWithMask(nn.Module):
    def __init__(
        self,
        resnet_model,
        aoa_aod_model,
        feature_dim,
        seat_output_dim,
        resp_output_dim,
        intermediate_dim=64,
    ):
        super(CombinedModelWithMask, self).__init__()
        # Existing components...
        self.shared_mask = nn.Parameter(torch.ones(12, 12), requires_grad=True)
        self.resnet_model = resnet_model
        self.aoa_aod_model = aoa_aod_model
        self.attention = AttentionMechanism(feature_dim * 2, intermediate_dim)

        # New components: Decoders for seat and respiration prediction
        self.seat_decoder = SeatDecoder(
            feature_dim, seat_output_dim
        )  # Adjust input/output dimensions as needed
        self.respiration_decoder = RespirationDecoder(
            feature_dim, resp_output_dim
        )  # Adjust input/output dimensions as needed

    def forward(self, x1, x2):
        # Assume x is input data that can be fed into both models
        # thresholded_mask = (self.shared_mask >= 0.5).float()
        # Get outputs from both models

        out_3d, _, _ = self.resnet_model(
            x1, self.shared_mask
        )  # Assuming modification to accept mask
        out_aoa_aod, _ = self.aoa_aod_model(
            x2, self.shared_mask
        )  # Assuming modification to accept mask

        # Combine outputs using attention
        combined_output = self.attention(out_3d, out_aoa_aod)

        # Decode for seat and respiration predictions
        seat_pred = self.seat_decoder(combined_output)
        resp_pred = self.respiration_decoder(combined_output)

        return seat_pred, resp_pred


class SeatDecoder1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SeatDecoder1, self).__init__()
        # Define the architecture for seat prediction
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        # self.fc4 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(
            16, output_dim
        )  # Assuming a specific output dimension for seat prediction
        # self.output_layers = nn.ModuleList([nn.Linear(16, 3) for _ in range(2)])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        # x = [output_layers(x) for output_layers in self.output_layers]
        return torch.sigmoid(x)


class SeatDecoder2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SeatDecoder2, self).__init__()
        # Define the architecture for seat prediction
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        # self.fc4 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(
            16, output_dim
        )  # Assuming a specific output dimension for seat prediction
        # self.output_layers = nn.ModuleList([nn.Linear(16, 3) for _ in range(2)])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        # x = [output_layers(x) for output_layers in self.output_layers]
        return torch.sigmoid(x)


class RespirationDecoder1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RespirationDecoder1, self).__init__()
        # Define the architecture for respiration prediction
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 16)
        self.fc5 = nn.Linear(
            16, output_dim
        )  # Assuming a specific output dimension for respiration prediction

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class RespirationDecoder2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RespirationDecoder2, self).__init__()
        # Define the architecture for respiration prediction
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 16)
        self.fc5 = nn.Linear(
            16, output_dim
        )  # Assuming a specific output dimension for respiration prediction

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    self.expansion * out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetDoppler(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNetDoppler, self).__init__()
        self.in_channels = 16

        self.conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        # Additional layers can be added here
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128 * block.expansion, num_classes)
        self.fc2 = nn.Linear(128 * block.expansion, 1)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out1 = self.fc(out)
        out2 = self.fc2(out)
        return out, out1, out2


class CombinedModel(nn.Module):
    def __init__(
        self,
        resnet_model,
        doppler_model,
        feature_dim,
        seat_output_dim,
        resp_output_dim,
        intermediate_dim=64,
    ):
        super(CombinedModel, self).__init__()
        # Existing components...
        self.shared_mask = nn.Parameter(torch.ones(16, 16), requires_grad=True)
        self.resnet_model = resnet_model
        self.doppler_model = doppler_model
        self.attention = AttentionMechanism(feature_dim * 2, intermediate_dim)

        # Optional: Feature normalization layers before fusion
        self.norm_resnet = nn.BatchNorm1d(feature_dim)
        self.norm_doppler = nn.BatchNorm1d(feature_dim)

        self.seat1Embeding = nn.Linear(feature_dim * 2, feature_dim * 2)
        self.seat2Embeding = nn.Linear(feature_dim * 2, feature_dim * 2)

        # New components: Decoders for seat and respiration prediction
        self.seat_decoder1 = SeatDecoder1(
            feature_dim * 2, seat_output_dim
        )  # Adjust input/output dimensions as needed
        self.respiration_decoder1 = RespirationDecoder2(
            feature_dim * 2, resp_output_dim
        )  # Adjust input/output dimensions as needed

        self.seat_decoder2 = SeatDecoder1(
            feature_dim * 2, seat_output_dim
        )  # Adjust input/output dimensions as needed
        self.respiration_decoder2 = RespirationDecoder2(
            feature_dim * 2, resp_output_dim
        )  # Adjust input/output dimensions as needed

    def forward(self, x1, x2):
        # Assume x is input data that can be fed into both models
        # thresholded_mask = (self.shared_mask >= 0.5).float()
        # Get outputs from both models
        # self.shared_mask = nn.Parameter(torch.ones(12, 12), requires_grad=True)
        (
            out_3d,
            _,
        ) = self.resnet_model(
            x1, self.shared_mask
        )  # Assuming modification to accept mask
        out_aoa_aod, _, _ = self.doppler_model(
            x2
        )  # Assuming modification to accept mask

        # Combine outputs using attention
        combined_output = self.attention(
            self.norm_resnet(out_3d), self.norm_doppler(out_aoa_aod)
        )

        combined_output = torch.cat(
            [self.norm_resnet(out_3d), self.norm_doppler(out_aoa_aod)], dim=1
        )  # Example: concatenation

        combined_output1 = self.seat1Embeding(combined_output)
        combined_output2 = self.seat2Embeding(combined_output)
        # Decode for seat and respiration predictions
        seat_pred1 = self.seat_decoder1(combined_output1)
        resp_pred1 = self.respiration_decoder1(combined_output1)

        seat_pred2 = self.seat_decoder2(combined_output2)
        resp_pred2 = self.respiration_decoder2(combined_output2)

        return seat_pred1, resp_pred1, seat_pred2, resp_pred2


class SeatDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SeatDecoder, self).__init__()
        # Define the architecture for seat prediction
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        # self.fc4 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(
            16, output_dim
        )  # Assuming a specific output dimension for seat prediction

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class RespirationDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RespirationDecoder, self).__init__()
        # Define the architecture for respiration prediction
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        # self.fc4 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(
            16, output_dim
        )  # Assuming a specific output dimension for respiration prediction

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        x = self.fc4(x)
        return x


class CombinedModelOneDecoder(nn.Module):
    def __init__(
        self,
        resnet_model,
        doppler_model,
        feature_dim,
        seat_output_dim,
        resp_output_dim,
        intermediate_dim=64,
    ):
        super(CombinedModelOneDecoder, self).__init__()
        # Existing components...
        self.shared_mask = nn.Parameter(torch.ones(16, 16), requires_grad=True)
        self.resnet_model = resnet_model
        self.doppler_model = doppler_model
        self.attention = AttentionMechanism(feature_dim * 2, intermediate_dim)

        # Optional: Feature normalization layers before fusion
        self.norm_resnet = nn.BatchNorm1d(feature_dim)
        self.norm_doppler = nn.BatchNorm1d(feature_dim)

        # New components: Decoders for seat and respiration prediction
        self.seat_decoder = SeatDecoder(
            feature_dim * 2, seat_output_dim
        )  # Adjust input/output dimensions as needed
        self.respiration_decoder = RespirationDecoder(
            feature_dim * 2, resp_output_dim
        )  # Adjust input/output dimensions as needed

    def forward(self, x1, x2):
        # Assume x is input data that can be fed into both models
        # thresholded_mask = (self.shared_mask >= 0.5).float()
        # Get outputs from both models
        # self.shared_mask = nn.Parameter(torch.ones(12, 12), requires_grad=True)
        out_3d, _ = self.resnet_model(
            x1, self.shared_mask
        )  # Assuming modification to accept mask
        out_aoa_aod, _, _ = self.doppler_model(
            x2
        )  # Assuming modification to accept mask

        # Combine outputs using attention
        combined_output = self.attention(
            self.norm_resnet(out_3d), self.norm_doppler(out_aoa_aod)
        )

        combined_output = torch.cat(
            [self.norm_resnet(out_3d), self.norm_doppler(out_aoa_aod)], dim=1
        )  # Example: concatenation

        # Decode for seat and respiration predictions
        seat_pred = self.seat_decoder(combined_output)
        resp_pred = self.respiration_decoder(combined_output)

        return seat_pred, resp_pred


class QKVAttention(nn.Module):
    def __init__(self, feature_dim, intermediate_dim=64):
        super(QKVAttention, self).__init__()
        self.query = nn.Linear(feature_dim, intermediate_dim)
        self.key = nn.Linear(feature_dim, intermediate_dim)
        self.value = nn.Linear(feature_dim, intermediate_dim)
        self.intermediate_dim = intermediate_dim

    def forward(self, x1, x2):
        Q = self.query(x1)
        K = self.key(x2)
        V = self.value(x2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (
            self.intermediate_dim**0.5
        )
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_probs, V)

        return attention_output


class CombinedModelAtt(nn.Module):
    def __init__(
        self,
        resnet_model,
        doppler_model,
        feature_dim,
        seat_output_dim,
        resp_output_dim,
        intermediate_dim=64,
    ):
        super(CombinedModelAtt, self).__init__()
        self.shared_mask = nn.Parameter(torch.ones(12, 12), requires_grad=True)
        self.resnet_model = resnet_model
        self.doppler_model = doppler_model

        # Replace AttentionMechanism with QKVAttention
        self.attention = QKVAttention(feature_dim, intermediate_dim)

        # Optional: Feature normalization layers before fusion
        self.norm_resnet = nn.BatchNorm1d(feature_dim)
        self.norm_doppler = nn.BatchNorm1d(feature_dim)

        # Adjusting dimensions for decoders as necessary
        # The output dimension of QKVAttention depends on how it's implemented.
        # Assuming it outputs a tensor with the same dimension as the inputs,
        # adjust the input dimensions of your decoders accordingly.
        self.seat_decoder = SeatDecoder(intermediate_dim, seat_output_dim)
        self.respiration_decoder = RespirationDecoder(intermediate_dim, resp_output_dim)

    def forward(self, x1, x2):
        out_3d, _, _ = self.resnet_model(
            x1, self.shared_mask
        )  # Adjust as necessary for actual model outputs
        out_aoa_aod, _, _ = self.doppler_model(x2)

        # Normalize features before combining
        norm_out_3d = self.norm_resnet(out_3d)
        norm_out_aoa_aod = self.norm_doppler(out_aoa_aod)

        # Combine outputs using QKV attention
        combined_output = self.attention(norm_out_3d, norm_out_aoa_aod)

        # Decode for seat and respiration predictions
        seat_pred = self.seat_decoder(combined_output)
        resp_pred = self.respiration_decoder(combined_output)

        return seat_pred, resp_pred


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == "mean":
            return torch.mean(F_loss)
        elif self.reduction == "sum":
            return torch.sum(F_loss)
        else:
            return F_loss
