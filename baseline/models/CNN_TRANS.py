import warnings
import torch.nn as nn
import torch
from models.CNN import CNN
from models.transformer import TransformerEncoderLayer, TransformerEncoder



class CNN_TRANS(nn.Module):

    def __init__(self, n_in_channel, nclass, normal_func="sparsemax", sparsity=1.3, attention=False, activation="Relu", dropout=0,
                 train_cnn=True,
                 cnn_integration=False, **kwargs):

        super(CNN_TRANS, self).__init__()

        self.n_in_channel = n_in_channel
        self.attention = attention
        self.cnn_integration = cnn_integration
        n_in_cnn = n_in_channel
        if cnn_integration:
            n_in_cnn = 1
        self.cnn = CNN(n_in_cnn, activation, dropout, **kwargs)
        if not train_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False
        self.train_cnn = train_cnn

        self.encoder_layer = TransformerEncoderLayer(d_model=128, nhead=16, normal_func=normal_func, sparsity=sparsity)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=1)

        self.dense = nn.Linear(128, nclass)

        self.sigmoid = nn.Sigmoid()
        if self.attention:
            self.dense_softmax = nn.Linear(128, nclass)
            self.softmax = nn.Softmax(dim=-1)

    def load_cnn(self, state_dict):
        self.cnn.load_state_dict(state_dict)
        if not self.train_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False

    def load_state_dict(self, state_dict, strict=True):
        self.cnn.load_state_dict(state_dict["cnn"])
        self.transformer_encoder.load_state_dict(state_dict["transformer_encoder"])
        self.dense.load_state_dict(state_dict["dense"])

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {"cnn": self.cnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      "transformer_encoder": self.transformer_encoder.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      'dense': self.dense.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)}
        return state_dict

    def save(self, filename):
        parameters = {'cnn': self.cnn.state_dict(), 'transformer_encoder': self.transformer_encoder.state_dict(), 'dense': self.dense.state_dict()}
        torch.save(parameters, filename)

    def forward(self, x, c_epoch=None):
        # input size : (batch_size, n_channels, n_frames, n_freq)
        if self.cnn_integration:
            bs_in, nc_in = x.size(0), x.size(1)
            x = x.view(bs_in * nc_in, 1, *x.shape[2:])

        # conv features
        x = self.cnn(x)
        bs, chan, frames, freq = x.size()
        if self.cnn_integration:
            x = x.reshape(bs_in, chan * nc_in, frames, freq)

        if freq != 1:
            warnings.warn(f"Output shape is: {(bs, frames, chan * freq)}, from {freq} staying freq")
            x = x.permute(0, 2, 1, 3)
            x = x.contiguous().view(bs, frames, chan * freq)
        else:
            x = x.squeeze(-1)
            x = x.permute(0, 2, 1)  # [bs, frames, chan]

        x = x.permute(1,0,2)

        # transformer features
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)

        strong = self.dense(x)  # [bs, frames, nclass]
        strong = self.sigmoid(strong)
        if self.attention:
            sof = self.dense_softmax(x)  # [bs, frames, nclass]
            sof = self.softmax(sof)
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1)   # [bs, nclass]
        else:
            weak = strong.mean(1)
        return strong, weak


if __name__ == '__main__':
    CNN_TRANS(64, 10, kernel_size=[3, 3, 3], padding=[1, 1, 1], stride=[1, 1, 1], nb_filters=[64, 64, 64],
         pooling=[(1, 4), (1, 4), (1, 4)])