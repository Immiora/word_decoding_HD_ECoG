'''
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, drop_ratio=.0, n_enc_layers=1, bidirectional=False):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = n_enc_layers
        self.gru = nn.GRU(input_size, hidden_size,
                          batch_first=True,
                          dropout=drop_ratio,
                          num_layers=n_enc_layers,
                          bidirectional=bidirectional)

    def forward(self, input):
        output, hidden = self.gru(input)
        if self.gru.bidirectional:
            output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        return output


class Decoder(nn.Module):
    def __init__(self, output_size, drop_ratio=.0, n_dec_layers=1):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.num_layers = n_dec_layers
        self.gru = nn.GRU(output_size, output_size,
                          batch_first=True,
                          dropout=drop_ratio,
                          num_layers=self.num_layers)

    def forward(self, encoder_outputs, out_len):

        # last encoder output is the first hidden
        hidden = encoder_outputs[:, :, -1]
        input = self.initInput(hidden.shape[1], hidden.device)
        output = []
        for i in range(out_len):
            temp, hidden = self.gru(input, hidden)
            input = temp.detach()
            output.append(temp)
        return torch.cat(output)

    def initInput(self, batch_size, device):
        return torch.zeros(batch_size, 1, self.output_size, device=device)


class Attention(nn.Module):
    """
    Inputs:
        last_hidden: (batch_size, hidden_size)
        encoder_outputs: (batch_size, max_time, hidden_size)
    Returns:
        attention_weights: (batch_size, max_time)
    """
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Ua = nn.Linear(hidden_size, hidden_size, bias=False)
        # self.va = nn.Parameter(torch.FloatTensor(batch_size, hidden_size)) # why batch-dependant?
        self.va = nn.Parameter(torch.FloatTensor(hidden_size))
        torch.nn.init.normal_(self.va) # otherwise init huge values

    def forward(self, last_hidden, encoder_outputs):
        batch_size, seq_lens, _ = encoder_outputs.size()
        attention_energies = self._score(last_hidden, encoder_outputs)

        return F.softmax(attention_energies, -1)

    def _score(self, last_hidden, encoder_outputs):
        """
        Computes an attention score
        :param last_hidden: (batch_size, hidden_dim)
        :param encoder_outputs: (batch_size, max_time, hidden_dim)
        :return: a score (batch_size, max_time)
        """
        x = last_hidden.transpose(0, 1)
        out = torch.tanh(self.Wa(x).sum(1, keepdim=True) + self.Ua(encoder_outputs))
        #return out.bmm(self.va.unsqueeze(2)).squeeze(-1)
        return out.matmul(self.va)


class AttentionDecoder(nn.Module):
    def __init__(self, output_size, drop_ratio=.0, n_dec_layers=1):
        super(AttentionDecoder, self).__init__()
        self.num_layers = n_dec_layers
        self.output_size = output_size
        self.gru = nn.GRU(2*output_size, output_size,
                          batch_first=True,
                          dropout=drop_ratio,
                          num_layers=self.num_layers)
        self.attn = Attention(output_size)

    def forward(self, encoder_outputs, out_len):
        input = self.initInput(encoder_outputs.shape[0], encoder_outputs.device)
        hidden = self.initHidden(encoder_outputs.shape[0], encoder_outputs.device)

        output = []
        for i in range(out_len):
            # Calculate attention weights and apply to encoder outputs
            attn_weights = self.attn(hidden, encoder_outputs)
            context = attn_weights.unsqueeze(1).bmm(encoder_outputs)  # B x 1 x N
            concat_input = torch.cat((input, context), 2)

            temp, hidden = self.gru(concat_input, hidden)

            input = temp.detach()
            output.append(temp)

        return torch.cat(output), attn_weights

    def initInput(self, batch_size, device):
        return torch.zeros(batch_size, 1, self.output_size, device=device)

    def initHidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.output_size, device=device)


class Seq2seq(nn.Module):
    def __init__(self, input_size, output_size,
                 drop_ratio=0,
                 n_enc_layers=1,
                 n_dec_layers=1,
                 enc_bidirectional=False):
        super(Seq2seq, self).__init__()
        self.encoder = Encoder(input_size, output_size, drop_ratio, n_enc_layers, enc_bidirectional)
        #self.decoder = Decoder(output_size)
        self.decoder = AttentionDecoder(output_size, drop_ratio, n_dec_layers)


    def _forward(self, x, out_len=1):
        encoder_outputs = self.encoder(torch.transpose(x, 1, -1)) # x should be Batch x Length x Channels
        dec_output, attn_weights = self.decoder(encoder_outputs, out_len)
        return torch.transpose(dec_output, 1, -1), attn_weights

    def forward(self, x, out_len=1):
        output, _ = self._forward(x, out_len)
        return output

    def get_attn_weights(self, x, out_len=1):
        _, attn_weights = self._forward(x, out_len)
        return attn_weights

def make_model(in_dim, out_dim, **kwargs):
    model = Seq2seq(in_dim, out_dim, **kwargs)
    return model