import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .utils import preprocess_adj_new, preprocess_adj_new1, my_softmax


# --- Классы для НЕПРЕРЫВНЫХ данных ---

class MLPEncoder(nn.Module):
    def __init__(self, n_in, n_xdims, n_hid, n_out, adj_A, batch_size, do_prob=0., factor=True, tol=0.1):
        super(MLPEncoder, self).__init__()
        self.adj_A = nn.Parameter(Variable(torch.from_numpy(adj_A).double(), requires_grad=True))
        self.Wa = nn.Parameter(torch.zeros(n_out), requires_grad=True)
        self.fc1 = nn.Linear(n_xdims, n_hid, bias=True)
        self.fc2 = nn.Linear(n_hid, n_out, bias=True)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, inputs):
        adj_A1 = torch.sinh(3. * self.adj_A)
        adj_Aforz = preprocess_adj_new(adj_A1)
        H1 = F.relu((self.fc1(inputs)))
        x = (self.fc2(H1))
        logits = torch.matmul(adj_Aforz, x + self.Wa) - self.Wa
        return x, logits, adj_A1, self.adj_A, self.Wa


class MLPDecoder(nn.Module):
    def __init__(self, n_in_z, n_hid, n_out, **kwargs):
        super(MLPDecoder, self).__init__()
        self.out_fc1 = nn.Linear(n_in_z, n_hid, bias=True)
        self.out_fc2 = nn.Linear(n_hid, n_out, bias=True)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)

    def forward(self, input_z, origin_A, Wa):
        adj_A_inv = preprocess_adj_new1(origin_A)
        mat_z = torch.matmul(adj_A_inv, input_z + Wa) - Wa
        H3 = F.relu(self.out_fc1((mat_z)))
        out = self.out_fc2(H3)
        return mat_z, out


# --- Классы для ДИСКРЕТНЫХ данных ---

class MLPDiscreteEncoder(nn.Module):
    """ Правильный Encoder для дискретных данных. """

    def __init__(self, n_vars, n_categories, n_hid, n_out_latent, adj_A, **kwargs):
        super(MLPDiscreteEncoder, self).__init__()
        self.adj_A = nn.Parameter(Variable(torch.from_numpy(adj_A).double(), requires_grad=True))
        self.Wa = nn.Parameter(torch.zeros(n_out_latent), requires_grad=True)

        # Слой Embedding для преобразования категорий в векторы
        self.embed = nn.Embedding(n_categories, n_hid)
        self.fc1 = nn.Linear(n_hid, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out_latent)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, inputs_long):
        # inputs_long shape: [batch, n_vars]
        adj_A1 = torch.sinh(3. * self.adj_A)
        adj_Aforz = preprocess_adj_new(adj_A1)

        # [batch, n_vars] -> [batch, n_vars, n_hid]
        embedded = self.embed(inputs_long)

        h1 = F.relu(self.fc1(embedded))
        x = self.fc2(h1)  # [batch, n_vars, n_out_latent]

        logits = torch.matmul(adj_Aforz, x + self.Wa) - self.Wa

        return x, logits, adj_A1, self.adj_A, self.Wa


class MLPDiscreteDecoder(nn.Module):
    def __init__(self, n_in_z, n_hid, n_out_classes, n_vars, **kwargs):
        super(MLPDiscreteDecoder, self).__init__()
        self.n_vars = n_vars
        # Теперь MLP работает для каждой переменной отдельно
        self.out_fc1 = nn.Linear(n_in_z, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_out_classes)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)

    def forward(self, input_z, origin_A, Wa):
        adj_A_inv = preprocess_adj_new1(origin_A)
        # input_z shape: [batch, n_vars, n_in_z]
        mat_z_latent = torch.matmul(adj_A_inv, input_z + Wa) - Wa

        h1 = F.relu(self.out_fc1(mat_z_latent))
        h2 = F.relu(self.out_fc2(h1))
        logits_per_var = self.out_fc3(h2)

        probs = my_softmax(logits_per_var, axis=-1)

        return mat_z_latent, probs