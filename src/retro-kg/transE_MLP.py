import numpy as np
import torch
import torch.nn as nn


class Dense(nn.Module):
    """
    Dense layer with activation function.

    Args:
        in_features (int): input feature size
        out_features (int): output feature size
        hidden_act (nn.Module): activation function (e.g. nn.ReLU())
    """

    def __init__(self, in_features, out_features, hidden_act):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.hidden_act = hidden_act

    def forward(self, x):
        return self.hidden_act(self.linear(x))


class TransE(nn.Module):
    def __init__(
        self,
        n_templates,
        device,
        norm=2,
        fp_dim=2048,
        dropout=0.3,
        hidden_sizes=[512, 512],
        hidden_activation=nn.ReLU(),
        output_dim=64,
        margin=1.0,
    ):
        super(TransE, self).__init__()
        self.relation_count = n_templates
        self.device = device
        self.norm = norm
        self.fp_dim = fp_dim
        self.emb_dim = output_dim
        self.criterion = nn.MarginRankingLoss(margin=margin, reduction="none")
        self.relations_emb = self._init_relation_emb()
        self.layers = self._build_layers(
            fp_dim, output_dim, hidden_sizes, hidden_activation
        )
        self.dropout = nn.Dropout(dropout)

    def _build_layers(self, fp_size, output_dim, hidden_sizes, hidden_activation):
        layers = nn.ModuleList(
            [Dense(fp_size, hidden_sizes[0], hidden_act=hidden_activation)]
        )

        for layer_i in range(len(hidden_sizes) - 1):
            in_features = hidden_sizes[layer_i]
            out_features = (
                hidden_sizes[layer_i + 1]
                if layer_i < len(hidden_sizes) - 2
                else output_dim
            )
            layer = Dense(in_features, out_features, hidden_act=hidden_activation)
            layers.append(layer)

        return layers

    def _init_relation_emb(self):
        relations_emb = nn.Embedding(
            num_embeddings=self.relation_count,
            embedding_dim=self.emb_dim,
        )
        uniform_range = 6 / np.sqrt(self.emb_dim)
        relations_emb.weight.data.uniform_(-uniform_range, uniform_range)
        relations_emb.weight.data = nn.functional.normalize(
            relations_emb.weight.data, p=2, dim=1
        )
        return relations_emb

    def run_layers(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)
        return nn.functional.normalize(x, p=2, dim=1)

    def forward(self, positive_triplets, negative_triplets):
        """ """
        pos_head, pos_tail, pos_rel = positive_triplets
        neg_head, neg_tail, neg_rel = negative_triplets
        positive_distances = self.distance(pos_head, pos_tail, pos_rel)
        negative_distances = self.distance(neg_head, neg_tail, neg_rel)

        return (
            self.loss(positive_distances, negative_distances),
            positive_distances,
            negative_distances,
        )

    def loss(self, positive_distances, negative_distances):
        target = torch.tensor([-1], dtype=torch.long, device=self.device)
        return self.criterion(positive_distances, negative_distances, target)

    def distance(self, heads, tails, relations):
        """ """
        heads = self.run_layers(heads)
        tails = self.run_layers(tails)
        relations = self.relations_emb(relations)
        return (heads + relations - tails).norm(p=self.norm, dim=1)
