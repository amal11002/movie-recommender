import torch
import torch.nn as nn

class NeuMF(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=32, mlp_layers=[64, 32, 16]):
        super(NeuMF, self).__init__()

        # --- Branche GMF ---
        self.gmf_user_embedding  = nn.Embedding(num_users, embedding_dim)
        self.gmf_movie_embedding = nn.Embedding(num_movies, embedding_dim)

        # --- Branche MLP ---
        self.mlp_user_embedding  = nn.Embedding(num_users, embedding_dim)
        self.mlp_movie_embedding = nn.Embedding(num_movies, embedding_dim)

        mlp_input_dim = embedding_dim * 2
        layers = []
        for output_dim in mlp_layers:
            layers.append(nn.Linear(mlp_input_dim, output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            mlp_input_dim = output_dim
        self.mlp = nn.Sequential(*layers)

        # --- Couche de fusion + Sigmoid ---
        self.output_layer = nn.Sequential(
            nn.Linear(embedding_dim + mlp_layers[-1], 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for embedding in [self.gmf_user_embedding, self.gmf_movie_embedding,
                          self.mlp_user_embedding, self.mlp_movie_embedding]:
            nn.init.normal_(embedding.weight, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        nn.init.kaiming_uniform_(self.output_layer[0].weight)

    def forward(self, user_ids, movie_ids):
        # Branche GMF
        gmf_out = self.gmf_user_embedding(user_ids) * self.gmf_movie_embedding(movie_ids)

        # Branche MLP
        mlp_input = torch.cat([
            self.mlp_user_embedding(user_ids),
            self.mlp_movie_embedding(movie_ids)
        ], dim=-1)
        mlp_out = self.mlp(mlp_input)

        # Fusion + Sigmoid
        combined = torch.cat([gmf_out, mlp_out], dim=-1)
        return self.output_layer(combined).squeeze()