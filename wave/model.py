import torch
import torch.nn as nn


class GeneVAE(nn.Module):
    def __init__(self, input_dim=978, latent_dim=128, hidden_dim1=512, hidden_dim2=256):
        super(GeneVAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU()
        )
        # VAE latent
        self.fc_mu = nn.Linear(hidden_dim2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim2, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, input_dim)
        )

    def reparameterize(self, mu, logvar):
        """ Reparameterization technique """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """ VAE forward """
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar, encoded  # Return encoded for subsequent concatenation.



class DrugNN(nn.Module):
    def __init__(self, input_dim=2048, output_dim=128):
        super(DrugNN, self).__init__()
        self.drug_fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.ReLU()
        )

    def forward(self, drug_input):
        return self.drug_fc(drug_input)



class GeneDrugFusion(nn.Module):
    def __init__(self, gene_dim=256, drug_dim=128, hidden_dim=512, output_dim=978):
        super(GeneDrugFusion, self).__init__()

        self.fusion_layer = nn.Sequential(
            nn.Linear(gene_dim + drug_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, gene_latent, drug_feat):
        fusion_input = torch.cat([gene_latent, drug_feat], dim=1)
        return self.fusion_layer(fusion_input)



class WAVE(nn.Module):
    def __init__(self):
        super(WAVE, self).__init__()

        self.gene_vae = GeneVAE()
        self.drug_feature = DrugNN()
        self.fusion = GeneDrugFusion()

    def forward(self, gene_expr, drug_fp):
        # VAE
        reconstructed_gene, mu, logvar, gene_latent = self.gene_vae(gene_expr)

        # Drug feature extraction
        drug_embedding = self.drug_feature(drug_fp)

        # Gene expression change
        delta_expr = self.fusion(gene_latent, drug_embedding)

        # the final gene expression = VAE output + delta_expr
        final_gene_expr = reconstructed_gene + delta_expr

        return final_gene_expr, mu, logvar, delta_expr