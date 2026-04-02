import torch
import torch.nn as nn

class GeneVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims, dropout_rate=0.1):
        super(GeneVAE, self).__init__()
        encoder_layers = []
        last_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(last_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout_rate))
            last_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(last_dim, latent_dim)
        self.fc_logvar = nn.Linear(last_dim, latent_dim)
        decoder_layers = []
        last_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(last_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(dropout_rate))
            last_dim = h_dim
        decoder_layers.append(nn.Linear(last_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar, z

class DrugNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, dropout_rate=0.1):
        super(DrugNN, self).__init__()
        layers = []
        last_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            last_dim = h_dim
        
        layers.append(nn.Linear(last_dim, output_dim))
        layers.append(nn.ReLU())
        self.drug_fc = nn.Sequential(*layers)

    def forward(self, drug_input):
        return self.drug_fc(drug_input)

class GeneDrugFusion(nn.Module):
    def __init__(self, gene_dim, drug_dim, output_dim, hidden_dims, dropout_rate=0.1):
        super(GeneDrugFusion, self).__init__()
        layers = []
        last_dim = gene_dim + drug_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            last_dim = h_dim
            
        layers.append(nn.Linear(last_dim, output_dim))
        self.fusion_layer = nn.Sequential(*layers)

    def forward(self, gene_latent, drug_feat):
        fusion_input = torch.cat([gene_latent, drug_feat], dim=1)
        return self.fusion_layer(fusion_input)

class WAVE(nn.Module):
    def __init__(self, config):
        super(WAVE, self).__init__()
        
        self.dr = getattr(config, 'dropout_rate', 0.1)
        self.gene_vae = GeneVAE(
            input_dim=config.input_dim,
            latent_dim=config.latent_dim,
            hidden_dims=config.vae_hidden_dims,
            dropout_rate=self.dr
        )
        self.drug_feature = DrugNN(
            input_dim=config.drug_input_dim,
            output_dim=config.drug_output_dim,
            hidden_dims=config.drug_hidden_dims,
            dropout_rate=self.dr
        )
        
        self.fusion = GeneDrugFusion(
            gene_dim=config.latent_dim, 
            drug_dim=config.drug_output_dim, 
            hidden_dims=config.fusion_hidden_dims, 
            output_dim=config.output_dim,
            dropout_rate=self.dr
        )

    def forward(self, gene_expr, drug_fp):
        reconstructed_gene, mu, logvar, z = self.gene_vae(gene_expr)
        drug_embedding = self.drug_feature(drug_fp)
        if self.training:
            fusion_input = z
        else:
            fusion_input = mu
        delta_expr = self.fusion(fusion_input, drug_embedding)
        final_gene_expr = reconstructed_gene + delta_expr
        return final_gene_expr, mu, logvar, delta_expr