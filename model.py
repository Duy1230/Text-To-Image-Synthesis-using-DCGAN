import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Generator(nn.Module):

    def __init__(self, noise_size, feature_size, num_channels, embedding_size, reduced_dim_size):
        super(Generator, self).__init__()
        ### Your Code Here
        # Reduce dim size for embedding vectors of captiopn
        self.reduced_dim_size = reduced_dim_size
        self.text_encoder = nn.Sequential(
            nn.Linear(in_features=embedding_size, out_features=self.reduced_dim_size),
            nn.BatchNorm1d(num_features=self.reduced_dim_size),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.upSampler = nn.Sequential(
            nn.ConvTranspose2d(in_channels=reduced_dim_size + noise_size, out_channels=feature_size * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=feature_size * 4),
            nn.GELU(),

            nn.ConvTranspose2d(in_channels=feature_size * 4, out_channels=feature_size * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=feature_size * 2),
            nn.GELU(),

            nn.ConvTranspose2d(in_channels=feature_size * 2, out_channels=feature_size, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=feature_size),
            nn.GELU(),

            nn.ConvTranspose2d(in_channels=feature_size, out_channels= int(feature_size/2), kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=int(feature_size/2)),
            nn.GELU(),

            nn.ConvTranspose2d(in_channels=int(feature_size/2), out_channels= int(feature_size/4), kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=int(feature_size/4)),
            nn.GELU(),

            nn.ConvTranspose2d(in_channels=int(feature_size/4), out_channels= num_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()          
        )

    def forward(self, noise, text_embeddings):
        encoded_text = self.text_encoder(text_embeddings)
        input_vector = torch.cat([noise, encoded_text], dim=1).unsqueeze(2).unsqueeze(2)
        output = self.upSampler(input_vector)
        return output
    

generator = Generator(100, 128, 3, 768, 256).to(device)
generator.load_state_dict(torch.load("checkpoints/generator.pth"))
generator.eval()