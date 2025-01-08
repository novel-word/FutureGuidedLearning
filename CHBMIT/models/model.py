import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_LSTM_Model(nn.Module):
    def __init__(self, X_train_shape):
        super(CNN_LSTM_Model, self).__init__()
        
        # CNN module
        self.bn1 = nn.BatchNorm3d(num_features=X_train_shape[1])
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, 
                               kernel_size=(int(X_train_shape[2]/2), 5, 5), stride=(1, 2, 2))
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.bn2 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.pool2 = nn.AvgPool3d(kernel_size=3, stride=2)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        # LSTM module
        self.lstm = nn.LSTM(input_size=32*60, hidden_size=512, num_layers=3, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 2)  # 2 output classes
        
    def forward(self, x, verbose=False):
        # CNN forward pass
        if verbose: print('input size', x.shape)
        x = self.bn1(x)
        x = F.relu(self.conv1(x))
        if verbose: print('shape after conv1:', x.shape)
        x = self.pool1(x)
        if verbose: print('shape after pool1:', x.shape)

        x = self.bn2(x)
        x = F.relu(self.conv2(x))
        if verbose: print('shape after conv2:', x.shape)
        x = self.pool2(x)
        if verbose: print('shape after pool2:', x.shape)

        # Reshape for LSTM
        x = x.squeeze(2)
        x = self.flatten(x)
        if verbose: print('shape after flatten:', x.shape)
        # LSTM forward pass

        x, _ = self.lstm(x)
        if verbose: print('shape after LSTM:', x.shape)
        
        # Fully connected layers
        x = self.fc1(x)  # Select the last time step's output
        if verbose: print('shape after fc1:', x.shape)
        x = self.dropout(x)
        x = self.fc2(x)
        if verbose: print('shape after fc2:', x.shape)
        return x
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (batch_size, 1, timesteps, frequency)
        x = self.proj(x)  # (batch_size, embed_dim, n_patches_h, n_patches_w)
        x = x.flatten(2)  # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, hidden_dim, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        return self.transformer_encoder(x)

class MViT(nn.Module):
    def __init__(self, X_shape, in_channels, num_classes, patch_size, 
                 embed_dim, num_heads, hidden_dim, num_layers, dropout):
        super(MViT, self).__init__()
        
        self.patch_embeds = nn.ModuleList([PatchEmbedding(patch_size, embed_dim) for _ in range(in_channels)])
        self.pos_embeds = nn.ParameterList([nn.Parameter(torch.zeros(1, (X_shape[3] // patch_size[0]) * (X_shape[4] // patch_size[1]) + 1, embed_dim)) for _ in range(in_channels)])
        self.cls_tokens = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, embed_dim)) for _ in range(in_channels)])
        self.encoders = nn.ModuleList([TransformerEncoder(embed_dim, num_heads, hidden_dim, num_layers, dropout) for _ in range(in_channels)])
        
        self.head = nn.Linear(embed_dim * in_channels, num_classes)

    def forward(self, x):
        # x: (batch_size, channels, timesteps, frequency)
        x = torch.squeeze(x)
        outputs = []
        
        for i in range(x.size(1)):  # Iterate over channels
            channel_input = x[:, i:i+1, :, :]  # (batch_size, 1, timesteps, frequency)
            
            # Patch embedding
            embedded_patches = self.patch_embeds[i](channel_input)
            
            # Add classification token
            batch_size, n_patches, _ = embedded_patches.size()
            cls_tokens = self.cls_tokens[i].expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)
            x_i = torch.cat((cls_tokens, embedded_patches), dim=1)  # (batch_size, n_patches+1, embed_dim)
            
            # Add positional encoding
            x_i += self.pos_embeds[i][:, :n_patches+1, :]
            
            # Pass through Transformer encoder
            encoded_output = self.encoders[i](x_i)
            
            # Take the cls token output
            outputs.append(encoded_output[:, 0])  # (batch_size, embed_dim)
        
        # Concatenate outputs from all channels
        x = torch.cat(outputs, dim=1)  # (batch_size, embed_dim * in_channels)
        
        # Classification head
        x = self.head(x)  # (batch_size, num_classes)
        
        return x