import torch
import torch.nn as nn
import torchvision.models as models

class VisualSpeechModel(nn.Module):
    def __init__(self, vocab_size):
        super(VisualSpeechModel, self).__init__()
        
        # 1. 3D-CNN Frontend
        # Input shape: (B, 1, T, 96, 96)
        self.frontend3d = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        
        # 2. 2D ResNet Feature Extractor
        # We use a standard ResNet18 but replace the first conv layer to accept 64 channels
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Remove the classification head (avgpool and fc)
        self.resnet2d = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            nn.AdaptiveAvgPool2d(1)
        )
        
        # 3. Sequence Modeling Backend (2-layer BiGRU)
        # ResNet18 outputs 512 channels
        self.backend_gru = nn.GRU(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # 4. Linear Output Layer
        # Output of BiGRU is hidden_size * 2 = 512
        self.fc = nn.Linear(512, vocab_size)
        
    def forward(self, x):
        # x shape: (B, C, T, H, W) e.g., (B, 1, T, 96, 96)
        B, C, T, H, W = x.size()
        
        # Pass through 3D CNN
        x = self.frontend3d(x)
        
        # Prepare for 2D ResNet
        # Current shape: (B, 64, T, H_out, W_out)
        _, C_out, _, H_out, W_out = x.size()
        
        # Transpose to (B, T, C_out, H_out, W_out) and then merge B and T
        x = x.transpose(1, 2).contiguous()
        x = x.view(-1, C_out, H_out, W_out) # Shape: (B*T, 64, H_out, W_out)
        
        # Pass through 2D ResNet
        x = self.resnet2d(x) # Shape: (B*T, 512, 1, 1)
        
        # Flatten spatial dimensions
        x = x.view(B, T, -1) # Shape: (B, T, 512)
        
        # Pass through Sequence Model
        x, _ = self.backend_gru(x) # Shape: (B, T, 512)
        
        # Project to vocabulary size
        logits = self.fc(x) # Shape: (B, T, vocab_size)
        
        return logits

if __name__ == "__main__":
    # Sample forward-pass unit test
    print("Testing VisualSpeechModel forward pass...")
    
    batch_size = 2
    channels = 1
    frames = 75
    height = 96
    width = 96
    vocab_size = 28  # e.g., 26 letters + space + blank token
    
    # Create dummy input tensor (B, C, T, H, W)
    dummy_input = torch.randn(batch_size, channels, frames, height, width)
    print(f"Input shape: {dummy_input.shape}")
    
    # Initialize model
    model = VisualSpeechModel(vocab_size=vocab_size)
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
        
    print(f"Output shape: {output.shape}")
    
    expected_shape = (batch_size, frames, vocab_size)
    if output.shape == expected_shape:
        print("SUCCESS: Output shape matches expected dimensions!")
    else:
        print(f"ERROR: Output shape is {output.shape}, expected {expected_shape}")
