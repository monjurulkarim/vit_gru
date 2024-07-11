import torch.nn as nn
import torch, math

import torch.nn.functional as F
import torch.optim as optim


class AdvancedGRUModel(nn.Module):
    def __init__(self, bb_input_size=4, cnn_feature_size=768, hidden_size=256, output_size=4, num_layers=1, num_timesteps=12):
        super(AdvancedGRUModel, self).__init__()
        
        self.bb_input_size = bb_input_size
        self.cnn_feature_size = cnn_feature_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        
        # GRU for bounding box processing
        self.bb_gru = nn.GRU(bb_input_size, hidden_size, num_layers, batch_first=True)
        
        # GRU for combined features
        self.combined_gru = nn.GRU(hidden_size + cnn_feature_size, hidden_size, num_layers, batch_first=True)
        
        # Prediction network
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_size * num_timesteps)
        )
        
    def forward(self, bb_input, cnn_features):
        batch_size = bb_input.size(0)
        
        # Process bounding box with BB GRU
        _, bb_hidden = self.bb_gru(bb_input)
        
        # Combine BB GRU output with CNN features
        bb_hidden_squeezed = bb_hidden[-1].unsqueeze(0)  # Take the last layer's hidden state

        combined_input = torch.cat((bb_hidden_squeezed, cnn_features), dim=-1)    
        # Process combined features with Combined GRU
        _, combined_hidden = self.combined_gru(combined_input)
        
        # Use the last hidden state for prediction
        last_hidden = combined_hidden[-1]
        
        # Predict bounding boxes for all timesteps
        output = self.predictor(last_hidden)
        
        # Reshape output to [batch_size, num_timesteps, 4]
        output = output.view(batch_size, self.num_timesteps, -1)
        
        return output


## Large model
# class GRUModel(nn.Module):
    # def __init__(self, input_size=772, hidden_size=512, output_size=4, num_layers=2, num_timesteps=12):
    #     super(GRUModel, self).__init__()
        
    #     self.input_size = input_size
    #     self.hidden_size = hidden_size
    #     self.num_layers = num_layers
    #     self.num_timesteps = num_timesteps
        
    #     # GRU layer
    #     self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
    #     # Attention mechanism
    #     self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=8)
        
    #     # Adjusted predictor network
    #     self.predictor = nn.Sequential(
    #         nn.Linear(hidden_size * 4, 512),  # *4 because we're concatenating bidirectional output and hidden state
    #         nn.ReLU(),
    #         nn.Dropout(0.2),
    #         nn.Linear(512, 256),
    #         nn.ReLU(),
    #         nn.Dropout(0.2),
    #         nn.Linear(256, output_size)
    #     )
        
    #     # Feedback network
    #     self.feedback = nn.Sequential(
    #         nn.Linear(output_size, 256),
    #         nn.ReLU(),
    #         nn.Linear(256, 512),
    #         nn.ReLU(),
    #         nn.Linear(512, input_size)
    #     )
        
    # def forward(self, x):
    #     batch_size = x.size(0)
        
    #     # Initialize hidden state
    #     h = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        
    #     outputs = []
    #     for _ in range(self.num_timesteps):
    #         # GRU pass
    #         gru_out, h = self.gru(x, h)
            
    #         # Apply attention
    #         attn_out, _ = self.attention(gru_out, gru_out, gru_out)
            
    #         # Concatenate attention output with last hidden state
    #         combined = torch.cat((attn_out[:, -1, :], h[-2:].transpose(0,1).contiguous().view(batch_size, -1)), dim=1)
            
    #         # Predict output
    #         output = self.predictor(combined)
    #         outputs.append(output)
            
    #         # Prepare feedback
    #         feedback = self.feedback(output)
    #         x = feedback.unsqueeze(1)  # Update x for the next iteration
        
    #     # Stack the outputs
    #     outputs = torch.stack(outputs, dim=1)
        
    #     return outputs


# class GRUModel(nn.Module):
#     def __init__(self, input_size=772, hidden_size=256, output_size=4, num_layers=2, num_timesteps=12):
#         super(GRUModel, self).__init__()
        
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.num_timesteps = num_timesteps
        
#         # GRU layer
#         self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        
#         # Two-layer predictor
#         self.predictor = nn.Sequential(
#             nn.Linear(hidden_size, 128),
#             nn.ReLU(),
#             nn.Linear(128, output_size)
#         )
        
#         # Layer to map output back to input size for feedback
#         self.feedback = nn.Linear(output_size, input_size)
        
#     def forward(self, x):
#         # Initialize hidden state with zeros
#         h = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
#         # Initial GRU pass
#         _, h = self.gru(x, h)
        
#         outputs = []
#         for _ in range(self.num_timesteps):
#             # Predict output using current hidden state
#             output = self.predictor(h[-1])
#             outputs.append(output)
            
#             # Prepare feedback: map output to input size
#             feedback = self.feedback(output)
#             feedback = feedback.unsqueeze(1)  # Add time dimension
            
#             # Update hidden state using feedback
#             _, h = self.gru(feedback, h)
        
#         # Stack the outputs
#         outputs = torch.stack(outputs, dim=1)
        
#         return outputs






