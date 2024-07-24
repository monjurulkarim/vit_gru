import torch.nn as nn
import torch, math

import torch.nn.functional as F
import torch.optim as optim



# class AttentionLayer(nn.Module):
#     def __init__(self, hidden_size):
#         super(AttentionLayer, self).__init__()
#         self.attn = nn.Linear(hidden_size * 2, hidden_size)
#         self.v = nn.Linear(hidden_size, 1, bias=False)

#     def forward(self, hidden, encoder_outputs):
#         seq_len = encoder_outputs.size(1)
#         hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
#         energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
#         attention = self.v(energy).squeeze(2)
#         return F.softmax(attention, dim=1)

# class AdvancedGRUModel(nn.Module):
#     def __init__(self, bb_input_size=4, cnn_feature_size=768, bb_embedding_size=32, cnn_embedding_size=128, hidden_size=256, output_size=4, num_layers=2, num_timesteps=12):
#         super(AdvancedGRUModel, self).__init__()
        
#         self.bb_input_size = bb_input_size
#         self.cnn_feature_size = cnn_feature_size
#         self.bb_embedding_size = bb_embedding_size
#         self.cnn_embedding_size = cnn_embedding_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size  # Added this line
#         self.num_layers = num_layers
#         self.num_timesteps = num_timesteps

#         # Input embedding layers
#         self.bb_embedding = nn.Linear(bb_input_size, bb_embedding_size)
#         self.cnn_embedding = nn.Linear(cnn_feature_size, cnn_embedding_size)
        
#         # GRU for bounding box processing
#         self.bb_gru = nn.GRU(bb_embedding_size, hidden_size, num_layers, batch_first=True)
        
#         # GRU for combined features
#         self.combined_gru = nn.GRU(hidden_size + cnn_embedding_size, hidden_size, num_layers, batch_first=True)
        
#         # Attention layer
#         self.attention = AttentionLayer(hidden_size)
        
#         # Prediction network
#         self.predictor = nn.Sequential(
#             nn.Linear(hidden_size * 2, hidden_size),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(hidden_size, output_size)
#         )
        
#     def forward(self, bb_input, cnn_features):
#         batch_size = bb_input.size(0)

#         # Ensure bb_input is 3D: [batch_size, seq_len, feature_size]
#         if bb_input.dim() == 2:
#             bb_input = bb_input.unsqueeze(1)
        
#         # Ensure cnn_features is 2D: [batch_size, feature_size]
#         if cnn_features.dim() == 3:
#             cnn_features = cnn_features.squeeze(1)
#         elif cnn_features.dim() == 4:
#             cnn_features = cnn_features.squeeze(1).squeeze(1)

#         # Embed inputs
#         bb_embedded = self.bb_embedding(bb_input)
#         cnn_embedded = self.cnn_embedding(cnn_features).unsqueeze(1)
        
#         # Process bounding box with BB GRU
#         bb_output, bb_hidden = self.bb_gru(bb_embedded)
        
#         # Combine BB GRU output with CNN features
#         combined_input = torch.cat((bb_output, cnn_embedded.expand(-1, bb_output.size(1), -1)), dim=-1)
        
#         # Process combined features with Combined GRU
#         combined_output, combined_hidden = self.combined_gru(combined_input)
        
#         # Initialize output tensor
#         outputs = torch.zeros(batch_size, self.num_timesteps, self.output_size).to(bb_input.device)
        
#         # Generate sequence
#         for t in range(self.num_timesteps):
#             # Calculate attention weights
#             attn_weights = self.attention(combined_hidden[-1], combined_output)
#             context = torch.bmm(attn_weights.unsqueeze(1), combined_output).squeeze(1)
            
#             # Predict next bounding box
#             prediction = self.predictor(torch.cat((combined_hidden[-1], context), dim=1))
#             outputs[:, t, :] = prediction
        
#         return outputs

#12-56
class AdvancedGRUModel(nn.Module):
    def __init__(self, bb_input_size=4, cnn_feature_size=768,bb_embedding_size = 32,cnn_embedding_size=128, hidden_size=256, output_size=4, num_layers=2, num_timesteps=20):
        super(AdvancedGRUModel, self).__init__()
        
        self.bb_input_size = bb_input_size
        self.cnn_feature_size = cnn_feature_size
        self.bb_embedding_size = bb_embedding_size
        self.cnn_embedding_size = cnn_embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps

        # Input embedding layers
        self.bb_embedding = nn.Linear(bb_input_size, bb_embedding_size)
        self.cnn_embedding = nn.Linear(cnn_feature_size, cnn_embedding_size)
        
        # GRU for bounding box processing
        self.bb_gru = nn.GRU(bb_embedding_size, hidden_size, num_layers, batch_first=True)
        
        # GRU for combined features
        self.combined_gru = nn.GRU(hidden_size + cnn_embedding_size, hidden_size, num_layers, batch_first=True)
        
        # Prediction network
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size * num_timesteps)
        )
        
    def forward(self, bb_input, cnn_features):
        batch_size = bb_input.size(0)

        # Embed inputs
        bb_embedded = self.bb_embedding(bb_input)
        cnn_embedded = self.cnn_embedding(cnn_features)
        
        # Process bounding box with BB GRU
        bb_output, bb_hidden = self.bb_gru(bb_embedded)
        
        # Combine BB GRU output with CNN features
        combined_input = torch.cat((bb_output, cnn_embedded.expand(-1, bb_output.size(1), -1)), dim=-1)
        
        # Process combined features with Combined GRU
        combined_output, combined_hidden = self.combined_gru(combined_input)
        
        # Use the last hidden state for prediction
        last_hidden = combined_hidden[-1]
        
        # Predict bounding boxes for all timesteps
        output = self.predictor(last_hidden)
        
        # Reshape output to [batch_size, num_timesteps, 4]
        output = output.view(batch_size, self.num_timesteps, -1)
        
        return output


# Large model
# class GRUModel(nn.Module):
#     def __init__(self, input_size=772, hidden_size=512, output_size=4, num_layers=2, num_timesteps=12):
#         super(GRUModel, self).__init__()
        
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.num_timesteps = num_timesteps
        
#         # GRU layer
#         self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
#         # Attention mechanism
#         self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=8)
        
#         # Adjusted predictor network
#         self.predictor = nn.Sequential(
#             nn.Linear(hidden_size * 4, 512),  # *4 because we're concatenating bidirectional output and hidden state
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(256, output_size)
#         )
        
#         # Feedback network
#         self.feedback = nn.Sequential(
#             nn.Linear(output_size, 256),
#             nn.ReLU(),
#             nn.Linear(256, 512),
#             nn.ReLU(),
#             nn.Linear(512, input_size)
#         )
        
#     def forward(self, x):
#         batch_size = x.size(0)
        
#         # Initialize hidden state
#         h = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        
#         outputs = []
#         for _ in range(self.num_timesteps):
#             # GRU pass
#             gru_out, h = self.gru(x, h)
            
#             # Apply attention
#             attn_out, _ = self.attention(gru_out, gru_out, gru_out)
            
#             # Concatenate attention output with last hidden state
#             combined = torch.cat((attn_out[:, -1, :], h[-2:].transpose(0,1).contiguous().view(batch_size, -1)), dim=1)
            
#             # Predict output
#             output = self.predictor(combined)
#             outputs.append(output)
            
#             # Prepare feedback
#             feedback = self.feedback(output)
#             x = feedback.unsqueeze(1)  # Update x for the next iteration
        
#         # Stack the outputs
#         outputs = torch.stack(outputs, dim=1)
        
#         return outputs


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






