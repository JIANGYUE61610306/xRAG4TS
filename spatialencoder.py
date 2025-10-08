import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from util import *
import time
import copy
import torch.optim as optim 


class TSMAE(nn.Module):
    def __init__(self, patch_size, in_channel, embed_dim, num_heads, mlp_ratio, dropout, num_token, mask_ratio, encoder_depth, decoder_depth,
                 mode="pre-train", distiler=False):
        super().__init__()
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_token = num_token
        self.mask_ratio = mask_ratio
        self.encoder_depth = encoder_depth
        self.mode = mode
        self.mlp_ratio = mlp_ratio
        self.distiler = distiler
        self.selected_feature = 0
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.patch_embedding = PatchEmbedding(patch_size, in_channel, embed_dim, norm_layer=None)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout=dropout)
        self.mask = MaskGenerator(num_token, mask_ratio)
        self.encoder = nn.Linear(embed_dim, embed_dim)
        self.enc_2_dec_emb = nn.Linear(embed_dim, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.decoder = TransformerLayers(embed_dim, decoder_depth, mlp_ratio, num_heads, dropout)
        self.output_layer = nn.Linear(embed_dim, patch_size)
        self.initialize_weights()
        if self.distiler:
            self.distiler_unit = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, self.embed_dim)
            )
        self.spatialcnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1)
        )

    def initialize_weights(self):
        nn.init.uniform_(self.positional_encoding.position_embedding, -.02, .02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def encoding(self, long_term_history, mask=True):
        t0 = time.time()
        batch_size, num_nodes, _, _ = long_term_history.shape
        patches = self.patch_embedding(long_term_history)
        patches = patches.transpose(-1, -2)
        patches = self.positional_encoding(patches)
        actual_num_token = patches.shape[2]
        if mask:
            unmasked_token_index, masked_token_index = self.mask(actual_num_token)
            encoder_input = patches[:, :, unmasked_token_index, :]
        else:
            unmasked_token_index, masked_token_index = None, None
            encoder_input = patches
        hidden_states_unmasked = self.encoder(encoder_input)
        hidden_states_unmasked = F.relu(hidden_states_unmasked)
        encoder_input = encoder_input.view(batch_size * num_nodes, 1, -1, self.embed_dim)
        hidden_states_unmasked = self.spatialcnn(encoder_input)
        hidden_states_unmasked = hidden_states_unmasked.view(batch_size, num_nodes, -1, self.embed_dim)
        hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(batch_size, num_nodes, -1, self.embed_dim)
        return hidden_states_unmasked, unmasked_token_index, masked_token_index

    def decoding(self, hidden_states_unmasked, masked_token_index):
        batch_size, num_nodes, _, _ = hidden_states_unmasked.shape
        hidden_states_unmasked = self.enc_2_dec_emb(hidden_states_unmasked)
        hidden_states_masked = self.positional_encoding(
            self.mask_token.expand(batch_size, num_nodes, len(masked_token_index), hidden_states_unmasked.shape[-1]),
            index=masked_token_index
        )
        hidden_states_full = torch.cat([hidden_states_unmasked, hidden_states_masked], dim=-2)
        hidden_states_full = self.decoder(hidden_states_full)
        hidden_states_full = self.decoder_norm(hidden_states_full)
        reconstruction_full = self.output_layer(hidden_states_full.view(batch_size, num_nodes, -1, self.embed_dim))
        return reconstruction_full

    def get_reconstructed_masked_tokens(self, reconstruction_full, real_value_full, unmasked_token_index, masked_token_index):
        batch_size, num_nodes, _, _ = reconstruction_full.shape
        reconstruction_masked_tokens = reconstruction_full[:, :, len(unmasked_token_index):, :]
        reconstruction_masked_tokens = reconstruction_masked_tokens.view(batch_size, num_nodes, -1).transpose(1, 2)
        label_full = real_value_full.permute(0, 3, 1, 2).unfold(1, self.patch_size, self.patch_size)
        label_full = label_full[:, :, :, self.selected_feature, :].transpose(1, 2)
        label_masked_tokens = label_full[:, :, masked_token_index, :].contiguous()
        label_masked_tokens = label_masked_tokens.view(batch_size, num_nodes, -1).transpose(1, 2)
        return reconstruction_masked_tokens, label_masked_tokens

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, batch_seen: int = None, epoch: int = None, **kwargs) -> torch.Tensor:
        history_data = history_data.permute(0, 2, 3, 1)
        if self.mode == "pre-train":
            t1 = time.time()
            hidden_states_unmasked, unmasked_token_index, masked_token_index = self.encoding(history_data)
            reconstruction_full = self.decoding(hidden_states_unmasked, masked_token_index)
            reconstruction_masked_tokens, label_masked_tokens = self.get_reconstructed_masked_tokens(
                reconstruction_full, history_data, unmasked_token_index, masked_token_index
            )
            return reconstruction_masked_tokens, label_masked_tokens
        else:
            hidden_states_full, _, _ = self.encoding(history_data, mask=False)
            return hidden_states_full


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channel, embed_dim, norm_layer):
        super().__init__()
        self.output_channel = embed_dim
        self.len_patch = patch_size
        self.input_channel = in_channel
        self.output_channel = embed_dim
        self.input_embedding = nn.Conv2d(
            in_channel,
            embed_dim,
            kernel_size=(self.len_patch, 1),
            stride=(self.len_patch, 1)
        )
        self.norm_layer = norm_layer if norm_layer is not None else nn.Identity()

    def forward(self, long_term_history):
        batch_size, num_nodes, num_feat, len_time_series = long_term_history.shape
        long_term_history = long_term_history.unsqueeze(-1)
        long_term_history = long_term_history.reshape(batch_size * num_nodes, num_feat, len_time_series, 1)
        output = self.input_embedding(long_term_history)
        output = self.norm_layer(output)
        output = output.squeeze(-1).view(batch_size, num_nodes, self.output_channel, -1)
        assert output.shape[-1] == len_time_series / self.len_patch
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.position_embedding = nn.Parameter(torch.empty(max_len, hidden_dim), requires_grad=True)

    def forward(self, input_data, index=None, abs_idx=None):
        batch_size, num_nodes, num_patches, num_feat = input_data.shape
        input_data = input_data.view(batch_size * num_nodes, num_patches, num_feat)
        if index is None:
            pe = self.position_embedding[:input_data.size(1), :].unsqueeze(0)
        else:
            pe = self.position_embedding[index].unsqueeze(0)
        input_data = input_data + pe
        input_data = self.dropout(input_data)
        input_data = input_data.view(batch_size, num_nodes, num_patches, num_feat)
        return input_data


class MaskGenerator(nn.Module):
    def __init__(self, num_tokens, mask_ratio):
        super().__init__()
        self.num_tokens = num_tokens
        self.mask_ratio = mask_ratio
        self.sort = True

    def uniform_rand(self, actual_num_token=0):
        if actual_num_token == 0:
            actual_num_token = self.num_tokens
        mask = list(range(int(actual_num_token)))
        random.shuffle(mask)
        mask_len = int(actual_num_token * self.mask_ratio)
        self.masked_tokens = mask[:mask_len]
        self.unmasked_tokens = mask[mask_len:]
        if self.sort:
            self.masked_tokens = sorted(self.masked_tokens)
            self.unmasked_tokens = sorted(self.unmasked_tokens)
        return self.unmasked_tokens, self.masked_tokens

    def forward(self, actual_num_token=0):
        self.unmasked_tokens, self.masked_tokens = self.uniform_rand(actual_num_token)
        return self.unmasked_tokens, self.masked_tokens


class TransformerLayers(nn.Module):
    def __init__(self, hidden_dim, nlayers, mlp_ratio, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = hidden_dim
        encoder_layers = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * mlp_ratio, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    def forward(self, src):
        B, N, L, D = src.shape
        src = src * math.sqrt(self.d_model)
        src = src.view(B * N, L, D)
        src = src.transpose(0, 1)
        output = self.transformer_encoder(src, mask=None)
        output = output.transpose(0, 1).view(B, N, L, D)
        return output


class tsMAE_trainer:
    def __init__(self, args, scaler):
        self.args = args
        if args.model == 'TSMAE':
            self.tsmae_model = TSMAE(args.patch_size, args.in_channel, args.embed_dim, args.num_heads, args.mlp_ratio, args.dropout, 
                args.num_token, args.mask_ratio, args.encoder_depth, args.decoder_depth, mode='pre-train')
        self.device = torch.device(args.device)
        self.tsmae_model = self.tsmae_model.to(self.device)
        self.tsmae_opt = optim.Adam(self.tsmae_model.parameters(), betas=(0.9, 0.95), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.tsformer_opt, milestones=[50], gamma = 0.5)
        self.scaler = scaler 
        self.clip = 5
    
    def pre_train(self, history_seq):
        self.tsmae_model.train()
        self.tsmae_opt.zero_grad()
        history_seq = history_seq.to(self.device)
        if self.args.model == 'TSMAE':
            reconstructed, label = self.tsmae_model(history_seq[:,:,:,:self.args.in_channel])
            # print('reconstructed', reconstructed.shape)
            # print('label', label.shape)
        reconstructed = self.scaler.inverse_transform(reconstructed)
        label = self.scaler.inverse_transform(label)
        mae_loss = masked_mae(reconstructed, label, 0.0)
        mae_loss.backward()
        if self.clip is not None:
            nn.utils.clip_grad_norm_(self.tsmae_model.parameters(), self.clip)
        self.tsformer_opt.step()
        rmse_loss = masked_rmse(reconstructed, label, 0.0)
        mape_loss = masked_mape(reconstructed, label, 0.0)
        return mae_loss.item(), rmse_loss.item(), mape_loss.item()
        
    
    def eval_pretrain(self, history_seq):
        self.tsmae_model.eval()
        with torch.no_grad():
            history_seq = history_seq.to(self.device)
            if self.args.model == 'TSMAE':
                reconstructed, label = self.tsmae_model(history_seq[:,:,:,:self.args.in_channel])
            reconstructed = self.scaler.inverse_transform(reconstructed)
            label = self.scaler.inverse_transform(label)
            mae_loss = masked_mae(reconstructed, label, 0.0)
            rmse_loss = masked_rmse(reconstructed, label, 0.0)
            mape_loss = masked_mape(reconstructed, label, 0.0)
        return mae_loss.item(), rmse_loss.item(), mape_loss.item()
    
    def save_doc(self, history_seq):
        self.tsmae_model.mode = 'inference'
        self.tsmae_model.eval()
        with torch.no_grad():
            history_seq = history_seq.to(self.device)
            if self.args.model == 'TSMAE':
                hidden_states_full = self.tsmae_model(history_seq[:,:,:,:self.args.in_channel])
        
        return hidden_states_full