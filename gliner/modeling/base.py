from typing import Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
import warnings

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from transformers.utils import ModelOutput

from .encoder import Encoder, BiEncoder
from .layers import LstmSeq2SeqEncoder, CrossFuser, create_projection_layer
from .scorers import Scorer
from .loss_functions import focal_loss_with_logits
from .span_rep import SpanRepLayer

class InterHeadText(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Use a single projection for both inputs
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, text1, text2):
        # Project both text inputs using the same layer
        proj1 = self.proj(text1)
        proj2 = self.proj(text2)

        # Compute four interaction scores:
        a_11 = (text1 * proj1).sum(-1)  # self-interaction of text1
        a_12 = (text1 * proj2).sum(-1)  # cross-interaction: text1 with text2's projection
        a_21 = (text2 * proj1).sum(-1)  # cross-interaction: text2 with text1's projection
        a_22 = (text2 * proj2).sum(-1)  # self-interaction of text2

        # Stack and apply softmax to obtain weights
        weights1 = self.softmax(torch.stack([a_11, a_12], dim=0))
        weights2 = self.softmax(torch.stack([a_21, a_22], dim=0))
        w11, w12 = weights1.split([1, 1], dim=0)
        w21, w22 = weights2.split([1, 1], dim=0)

        # Reshape to allow broadcasting
        w11 = w11.squeeze(0).unsqueeze(-1)
        w12 = w12.squeeze(0).unsqueeze(-1)
        w21 = w21.squeeze(0).unsqueeze(-1)
        w22 = w22.squeeze(0).unsqueeze(-1)

        # Create new representations as weighted combinations:
        new_text1 = w11 * text1 + w12 * text2
        new_text2 = w21 * text1 + w22 * text2

        return new_text1, new_text2


class CrossAttentionHead(nn.Module):
    """
    A cross-attention layer that lets span representations (queries)
    attend to prompt embeddings (keys/values). This supports different
    sequence lengths between the two inputs.
    """
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, span_rep, prompt_emb):
        # span_rep: (B, N, hidden_dim) where N is total number of spans (flattened)
        # prompt_emb: (B, C, hidden_dim) where C is number of prompt tokens
        attn_output, _ = self.attn(query=span_rep, key=prompt_emb, value=prompt_emb)
        # Apply residual connection and normalization
        span_rep = self.norm(span_rep + self.dropout(attn_output))
        return span_rep
        
@dataclass
class GLiNERModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    prompts_embedding: Optional[torch.FloatTensor] = None
    prompts_embedding_mask: Optional[torch.LongTensor] = None
    words_embedding: Optional[torch.FloatTensor] = None
    mask: Optional[torch.LongTensor] = None

def extract_word_embeddings(token_embeds, words_mask, attention_mask, 
                                    batch_size, max_text_length, embed_dim, text_lengths):
    words_embedding = torch.zeros(
        batch_size, max_text_length, embed_dim, dtype=token_embeds.dtype, device=token_embeds.device
    )

    batch_indices, word_idx = torch.where(words_mask>0)
    
    target_word_idx = words_mask[batch_indices, word_idx]-1

    words_embedding[batch_indices, target_word_idx] = token_embeds[batch_indices, word_idx]
    
    aranged_word_idx = torch.arange(max_text_length, 
                                        dtype=attention_mask.dtype, 
                                        device=token_embeds.device).expand(batch_size, -1)
    mask = aranged_word_idx<text_lengths
    return words_embedding, mask


def extract_prompt_features_and_word_embeddings(config, token_embeds, input_ids, attention_mask, 
                                                                    text_lengths, words_mask, embed_ent_token = True, **kwargs):
    # getting prompt embeddings
    batch_size, sequence_length, embed_dim = token_embeds.shape

    class_token_mask = input_ids == config.class_token_index
    num_class_tokens = torch.sum(class_token_mask, dim=-1, keepdim=True)

    max_embed_dim = num_class_tokens.max()
    max_text_length = text_lengths.max()
    aranged_class_idx = torch.arange(max_embed_dim, 
                                        dtype=attention_mask.dtype, 
                                        device=token_embeds.device).expand(batch_size, -1)
    
    batch_indices, target_class_idx = torch.where(aranged_class_idx<num_class_tokens)
    _, class_indices = torch.where(class_token_mask)
    if not embed_ent_token:
        class_indices+=1

    prompts_embedding = torch.zeros(
        batch_size, max_embed_dim, embed_dim, dtype=token_embeds.dtype, device=token_embeds.device
    )

    prompts_embedding_mask = (aranged_class_idx < num_class_tokens).to(attention_mask.dtype)

    prompts_embedding[batch_indices, target_class_idx] = token_embeds[batch_indices, class_indices]
    
    #getting words embedding
    words_embedding, mask = extract_word_embeddings(token_embeds, words_mask, attention_mask, 
                                    batch_size, max_text_length, embed_dim, text_lengths)

    return prompts_embedding, prompts_embedding_mask, words_embedding, mask

class BaseModel(ABC, nn.Module):
    def __init__(self, config, from_pretrained = False):
        super(BaseModel, self).__init__()
        self.config = config
        
        if not config.labels_encoder:
            self.token_rep_layer = Encoder(config, from_pretrained)
        else:
            self.token_rep_layer = BiEncoder(config, from_pretrained)
        if self.config.has_rnn:
            self.rnn = LstmSeq2SeqEncoder(config)

        if config.post_fusion_schema:
            self.config.num_post_fusion_layers = 3
            print('Initializing cross fuser...')
            print('Post fusion layer:', config.post_fusion_schema)
            print('Number of post fusion layers:', config.num_post_fusion_layers)

            self.cross_fuser = CrossFuser(self.config.hidden_size,
                                            self.config.hidden_size,
                                            num_heads=self.token_rep_layer.bert_layer.model.config.num_attention_heads,
                                            num_layers=self.config.num_post_fusion_layers,
                                            dropout=config.dropout, 
                                            schema=config.post_fusion_schema)
            
    def features_enhancement(self, text_embeds, labels_embeds, text_mask=None, labels_mask=None):
        labels_embeds, text_embeds = self.cross_fuser(labels_embeds, text_embeds, labels_mask, text_mask)
        return text_embeds, labels_embeds
    
    def _extract_prompt_features_and_word_embeddings(self, token_embeds, input_ids, attention_mask, 
                                                                    text_lengths, words_mask):
        prompts_embedding, prompts_embedding_mask, words_embedding, mask = extract_prompt_features_and_word_embeddings(self.config, 
                                                                                                                       token_embeds, 
                                                                                                                       input_ids, 
                                                                                                                       attention_mask, 
                                                                                                                       text_lengths, 
                                                                                                                       words_mask,
                                                                                                                       self.config.embed_ent_token)
        return prompts_embedding, prompts_embedding_mask, words_embedding, mask

    def get_uni_representations(self, 
                input_ids: Optional[torch.FloatTensor] = None,
                attention_mask: Optional[torch.LongTensor] = None,
                text_lengths: Optional[torch.Tensor] = None,
                words_mask: Optional[torch.LongTensor] = None,
                **kwargs):
        
        token_embeds = self.token_rep_layer(input_ids, attention_mask, **kwargs)

        prompts_embedding, prompts_embedding_mask, words_embedding, mask = self._extract_prompt_features_and_word_embeddings(token_embeds, input_ids, attention_mask, 
                                                                                                        text_lengths, words_mask)
        
        if self.config.has_rnn:
            words_embedding = self.rnn(words_embedding, mask)

        return prompts_embedding, prompts_embedding_mask, words_embedding, mask
    
    def get_bi_representations(self, 
                input_ids: Optional[torch.FloatTensor] = None,
                attention_mask: Optional[torch.LongTensor] = None,
                labels_embeds: Optional[torch.FloatTensor] = None,
                labels_input_ids: Optional[torch.FloatTensor] = None,
                labels_attention_mask: Optional[torch.LongTensor] = None,
                text_lengths: Optional[torch.Tensor] = None,
                words_mask: Optional[torch.LongTensor] = None,
                **kwargs): 
        if labels_embeds is not None:
            token_embeds = self.token_rep_layer.encode_text(input_ids, attention_mask, **kwargs)
        else:
            token_embeds, labels_embeds = self.token_rep_layer(input_ids, attention_mask,
                                                           labels_input_ids, labels_attention_mask, 
                                                                                            **kwargs) 
        batch_size, sequence_length, embed_dim = token_embeds.shape
        max_text_length = text_lengths.max()
        words_embedding, mask = extract_word_embeddings(token_embeds, words_mask, attention_mask, 
                                    batch_size, max_text_length, embed_dim, text_lengths)
        
        labels_embeds = labels_embeds.unsqueeze(0)
        labels_embeds = labels_embeds.expand(batch_size, -1, -1)
        labels_mask = torch.ones(labels_embeds.shape[:-1], dtype=attention_mask.dtype,
                                                        device = attention_mask.device)

        labels_embeds = labels_embeds.to(words_embedding.dtype)
        if hasattr(self, "cross_fuser"):
            words_embedding, labels_embeds = self.features_enhancement(words_embedding, labels_embeds, text_mask=mask, labels_mask=labels_mask)
        
        return labels_embeds, labels_mask, words_embedding, mask

    def get_representations(self, 
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            labels_embeddings: Optional[torch.FloatTensor] = None,
            labels_input_ids: Optional[torch.FloatTensor] = None,
            labels_attention_mask: Optional[torch.LongTensor] = None,
            text_lengths: Optional[torch.Tensor] = None,
            words_mask: Optional[torch.LongTensor] = None,
            **kwargs):
        if self.config.labels_encoder:
            prompts_embedding, prompts_embedding_mask, words_embedding, mask = self.get_bi_representations(
                    input_ids, attention_mask, labels_embeddings, labels_input_ids, labels_attention_mask, 
                                                                        text_lengths, words_mask, **kwargs
            )
        else:
            prompts_embedding, prompts_embedding_mask, words_embedding, mask = self.get_uni_representations(
                                    input_ids, attention_mask, text_lengths, words_mask, **kwargs
            )
        return prompts_embedding, prompts_embedding_mask, words_embedding, mask 
    
    @abstractmethod
    def forward(self, x):
        pass

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor, 
                    alpha: float = -1., gamma: float = 0.0, label_smoothing: float = 0.0):
        
        all_losses = focal_loss_with_logits(logits, labels,
                                            alpha=alpha,
                                            gamma=gamma,
                                            label_smoothing=label_smoothing)
        return all_losses

    @abstractmethod
    def loss(self, x):
        pass
    
class SpanModel(BaseModel):
    def __init__(self, config, encoder_from_pretrained):
        super(SpanModel, self).__init__(config, encoder_from_pretrained)
        self.span_rep_layer = SpanRepLayer(
            span_mode=config.span_mode, 
            hidden_size=config.hidden_size, 
            max_width=config.max_width,
            dropout=config.dropout
        )
        self.prompt_rep_layer = create_projection_layer(config.hidden_size, config.dropout)
        
        # Create n layers of cross-attention interaction heads.
        self.num_interaction_layers = 5
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionHead(hidden_dim=config.hidden_size, num_heads=8, dropout=config.dropout)
            for _ in range(self.num_interaction_layers)
        ])

    def forward(self,        
                input_ids: Optional[torch.FloatTensor] = None,
                attention_mask: Optional[torch.LongTensor] = None,
                labels_embeddings: Optional[torch.FloatTensor] = None,
                labels_input_ids: Optional[torch.FloatTensor] = None,
                labels_attention_mask: Optional[torch.LongTensor] = None,
                words_embedding: Optional[torch.FloatTensor] = None,
                mask: Optional[torch.LongTensor] = None,
                prompts_embedding: Optional[torch.FloatTensor] = None,
                prompts_embedding_mask: Optional[torch.LongTensor] = None,
                words_mask: Optional[torch.LongTensor] = None,
                text_lengths: Optional[torch.Tensor] = None,
                span_idx: Optional[torch.LongTensor] = None,
                span_mask: Optional[torch.LongTensor] = None,
                labels: Optional[torch.FloatTensor] = None,
                **kwargs
                ):
        # Get initial representations (implementation provided elsewhere)
        prompts_embedding, prompts_embedding_mask, words_embedding, mask = self.get_representations(
            input_ids, attention_mask, 
            labels_embeddings, labels_input_ids, labels_attention_mask, 
            text_lengths, words_mask
        )
        span_idx = span_idx * span_mask.unsqueeze(-1)
        # Obtain span representations: shape (B, L, K, hidden_dim)
        span_rep = self.span_rep_layer(words_embedding, span_idx)
        # Get prompt embeddings: shape (B, C, hidden_dim)
        prompts_embedding = self.prompt_rep_layer(prompts_embedding)
        
        # The two inputs have different sequence lengths:
        #   span_rep: (B, L, K, hidden_dim) and prompts_embedding: (B, C, hidden_dim).
        # Flatten span_rep over the span dimensions (L and K) to get shape (B, N, hidden_dim), where N = L * K.
        B, L, K, D = span_rep.shape
        span_rep_flat = span_rep.view(B, L * K, D)
        
        # Apply each cross-attention layer to update span representations.
        for cross_attn in self.cross_attn_layers:
            span_rep_flat = cross_attn(span_rep_flat, prompts_embedding)
        
        # Reshape the updated span representations back to (B, L, K, hidden_dim)
        span_rep = span_rep_flat.view(B, L, K, D)
        
        # Compute matching scores via einsum:
        # span_rep: (B, L, K, hidden_dim)
        # prompts_embedding: (B, C, hidden_dim)
        # Resulting scores: (B, L, K, C)
        scores = torch.einsum("BLKD,BCD->BLKC", span_rep, prompts_embedding)

        loss = None
        if labels is not None:
            loss = self.loss(scores, labels, prompts_embedding_mask, span_mask, **kwargs)

        output = GLiNERModelOutput(
            logits=scores,
            loss=loss,
            prompts_embedding=prompts_embedding,
            prompts_embedding_mask=prompts_embedding_mask,
            words_embedding=words_embedding,
            mask=mask,
        )
        return output
    
    def loss(self, scores, labels, prompts_embedding_mask, mask_label,
                        alpha: float = -1., gamma: float = 0.0, label_smoothing: float = 0.0, 
                        reduction: str = 'sum', **kwargs):
        
        batch_size = scores.shape[0]
        num_classes = prompts_embedding_mask.shape[-1]

        scores = scores.view(-1, num_classes)
        labels = labels.view(-1, num_classes)
        
        all_losses = self._loss(scores, labels, alpha, gamma, label_smoothing)

        masked_loss = all_losses.view(batch_size, -1, num_classes) * prompts_embedding_mask.unsqueeze(1)
        all_losses = masked_loss.view(-1, num_classes)

        mask_label = mask_label.view(-1, 1)
        
        all_losses = all_losses * mask_label.float()

        if reduction == "mean":
            loss = all_losses.mean()
        elif reduction == 'sum':
            loss = all_losses.sum()
        else:
            warnings.warn(
                f"Invalid Value for config 'loss_reduction': '{reduction} \n Supported reduction modes:"
                f" 'none', 'mean', 'sum'. It will be used 'sum' instead.")
            loss = all_losses.sum()
        return loss
    
class TokenModel(BaseModel):
    def __init__(self, config, encoder_from_pretrained):
        super(TokenModel, self).__init__(config, encoder_from_pretrained)
        self.scorer = Scorer(config.hidden_size, config.dropout)

    def forward(self,        
                input_ids: Optional[torch.FloatTensor] = None,
                attention_mask: Optional[torch.LongTensor] = None,
                labels_embeddings: Optional[torch.FloatTensor] = None,
                labels_input_ids: Optional[torch.FloatTensor] = None,
                labels_attention_mask: Optional[torch.LongTensor] = None,
                words_embedding: Optional[torch.FloatTensor] = None,
                mask: Optional[torch.LongTensor] = None,
                prompts_embedding: Optional[torch.FloatTensor] = None,
                prompts_embedding_mask: Optional[torch.LongTensor] = None,
                words_mask: Optional[torch.LongTensor] = None,
                text_lengths: Optional[torch.Tensor] = None,
                labels: Optional[torch.FloatTensor] = None,
                **kwargs
                ):

        prompts_embedding, prompts_embedding_mask, words_embedding, mask = self.get_representations(input_ids, attention_mask,
                                                                            labels_embeddings, labels_input_ids, labels_attention_mask,
                                                                                                                text_lengths, words_mask)
        
        scores = self.scorer(words_embedding, prompts_embedding)

        loss = None
        if labels is not None:
            loss = self.loss(scores, labels, prompts_embedding_mask, mask, **kwargs)
        
        output = GLiNERModelOutput(
            logits=scores,
            loss=loss,
            prompts_embedding=prompts_embedding,
            prompts_embedding_mask=prompts_embedding_mask,
            words_embedding=words_embedding,
            mask=mask,
        )
        return output
    
    def loss(self, scores, labels, prompts_embedding_mask, mask, 
                        alpha: float = -1., gamma: float = 0.0, label_smoothing: float = 0.0, 
                        reduction: str = 'sum', **kwargs):
        all_losses = self._loss(scores, labels, alpha, gamma, label_smoothing)

        all_losses = all_losses * prompts_embedding_mask.unsqueeze(1) * mask.unsqueeze(-1)

        if reduction == "mean":
            loss = all_losses.mean()
        elif reduction == 'sum':
            loss = all_losses.sum()
        else:
            warnings.warn(
                f"Invalid Value for config 'loss_reduction': '{reduction} \n Supported reduction modes:"
                f" 'none', 'mean', 'sum'. It will be used 'sum' instead.")
            loss = all_losses.sum()
        return loss
