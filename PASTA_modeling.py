from typing import List, Optional, Tuple, Union


from transformers.models.bert.modeling_bert import BertPreTrainedModel,BertModel,BertEncoder,BertLayer, BertEmbeddings, BertPooler, BaseModelOutputWithPoolingAndCrossAttentions, BaseModelOutputWithPastAndCrossAttentions, BertForSequenceClassification, BertForTokenClassification
import torch
from torch import nn


class BertPastaEmbeddings(BertEmbeddings):
    def __init__(self, config):
        super().__init__(config)

    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        embeddings = super().forward(input_ids, token_type_ids,position_ids,inputs_embeds, past_key_values_length)
        ## get the positions of special tokens 
        is_cls_token = (input_ids == 101).long().to(input_ids.device) # shape (batch_size, seq_length)
        is_sep_token = (input_ids == 102).long().to(input_ids.device) # shape (batch_size, seq_length)
        special_token_positions = is_cls_token + is_sep_token # shape (batch_size, seq_length)
        num_specials = torch.sum(special_token_positions[0]).item()
        token_indices = special_token_positions.nonzero() # shape (num_specials*batch_size, 2)
        is_special_token_tensors = torch.zeros_like(special_token_positions).unsqueeze(dim=0).expand(num_specials, -1,-1).long().to(input_ids.device) # shape (num_specials, batch_size, seq_length), with each row being a one-hot tensor for a particular special token
        for i in range(len(token_indices)):
            current_token = i % num_specials
            is_special_token_tensors[current_token][token_indices[i][0],token_indices[i][1]] = 1
        
        return embeddings, num_specials, is_special_token_tensors






class BertPastaLayer(BertLayer):
    def __init__(self, config, num_special_tokens):
        super().__init__(config)
        self.config = config
        ## new attributes for PASTA layer
        self.num_special = num_special_tokens
        self.special_token_embedding = nn.Embedding(num_special_tokens, self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
    
    def forward(
        self,
        hidden_states: torch.Tensor,  # bsz x seq_len x dim
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        special_token_indices: Optional[list] = None,
    ) -> Tuple[torch.Tensor]:
        ## add trainable vectors to special tokens
        if special_token_indices is not None:
            assert special_token_indices.shape[0] <= self.num_special, "Initialized vectors should not be less than the number of special tokens in input!"
            for i in range(special_token_indices.shape[0]):
                special_token = self.special_token_embedding(torch.tensor([i]).to(next(self.parameters()).device)).squeeze()
                special_token_index = special_token_indices[i].unsqueeze(-1).expand(-1,-1, self.config.hidden_size)
                special_token_tensor = self.dropout(special_token * special_token_index)
                hidden_states = hidden_states + special_token_tensor
        
        return super().forward(
                hidden_states,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
        )


class BertPastaEncoder(BertEncoder):
    def __init__(self, config, num_special_tokens):
        super().__init__(config)
        self.layer = nn.ModuleList([BertPastaLayer(config, num_special_tokens) for _ in range(config.num_hidden_layers)])
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        is_special_token_tensors: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    print(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    is_special_token_tensors,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    is_special_token_tensors,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )



class BertPastaModel(BertModel):
    def __init__(self, config, num_special_tokens, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.embeddings = BertPastaEmbeddings(config)
        self.encoder = BertPastaEncoder(config, num_special_tokens)
        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output, num_specials, is_special_token_tensors = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_special_token_tensors=is_special_token_tensors,
        )
        sequence_output = encoder_outputs[0]
        
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None


        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )



class BertPastaForSequenceClassification(BertForSequenceClassification):
    def __init__(self,config, num_special_tokens):
        super().__init__(config)
        self.bert = BertPastaModel(config,num_special_tokens)


class BertPastaForTokenClassification(BertForTokenClassification):
    def __init__(self,config, num_special_tokens):
        super().__init__(config)
        self.bert = BertPastaModel(config,num_special_tokens)