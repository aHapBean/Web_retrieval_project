from transformers import BertModel, BertConfig
import torch.nn as nn

class CustomBertForSequenceClassification(nn.Module):
    def __init__(self, pretrained_model_name, num_labels=2, dropout_prob=0.1):
        super(CustomBertForSequenceClassification, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)

        # Extract hidden size from the BERT configuration
        hidden_size = self.bert.config.hidden_size

        # Add dropout and a linear layer for classification
        self.dropout = nn.Dropout(dropout_prob)
        # self.classifier1 = nn.Linear(hidden_size, num_labels)    # 
        self.classifier1 = nn.Linear(hidden_size, hidden_size // 2)
        self.classifier2 = nn.Linear(hidden_size // 2, num_labels)
        # self.conv1d = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size // 2, kernel_size=3)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier1(pooled_output)
        # logits = nn.ReLU()(logits)
        logits = self.classifier2(logits)
        return logits

# source code
# import torch
# import torch.nn as nn
# from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
# from typing import Optional, Tuple, Union
# from transformers import BertModel, BertPreTrainedModel
# from transformers.file_utils import SequenceClassifierOutput

# from transformers.models.bert.modeling_bert import BertPreTrainedModel, BERT_INPUTS_DOCSTRING, add_code_sample_docstrings, _CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION, _CONFIG_FOR_DOC, _SEQ_CLASS_EXPECTED_OUTPUT, _SEQ_CLASS_EXPECTED_LOSS

# class BertForSequenceClassification(BertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         self.config = config

#         self.bert = BertModel(config)
#         classifier_dropout = (
#             config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
#         )
#         self.dropout = nn.Dropout(classifier_dropout)
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)

#         # Initialize weights and apply final processing
#         self.post_init()

#     @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
#     @add_code_sample_docstrings(
#         checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
#         output_type=SequenceClassifierOutput,
#         config_class=_CONFIG_FOR_DOC,
#         expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
#         expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
#     )
#     def forward(
#         self,
#         input_ids: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         token_type_ids: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.Tensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         inputs_embeds: Optional[torch.Tensor] = None,
#         labels: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#             Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
#             config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
#             `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         pooled_output = outputs[1]

#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)

#         loss = None
#         if labels is not None:
#             if self.config.problem_type is None:
#                 if self.num_labels == 1:
#                     self.config.problem_type = "regression"
#                 elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
#                     self.config.problem_type = "single_label_classification"
#                 else:
#                     self.config.problem_type = "multi_label_classification"

#             if self.config.problem_type == "regression":
#                 loss_fct = MSELoss()
#                 if self.num_labels == 1:
#                     loss = loss_fct(logits.squeeze(), labels.squeeze())
#                 else:
#                     loss = loss_fct(logits, labels)
#             elif self.config.problem_type == "single_label_classification":
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             elif self.config.problem_type == "multi_label_classification":
#                 loss_fct = BCEWithLogitsLoss()
#                 loss = loss_fct(logits, labels)
#         if not return_dict:
#             output = (logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output

#         return SequenceClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )

