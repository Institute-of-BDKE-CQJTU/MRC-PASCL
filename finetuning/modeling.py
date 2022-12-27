import torch
from torch import nn
from torch.nn import Module, Linear, LayerNorm, CrossEntropyLoss
from torch.nn.parameter import Parameter

from transformers import BertPreTrainedModel, BertModel, RobertaModel
from transformers.modeling_bert import BertLMPredictionHead, ACT2FN
from transformers import PreTrainedTokenizer, BertTokenizer
from typing import Optional, Union, List, Dict, Tuple



def gather_positions(input_tensor, positions):
    """
    :param input_tensor: shape [batch_size, seq_length, dim]
    :param positions: shape [batch_size, num_positions]
    :return: [batch_size, num_positions, dim]
    """
    _, _, dim = input_tensor.size()
    index = positions.unsqueeze(-1).repeat(1, 1, dim)  # [batch_size, num_positions, dim]
    gathered_output = torch.gather(input_tensor, dim=1, index=index)  # [batch_size, num_positions, dim]
    return gathered_output


class FullyConnectedLayer(Module):
    def __init__(self, input_dim, output_dim, hidden_act="gelu"):
        super(FullyConnectedLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.dense = Linear(self.input_dim, self.output_dim)
        self.act_fn = ACT2FN[hidden_act]
        self.LayerNorm = LayerNorm(self.output_dim)

    def forward(self, inputs):
        temp = self.dense(inputs)
        temp = self.act_fn(temp)
        temp = self.LayerNorm(temp)
        return temp


class QuestionAwareSpanSelectionHead(Module):
    def __init__(self, config):
        super().__init__()

        self.query_start_transform = FullyConnectedLayer(config.hidden_size, config.hidden_size)
        self.query_end_transform = FullyConnectedLayer(config.hidden_size, config.hidden_size)
        self.start_transform = FullyConnectedLayer(config.hidden_size, config.hidden_size)
        self.end_transform = FullyConnectedLayer(config.hidden_size, config.hidden_size)

        self.start_classifier = Parameter(torch.Tensor(config.hidden_size, config.hidden_size))
        self.end_classifier = Parameter(torch.Tensor(config.hidden_size, config.hidden_size))

    def forward(self, inputs, positions):
        gathered_reps = gather_positions(inputs, positions)

        query_start_reps = self.query_start_transform(gathered_reps)  # [batch_size, num_positions, dim]
        query_end_reps = self.query_end_transform(gathered_reps)  # [batch_size, num_positions, dim]
        start_reps = self.start_transform(inputs)  # [batch_size, seq_length, dim]
        end_reps = self.end_transform(inputs)  # [batch_size, seq_length, dim]

        temp = torch.matmul(query_start_reps, self.start_classifier)  # [batch_size, num_positions, dim]
        start_reps = start_reps.permute(0, 2, 1)  # [batch_size, dim, seq_length]
        start_logits = torch.matmul(temp, start_reps)

        temp = torch.matmul(query_end_reps, self.end_classifier)
        end_reps = end_reps.permute(0, 2, 1)
        end_logits = torch.matmul(temp, end_reps)

        return start_logits, end_logits


class ClassificationHead(Module):
    def __init__(self, config):
        super().__init__()
        self.span_predictions = QuestionAwareSpanSelectionHead(config)

    def forward(self, inputs, positions):
        return self.span_predictions(inputs, positions)


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

class ModelWithQASSHead(BertPreTrainedModel):
    def __init__(self, config, replace_mask_with_question_token=False, 
                 mask_id=103, question_token_id=104, sep_id=102, initialize_new_qass=True):
        super().__init__(config)
        self.encoder_name = config.model_type
        if "roberta" in self.encoder_name:
            self.roberta = RobertaModel(config)
        else:
            self.bert = BertModel(config)
        self.initialize_new_qass = initialize_new_qass
        self.cls = ClassificationHead(config) if not self.initialize_new_qass else None
        self.new_cls = ClassificationHead(config) if self.initialize_new_qass else None

        self.awloss = AutomaticWeightedLoss(2)

        self.replace_mask_with_question_token = replace_mask_with_question_token
        self.mask_id = mask_id
        self.question_token_id = question_token_id
        self.sep_id = sep_id

        self.init_weights()

    def get_cls(self):
        return self.cls if not self.initialize_new_qass else self.new_cls

    def get_encoder(self):
        if "roberta" in self.encoder_name:
            return self.roberta
        return self.bert

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                negative_spans_index = None, negative_spans_weight = None, questions_idx = None,answers_idx = None,
                masked_positions=None, start_positions=None, end_positions=None):

        if attention_mask is not None:
            attention_mask[input_ids == self.sep_id] = 0
        if self.replace_mask_with_question_token:
            input_ids = input_ids.clone()
            input_ids[input_ids == self.mask_id] = self.question_token_id

        mask_positions_were_none = False
        if masked_positions is None:
            masked_position_for_each_example = torch.argmax((input_ids == self.question_token_id).int(), dim=-1)
            masked_positions = masked_position_for_each_example.unsqueeze(-1)
            mask_positions_were_none = True

        encoder = self.get_encoder()
        outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)

        sequence_output = outputs[0]  # [batch_size, max_length, dim]
        
        if negative_spans_index is not None:
            embedding_size = sequence_output.size(-1)
            questions_idx = questions_idx.unsqueeze(-1).expand(-1, -1, embedding_size)
            questions_embedding = torch.gather(sequence_output, 1, questions_idx).view(-1, embedding_size*2)
            # [bs, 2*dim]
            
            answers_idx = answers_idx.unsqueeze(-1).expand(-1, -1, embedding_size)
            answers_embedding = torch.gather(sequence_output, 1, answers_idx).view(-1, embedding_size*2)
            # [bs, 2*dim]
            
            negative_spans_index_dim_0, negative_spans_index_dim_1 = negative_spans_index.size(0), negative_spans_index.size(1)
            negative_spans_index = negative_spans_index.view(negative_spans_index_dim_0, -1).unsqueeze(-1).expand(-1, -1, embedding_size)
            negative_spans_embedding = torch.gather(sequence_output, -1, negative_spans_index).view(negative_spans_index_dim_0, negative_spans_index_dim_1, -1)
            # [bs, 5, dim*2]
            negative_spans_number = negative_spans_embedding.size(1)

            temperature = 0.05

            sim = Similarity(temp=temperature)
            positive_scores = sim(questions_embedding, answers_embedding)
            positive_score = positive_scores.sum()

            negative_score = torch.log(torch.cat((negative_spans_weight*torch.exp(sim(questions_embedding.unsqueeze(1), negative_spans_embedding)), torch.exp(positive_scores.unsqueeze(-1))), 1).sum(dim=-1).squeeze(-1)).sum()

            cl_span_loss = 1/(negative_spans_number+1) * (negative_score - positive_score)


        cls = self.get_cls()
        start_logits, end_logits = cls(sequence_output, masked_positions)

        if mask_positions_were_none:
            start_logits, end_logits = start_logits.squeeze(1), end_logits.squeeze(1)

        if attention_mask is not None:
            start_logits = start_logits + (1 - attention_mask) * -10000.0
            end_logits = end_logits + (1 - attention_mask) * -10000.0

        outputs = outputs[2:]
        outputs = (start_logits, end_logits, ) + outputs

        if start_positions is not None and end_positions is not None:
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions.long())
            end_loss = loss_fct(end_logits, end_positions.long())

            total_loss = (start_loss + end_loss) / 2
            if negative_spans_index is not None:
                total_loss = self.awloss(total_loss, cl_span_loss)

            outputs = (total_loss,) + outputs

        return outputs


