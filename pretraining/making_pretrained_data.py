import spacy
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('pretrained_model/splinter')

nlp = spacy.load('en_core_web_sm')

sentence_to_doc = dict()

STOPWORDS = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out',
             'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into',
             'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the',
             'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were',
             'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to',
             'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have',
             'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can',
             'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
             'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by',
             'doing', 'it', 'how', 'further', 'was', 'here', 'than', 'also', 'could', 'would']

pretrain_data = []

def detect_data(sentence:str):
    start_sent_tokens = tokenizer.tokenize(sentence)
    start_sent_doc = nlp(sentence)
    noun_spans = [chunk.text for chunk in start_sent_doc.noun_chunks]
    for ent in start_sent_doc.ents:
        noun_spans.append(ent.text)
    noun_spans = list(set(noun_spans))
    temp_dict = dict()
    for noun_span in noun_spans:
        if noun_span in STOPWORDS:
            continue
        span_tokens = tokenizer.tokenize(noun_span)
        if len(span_tokens) == 0:
            continue
        flag = 0
        for i in range(len(start_sent_tokens)):
            if start_sent_tokens[i] == span_tokens[0] and i + len(span_tokens) < len(start_sent_tokens):
                flag = 1
                for j in range(i+1, i+len(span_tokens)):
                    if start_sent_tokens[j] != span_tokens[j-i]:
                        flag = 0
                        break
            if flag == 1:
                temp_dict[noun_span] = [i, i+len(span_tokens)-1]
                break
    for key in temp_dict.keys():
        span_tokens = tokenizer.tokenize(key)
        flag = 0
        for i in range(len(sentence_to_doc[sentence])):
            if 1 + len(start_sent_tokens) + 1 + i + len(span_tokens) > 511:
                break
            if span_tokens[0] == sentence_to_doc[sentence][i] and i + len(span_tokens) < len(sentence_to_doc[sentence]):
                flag = 1
                for j in range(i+1, i+len(span_tokens)):
                    if span_tokens[j-i] != sentence_to_doc[sentence][j]:
                        flag = 0
                        break
            if flag == 1:
                temp_dict[key].extend([i, i+len(span_tokens)-1])
                break
    # 直接做成数据集
    number = 0
    for key in temp_dict.keys():
        if len(temp_dict[key]) == 4:
            number += 1
            question_start, question_end, answer_start, answer_end = temp_dict[key]
            input_tokens = ['[CLS]'] + start_sent_tokens[:question_start] + ['[QUESTION]']
            attention_mask = [1]*len(input_tokens)
            span_tokens = tokenizer.tokenize(key)
            for _ in range(len(span_tokens)-1):
                input_tokens.append('[PAD]')
                attention_mask.append(0)
            input_tokens += start_sent_tokens[question_end+1:] + ['[SEP]'] + sentence_to_doc[sentence][:509-len(start_sent_tokens)] + ['[SEP]']
            attention_mask.extend([1]*(len(input_tokens) - len(attention_mask)))
            token_type_ids = [0]*(len(start_sent_tokens)+2) + [1]*(len(input_tokens) - 2 - len(start_sent_tokens))
            input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
            for _ in range(512 - len(input_tokens)):
                input_ids.append(0)
                attention_mask.append(0)
                token_type_ids.append(0)
            pretrain_data.append([input_ids, attention_mask, token_type_ids, answer_start+2+len(start_sent_tokens), answer_end+2+len(start_sent_tokens)])

    return number

import pickle
file_number = 1

text_files = ['data/pretrain_text/ccnews_text_250.txt', 'data/pretrain_text/openwebtext_250.txt']
total_number = 0
for text_file in text_files:
    with open(text_file, 'r', encoding='UTF-8') as f:
        for line in f:
            line = line.strip()
            doc = nlp(line)
            sentence_to_doc.clear()
            for sent in doc.sents:
                sentence_to_doc[sent.text] = tokenizer.tokenize(line[sent.end_char:].strip(' '))
                if len(sentence_to_doc[sent.text]) < 400:
                    break
                total_number += detect_data(sent.text)
            print(total_number)
            if total_number // 192000 == file_number:
                with open('data/pretrain_data/pretrain_data_{}.pkl'.format(str(file_number)), 'wb') as f:
                    pickle.dump(pretrain_data[:192000], f)
                file_number += 1
                temp = pretrain_data[192000:]
                pretrain_data.clear()
                pretrain_data = temp
            
