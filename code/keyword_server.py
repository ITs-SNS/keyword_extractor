# -*- coding: utf-8 -*-
import socket
from _thread import *
import json
import pickle
import argparse
import torch
import re
from kobert import get_pytorch_kobert_model
from kobert import get_tokenizer
from model.net import KobertSequenceFeatureExtractor, KobertCRF, KobertBiLSTMCRF, KobertBiGRUCRF
from gluonnlp.data import SentencepieceTokenizer
from data_utils.utils import Config
from data_utils.vocab_tokenizer import Tokenizer
from data_utils.pad_sequence import keras_pad_fn
from pathlib import Path

import numpy as np
import itertools

from konlpy.tag import Hannanum
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

hannanum = Hannanum()
#한국어 불용어
stop_words="아 휴 아이구 아이쿠 아이고 어 나 우리 저희 따라 의해 을 를 에 의 가 으로 로 에게 뿐이다 의거하여 근거하여 입각하여 기준으로 예하면 예를 들면 예를 들자면 저 소인 소생 저희 지말고 하지마 하지마라 다른 물론 또한 그리고 비길수 없다 해서는 안된다 뿐만 아니라 만이 아니다 만은 아니다 막론하고 관계없이 그치지 않다 그러나 그런데 하지만 든간에 논하지 않다 따지지 않다 설사 비록 더라도 아니면 만 못하다 하는 편이 낫다 불문하고 향하여 향해서 향하다 쪽으로 틈타 이용하여 타다 오르다 제외하고 이 외에 이 밖에 하여야 비로소 한다면 몰라도 외에도 이곳 여기 부터 기점으로 따라서 할 생각이다 하려고하다 이리하여 그리하여 그렇게 함으로써 하지만 일때 할때 앞에서 중에서 보는데서 으로써 로써 까지 해야한다 일것이다 반드시 할줄알다 할수있다 할수있어 임에 틀림없다 한다면 등 등등 제 겨우 단지 다만 할뿐 딩동 댕그 대해서 대하여 대하면 훨씬 얼마나 얼마만큼 얼마큼 남짓 여 얼마간 약간 다소 좀 조금 다수 몇 얼마 지만 하물며 또한 그러나 그렇지만 하지만 이외에도 대해 말하자면 뿐이다 다음에 반대로 반대로 말하자면 이와 반대로 바꾸어서 말하면 바꾸어서 한다면 만약 그렇지않으면 까악 툭 딱 삐걱거리다 보드득 비걱거리다 꽈당 응당 해야한다 에 가서 각 각각 여러분 각종 각자 제각기 하도록하다 와 과 그러므로 그래서 고로 한 까닭에 하기 때문에 거니와 이지만 대하여 관하여 관한 과연 실로 아니나다를가 생각한대로 진짜로 한적이있다 하곤하였다 하 하하 허허 아하 거바 와 오 왜 어째서 무엇때문에 어찌 하겠는가 무슨 어디 어느곳 더군다나 하물며 더욱이는 어느때 언제 야 이봐 어이 여보시오 흐흐 흥 휴 헉헉 헐떡헐떡 영차 여차 어기여차 끙끙 아야 앗 아야 콸콸 졸졸 좍좍 뚝뚝 주룩주룩 솨 우르르 그래도 또 그리고 바꾸어말하면 바꾸어말하자면 혹은 혹시 답다 및 그에 따르는 때가 되어 즉 지든지 설령 가령 하더라도 할지라도 일지라도 지든지 몇 거의 하마터면 인젠 이젠 된바에야 된이상 만큼 어찌됏든 그위에 게다가 점에서 보아 비추어 보아 고려하면 하게될것이다 일것이다 비교적 좀 보다더 비하면 시키다 하게하다 할만하다 의해서 연이서 이어서 잇따라 뒤따라 뒤이어 결국 의지하여 기대여 통하여 자마자 더욱더 불구하고 얼마든지 마음대로 주저하지 않고 곧 즉시 바로 당장 하자마자 밖에 안된다 하면된다 그래 그렇지 요컨대 다시 말하자면 바꿔 말하면 즉 구체적으로 말하자면 시작하여 시초에 이상 허 헉 허걱 바와같이 해도좋다 해도된다 게다가 더구나 하물며 와르르 팍 퍽 펄렁 동안 이래 하고있었다 이었다 에서 로부터 까지 예하면 했어요 해요 함께 같이 더불어 마저 마저도 양자 모두 습니다 가까스로 하려고하다 즈음하여 다른 다른 방면으로 해봐요 습니까 했어요 말할것도 없고 무릎쓰고 개의치않고 하는것만 못하다 하는것이 낫다 매 매번 들 모 어느것 어느 로써 갖고말하자면 어디 어느쪽 어느것 어느해 어느 년도 라 해도 언젠가 어떤것 어느것 저기 저쪽 저것 그때 그럼 그러면 요만한걸 그래 그때 저것만큼 그저 이르기까지 할 줄 안다 할 힘이 있다 너 너희 당신 어찌 설마 차라리 할지언정 할지라도 할망정 할지언정 구토하다 게우다 토하다 메쓰겁다 옆사람 퉤 쳇 의거하여 근거하여 의해 따라 힘입어 그 다음 버금 두번째로 기타 첫번째로 나머지는 그중에서 견지에서 형식으로 쓰여 입장에서 위해서 단지 의해되다 하도록시키다 뿐만아니라 반대로 전후 전자 앞의것 잠시 잠깐 하면서 그렇지만 다음에 그러한즉 그런즉 남들 아무거나 어찌하든지 같다 비슷하다 예컨대 이럴정도로 어떻게 만약 만일 위에서 서술한바와같이 인 듯하다 하지 않는다면 만약에 무엇 무슨 어느 어떤 아래윗 조차 한데 그럼에도 불구하고 여전히 심지어 까지도 조차도 하지 않도록 않기 위하여 때 시각 무렵 시간 동안 어때 어떠한 하여금 네 예 우선 누구 누가 알겠는가 아무도 줄은모른다 줄은 몰랏다 하는 김에 겸사겸사 하는바 그런 까닭에 한 이유는 그러니 그러니까 때문에 그 너희 그들 너희들 타인 것 것들 너 위하여 공동으로 동시에 하기 위하여 어찌하여 무엇때문에 붕붕 윙윙 나 우리 엉엉 휘익 윙윙 오호 아하 어쨋든 만 못하다 하기보다는 차라리 하는 편이 낫다 흐흐 놀라다 상대적으로 말하자면 마치 아니라면 쉿 그렇지 않으면 그렇지 않다면 안 그러면 아니었다면 하든지 아니면 이라면 좋아 알았어 하는것도 그만이다 어쩔수 없다 하나 일 일반적으로 일단 한켠으로는 오자마자 이렇게되면 이와같다면 전부 한마디 한항목 근거로 하기에 아울러 하지 않도록 않기 위해서 이르기까지 이 되다 로 인하여 까닭으로 이유만으로 이로 인하여 그래서 이 때문에 그러므로 그런 까닭에 알 수 있다 결론을 낼 수 있다 으로 인하여 있다 어떤것 관계가 있다 관련이 있다 연관되다 어떤것들 에 대해 이리하여 그리하여 여부 하기보다는 하느니 하면 할수록 운운 이러이러하다 하구나 하도다 다시말하면 다음으로 에 있다 에 달려 있다 우리 우리들 오히려 하기는한데 어떻게 어떻해 어찌됏어 어때 어째서 본대로 자 이 이쪽 여기 이것 이번 이렇게말하자면 이런 이러한 이와 같은 요만큼 요만한 것 얼마 안 되는 것 이만큼 이 정도의 이렇게 많은 것 이와 같다 이때 이렇구나 것과 같이 끼익 삐걱 따위 와 같은 사람들 부류의 사람들 왜냐하면 중의하나 오직 오로지 에 한하다 하기만 하면 도착하다 까지 미치다 도달하다 정도에 이르다 할 지경이다 결과에 이르다 관해서는 여러분 하고 있다 한 후 혼자 자기 자기집 자신 우에 종합한것과같이 총적으로 보면 총적으로 말하면 총적으로 대로 하다 으로서 참 그만이다 할 따름이다 쿵 탕탕 쾅쾅 둥둥 봐 봐라 아이야 아니 와아 응 아이 참나 년 월 일 령 영 일 이 삼 사 오 육 륙 칠 팔 구 이천육 이천칠 이천팔 이천구 하나 둘 셋 넷 다섯 여섯 일곱 여덟 아홉 령 영 이 있 하 것 들 그 되 수 이 보 않 없 나 사람 주 아니 등 같 우리 때 년 가 한 지 대하 오 말 일 그렇 위하 때문 그것 두 말하 알 그러나 받 못하 일 그런 또 문제 더 사회 많 그리고 좋 크 따르 중 나오 가지 씨 시키 만들 지금 생각하 그러 속 하나 집 살 모르 적 월 데 자신 안 어떤 내 내 경우 명 생각 시간 그녀 다시 이런 앞 보이 번 나 다른 어떻 여자 개 전 들 사실 이렇 점 싶 말 정도 좀 원 잘 통하 놓"
stop_words=stop_words.split(' ')
post_position_words="은 는 이 가 의 을 를".split(' ')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# SBERT를 로드
sbert_model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
sbert_model = sbert_model.to(device)


class DecoderFromNamedEntitySequence():
    def __init__(self, tokenizer, index_to_ner):
        self.tokenizer = tokenizer
        self.index_to_ner = index_to_ner

    def __call__(self, list_of_input_ids, list_of_pred_ids):
        input_token = self.tokenizer.decode_token_ids(list_of_input_ids)[0]
        pred_ner_tag = [self.index_to_ner[pred_id] for pred_id in list_of_pred_ids[0]]

        # ----------------------------- parsing list_of_ner_word ----------------------------- #
        list_of_ner_word = []
        entity_word, entity_tag, prev_entity_tag = "", "", ""
        for i, pred_ner_tag_str in enumerate(pred_ner_tag):
            if "B-" in pred_ner_tag_str:
                entity_tag = pred_ner_tag_str[-3:]

                if prev_entity_tag != entity_tag and prev_entity_tag != "":
                    list_of_ner_word.append({"word": entity_word.replace("▁", " "), "tag": prev_entity_tag, "prob": None})

                entity_word = input_token[i]
                prev_entity_tag = entity_tag
            elif "I-"+entity_tag in pred_ner_tag_str:
                entity_word += input_token[i]
            else:
                if entity_word != "" and entity_tag != "":
                    list_of_ner_word.append({"word":entity_word.replace("▁", " "), "tag":entity_tag, "prob":None})
                entity_word, entity_tag, prev_entity_tag = "", "", ""


        # ----------------------------- parsing decoding_ner_sentence ----------------------------- #
        decoding_ner_sentence = ""
        is_prev_entity = False
        prev_entity_tag = ""
        is_there_B_before_I = False

        for token_str, pred_ner_tag_str in zip(input_token, pred_ner_tag):
            token_str = token_str.replace('▁', ' ')  # '▁' 토큰을 띄어쓰기로 교체

            if 'B-' in pred_ner_tag_str:
                if is_prev_entity is True:
                    decoding_ner_sentence += ':' + prev_entity_tag+ '>'

                if token_str[0] == ' ':
                    token_str = list(token_str)
                    token_str[0] = ' <'
                    token_str = ''.join(token_str)
                    decoding_ner_sentence += token_str
                else:
                    decoding_ner_sentence += '<' + token_str
                is_prev_entity = True
                prev_entity_tag = pred_ner_tag_str[-3:] # 첫번째 예측을 기준으로 하겠음
                is_there_B_before_I = True

            elif 'I-' in pred_ner_tag_str:
                decoding_ner_sentence += token_str

                if is_there_B_before_I is True: # I가 나오기전에 B가 있어야하도록 체크
                    is_prev_entity = True
            else:
                if is_prev_entity is True:
                    decoding_ner_sentence += ':' + prev_entity_tag+ '>' + token_str
                    is_prev_entity = False
                    is_there_B_before_I = False
                else:
                    decoding_ner_sentence += token_str

        return list_of_ner_word, decoding_ner_sentence

def mmr(doc_embedding, candidate_embeddings, words, top_n, diversity):

    # 문서와 각 키워드들 간의 유사도가 적혀있는 리스트
    word_doc_similarity = cosine_similarity(candidate_embeddings, doc_embedding)

    # 각 키워드들 간의 유사도
    word_similarity = cosine_similarity(candidate_embeddings)

    # 문서와 가장 높은 유사도를 가진 키워드의 인덱스를 추출.
    # 만약, 2번 문서가 가장 유사도가 높았다면
    # keywords_idx = [2]
    keywords_idx = [np.argmax(word_doc_similarity)]

    # 가장 높은 유사도를 가진 키워드의 인덱스를 제외한 문서의 인덱스들
    # 만약, 2번 문서가 가장 유사도가 높았다면
    # ==> candidates_idx = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10 ... 중략 ...]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    # 최고의 키워드는 이미 추출했으므로 top_n-1번만큼 아래를 반복.
    # ex) top_n = 5라면, 아래의 loop는 4번 반복됨.
    for _ in range(top_n - 1):
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # MMR을 계산
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # keywords & candidates를 업데이트
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [re.sub(r"[^가-힣a-zA-Z0-9\+\-\n\s]","", words[idx]) for idx in keywords_idx]

def keyword_extraction(doc_with_newlines):
    doc=doc_with_newlines.replace("\n", "")

    #문장을 분리하였지만, 아직 앞뒤 공백이 있음
    temp_texts =  doc.split(". ")

    #앞뒤 공백이 제거된 문자열들이 저장되는 list
    texts=[]
    for text in temp_texts:
        text = text.strip()
        text = re.sub(r"[^가-힣a-zA-Z0-9\+\-\n\(\)\.\,\s]","", text)
        if len(text) > 512:
            continue;
        if text is not '':
            texts.append(text)
    
    # 품사 태깅
    tokenized_doc = hannanum.pos(doc)

    #print(tokenized_doc)
    # 형태소 분석기를 통해 명사 추출한 문서를 만듦
    tokenized_nouns = ' '.join([word[0] for word in tokenized_doc
                        if (word[1] == 'N' or word[1] == 'F')
                        and not word[0].isdigit() 
                        and len(word[0])>1
                        and word[0] not in stop_words])

    n_gram_range = (1, 1)
    # CountVectorizer를 사용하여 n-gram을 추출(n_gram_range에 해당하는)
    count = CountVectorizer(ngram_range=n_gram_range).fit([tokenized_nouns])
    candidates = count.get_feature_names()

    #doc를 임베딩
    doc_embedding = sbert_model.encode([doc])
    #n_gram들을 임베딩
    candidate_embeddings = sbert_model.encode(candidates)

    org = set([])
    per = set([])
    poh = set([])
    for text in texts:
        list_of_input_ids = tokenizer.list_of_string_to_list_of_cls_sep_token_ids([text])
        ner_model_input = torch.tensor(list_of_input_ids).long()

        ## for bert alone
        # y_pred = model(x_input)
        # list_of_pred_ids = y_pred.max(dim=-1)[1].tolist()

        ## for bert crf
        list_of_pred_ids = ner_model(ner_model_input)

        list_of_ner_word, decoding_ner_sentence = decoder_from_res(list_of_input_ids=list_of_input_ids, list_of_pred_ids=list_of_pred_ids)

        for ner_word in list_of_ner_word:
            tag=ner_word['tag']
            word = ner_word['word'].strip()
            if("[UNK]" in word):
                continue
            word = re.sub(r"[^가-힣a-zA-Z0-9\+\-\n\s]","", word)
            if(tag == 'ORG' and len(word)>1):
               org.add(word)
            if(tag == 'PER' and len(word)>1):
                per.add(word)
            if(tag == 'POH' and len(word)>1):
                poh.add(word)

    # diversity값이 클수록, 키워드간의 유사성이 최소화 됨, 문서와 키워드 사이의 유사성은 극대화
    mmr_keywords=mmr(doc_embedding, candidate_embeddings, candidates, top_n=7, diversity=0.2)

    #공백이 있는 후보 키워드들을 저장
    space_candidate_keywords=set([])
    space_candidate_keywords = space_candidate_keywords.union(org)
    space_candidate_keywords = space_candidate_keywords.union(per)
    space_candidate_keywords = space_candidate_keywords.union(poh)
    space_candidate_keywords = space_candidate_keywords.union(mmr_keywords)

    #후보 키워드들의 공백을 없앰
    candidate_keywords=set([])
    for keyword in space_candidate_keywords:
        keyword = keyword.replace(' ', '')
        if len(keyword) > 0:
            candidate_keywords.add(keyword)

    #후보키워드들 중  상위 구함
    doc_without_space=doc.replace(' ', '')
    count_dict = set([(token, doc_without_space.count(token)) for token in candidate_keywords ])
    ranked_words = sorted(count_dict, key=lambda x:x[1], reverse=True)[:15]
    
    predict_keywords = list([keyword for keyword, freq in ranked_words 
            if keyword.strip()[-1] not in post_position_words])

    return ' '.join(list(predict_keywords))+'\n'

# 쓰레드에서 실행되는 코드.
# 접속한 클라이언트마다 새로운 쓰레드가 생성되어 통신
def threaded(client_socket, addr):
    print('>> Connected by :', addr[0], ':', addr[1])

    # 클라이언트가 접속을 끊을 때 까지 반복
    while True:

        try:
            print("수신")
            # 데이터가 수신되면 클라이언트에 다시 전송합니다.(에코)
            document = ""
            while '\n' not in document:
                document += client_socket.recv(100000).decode()

            if not document:
                print('>> Disconnected by ' + addr[0], ':', addr[1])
                break

            print('>> Received from ' + addr[0], ':', addr[1])
            print('>> document : ', document)
            predict_keywords = keyword_extraction(document) 
            print('>> keywords : ',predict_keywords)           
            # 클라이언트한테 결과 반환
            client_socket.send(predict_keywords.encode())

        except ConnectionResetError as e:
            print('>> Disconnected by ' + addr[0], ':', addr[1])
            break

    if client_socket in client_sockets :
        client_sockets.remove(client_socket)

    client_socket.close()


#parser, ner_model경로 설정

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data_in', help="Directory containing config.json of data")
parser.add_argument('--model_dir', default='./experiments/base_model_with_crf_val', help="Directory containing config.json of model")
args, _ = parser.parse_known_args()
ner_model_dir = Path(args.model_dir)
ner_model_config = Config(json_path=ner_model_dir / 'config.json')


# Vocab & Tokenizer
# tok_path = get_tokenizer() # ./tokenizer_78b3253a26.model
tok_path = "./ptr_lm_model/tokenizer_78b3253a26.model"
ptr_tokenizer = SentencepieceTokenizer(tok_path)
print(ner_model_dir)

# load vocab & tokenizer
with open(ner_model_dir / "vocab.pkl", 'rb') as f:
    vocab = pickle.load(f)

tokenizer = Tokenizer(vocab=vocab, split_fn=ptr_tokenizer, pad_fn=keras_pad_fn, maxlen=ner_model_config.maxlen)

# load ner_to_index.json
with open(ner_model_dir / "ner_to_index.json", 'rb') as f:
    ner_to_index = json.load(f)
    index_to_ner = {v: k for k, v in ner_to_index.items()}

ner_model = KobertCRF(config=ner_model_config, num_classes=len(ner_to_index), vocab=vocab)

# load
ner_model_dict = ner_model.state_dict()
checkpoint = torch.load("./experiments/base_model_with_crf_val/best-epoch-9-step-750-acc-0.957.bin", map_location=torch.device('cpu'))

convert_keys = {}
for k, v in checkpoint['model_state_dict'].items():
    new_key_name = k.replace("module.", '')
    if new_key_name not in ner_model_dict:
        print("{} is not int model_dict".format(new_key_name))
        continue
    convert_keys[new_key_name] = v

ner_model.load_state_dict(convert_keys)
ner_model.eval()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# n_gpu = torch.cuda.device_count()
# if n_gpu > 1:
#     model = torch.nn.DataParallel(model)
ner_model.to(device)

decoder_from_res = DecoderFromNamedEntitySequence(tokenizer=tokenizer, index_to_ner=index_to_ner)



client_sockets = [] # 서버에 접속한 클라이언트 목록

# 서버 IP 및 열어줄 포트
HOST = '127.0.0.1'
PORT = 8000

# 서버 소켓 생성
print('>> Server Start')
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen()

try:
    while True:
        print('>> 새로운 클라이언트 wait')

        client_socket, addr = server_socket.accept()
        client_sockets.append(client_socket)
        start_new_thread(threaded, (client_socket, addr))
        print("클라이언트 수 : ", len(client_sockets))
        
except Exception as e :
    print ('에러 : ',e)

finally:
    server_socket.close()