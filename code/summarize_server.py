# -*- coding: utf-8 -*-
import socket
from _thread import *
import json
import pickle
import argparse
from pathlib import Path
from transformers import AutoTokenizer, BartConfig, BartForConditionalGeneration
import torch

def news_summarization(doc_with_newlines):
    doc = doc_with_newlines.replace("\n", "")
    print(doc)
    raw_input_ids = tokenizer.encode(doc, max_length=1024)
    input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]
    summary_ids = model.generate(torch.tensor([input_ids]).to(device),
                                 max_length=100,
                                 num_beams=3,
                                 length_penalty=0.6
                                 )
    summary_ids = summary_ids.squeeze().tolist()
    print(summary_ids)
    summary_text = tokenizer.decode(summary_ids, skip_special_tokens=True)


    return summary_text + '\n'


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

            print('>> Received from ' + addr[0], ':', addr[1], document)

            news_summary = news_summarization(document)
            print(news_summary)
            # 클라이언트한테 결과 반환
            client_socket.send(news_summary.encode())

        except ConnectionResetError as e:
            print('>> Disconnected by ' + addr[0], ':', addr[1])
            break

    if client_socket in client_sockets:
        client_sockets.remove(client_socket)

    client_socket.close()


# Load ML Model
model_dir = 'hyunwoongko/kobart'
model_config = BartConfig.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = BartForConditionalGeneration.from_pretrained(model_dir, config=model_config)
checkpoint = torch.load('/Users/baeseonghyun/Desktop/summarize/r3f_test7_model.ckpt', map_location=lambda storage, loc: storage)['state_dict']

for key in list(checkpoint.keys()):
    if 'model.' in key:
        checkpoint[key.replace('model.', '', 1)] = checkpoint[key]
        del checkpoint[key]

model.load_state_dict(checkpoint)
model.eval()
torch.set_grad_enabled(False)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

client_sockets = []  # 서버에 접속한 클라이언트 목록

# 서버 IP 및 열어줄 포트
HOST = '127.0.0.1'
PORT = 12044

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

except Exception as e:
    print('에러 : ', e)

finally:
    server_socket.close()