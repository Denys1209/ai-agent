import base64
import argparse
from concurrent import futures
import threading
import re

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Gemma4ChatHandler

import grpc
import chat_pb2
import chat_pb2_grpc


class MyGemma4Handler():

    def __init__(self):
        print("Loading model, this can take a little while...")


        self.MAX_OUTPUT_TOKENS = 4096
        self.THINK_PATTERN = re.compile(r"<\|channel>thought\n(.*?)<channel\|>", re.DOTALL)

        chat_handler = Gemma4ChatHandler(
            clip_model_path = ".\\aiModel\\mmproj-F16.gguf",
            enable_thinking = True
        )

        self.llm = Llama(
            model_path = ".\\aiModel\\gemma-4-E4B-it-Q8_0.gguf",
            chat_handler = chat_handler,
            n_gpu_layers = 'all',
            n_ctx = 16384,
            verbose = False
        )

        self.llmlock = threading.Lock()

        print("Model loaded. Type your message and press enter. Type 'stop' to quite\n")
    
    def imageBytesToDataUrl(self, data: bytes, minType: str) -> str:
        b64 = base64.b64encode(data).decode("utf-8")

        return f"data:{minType};base64,{b64}"


    def splitThinking(self, fullText:str) -> tuple[str, str]:
        match = self.THINK_PATTERN.search(fullText)
        thinking = match.group(1).strip() if match else ""
        
        answer = self.THINK_PATTERN.sub("", fullText).strip()

        return thinking, answer
    
    def messageToDict(self, msg:chat_pb2.Message):
        if len(msg.imageAttachment) > 0:
            content = []

            for img in msg.imageAttachment:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": self.imageBytesToDataUrl(img.data, img.mimeType)
                    })
            if msg.text:
                content.append({"type": "text", "text": msg.text})
        else:
            content = msg.text
        
        return {
            "role": chat_pb2.Roles.Name(msg.role).lower(),
            "content": content
        }
    
    def processHistory(self, history: chat_pb2.History):
        messages = list()
        for i in history.messages:
            messages.append(self.messageToDict(i))
        return messages
    
    def buildMessage(self, history: chat_pb2.History, userMessage:chat_pb2.Message):
        messages = self.processHistory(history)

        messages.append(self.messageToDict(userMessage))

        return messages
    
    def makeRequestToModel(self, Request: chat_pb2.Request):
        messages = self.buildMessage(Request.history, Request.userMessage)
        print("Assistant: ", end="", flush=True)
        fullReply = ""
        with self.llmlock: 
            for chunk in self.llm.create_chat_completion(messages=messages, max_tokens=self.MAX_OUTPUT_TOKENS, stream=True):
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    print(delta, end="", flush=True)
                    fullReply += delta
        
        thoughts, answer = self.splitThinking(fullReply)

        return thoughts, answer
    
    def makeRequestToModelGetStream(self, Request: chat_pb2.Request):
        messages = self.buildMessage(Request.history, Request.userMessage)
        print("Assistant: ", end="", flush=True)
        with self.llmlock:
            for chunk in self.llm.create_chat_completion(messages=messages, max_tokens=self.MAX_OUTPUT_TOKENS, stream=True):
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    print(delta, end="", flush=True)
                    yield chat_pb2.Response(answer=delta, thoughts="");

class Gemma4(chat_pb2_grpc.Gemma4Server):

    def __init__(self, handler:MyGemma4Handler):
        super().__init__()
        self.modelHandler = handler
    
    def MakeRequest(self, request, context):
        thoughts, answer = self.modelHandler.makeRequestToModel(request)

        return chat_pb2.Response(answer=answer, thoughts=thoughts)
    
    def MakeRequestStreamBackTokenByToken(self, request, context):
        return self.modelHandler.makeRequestToModelGetStream(request)
        

def serve():
    port = "50051"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    chat_pb2_grpc.add_Gemma4ServerServicer_to_server(Gemma4(MyGemma4Handler()), server)

    server.add_insecure_port("[::]:" + port)
    server.start()

    print("Server started, listening on " + port)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()

