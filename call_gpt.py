import random
import time
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI
import openai


API_INFOS = {
    "OpenAI-GPT-4o": [
        {
            "endpoints": "https://conversationhubeastus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o"
        },
        {
            "endpoints": "https://conversationhubeastus2.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o"
        },
        {
            "endpoints": "https://conversationhubnorthcentralus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o"
        },
        {
            "endpoints": "https://conversationhubsouthcentralus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o"
        },
        {
            "endpoints": "https://conversationhubwestus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o"
        },
        {
            "endpoints": "https://conversationhubwestus3.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o"
        },
    ],
    "OpenAI-GPT-4o-mini": [
        {
            "endpoints": "https://conversationhubeastus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o-mini"
        },
        {
            "endpoints": "https://conversationhubswedencentral.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o-mini"
        },
        {
            "endpoints": "https://readinswedencentral.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o-mini"
        },
        {
            "endpoints": "https://readineastus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o-mini"
        },
    ],
    "OpenAI-GPT-4o-mini-0903-gen-4th": [
        {
            "endpoints": "https://readinnorthcentralus.openai.azure.com/",
            "speed": 150,
            "model": "gpt-4o-mini-2024-07-18-qilong-0903-gen-4th"
        },
    ],
}


class Openai():
    def __init__(
            self,
            apis,
            identity_id="18731b9d-0488-49be-bb05-6ccb08f78cf3",

    ):
        self.identity_id = identity_id
        flag = True
        while flag:
            try:
                self.token_provider = get_bearer_token_provider(
                    DefaultAzureCredential(managed_identity_client_id=self.identity_id),
                    "https://cognitiveservices.azure.com/.default"
                )
                flag = False
                break
            except:
                continue

        self.clients_weight = [apis[i]['speed'] for i in range(len(apis))]
        weight_sum = sum(self.clients_weight)
        for i in range(len(self.clients_weight)):
            self.clients_weight[i] /= weight_sum


        selected_api = random.choices(apis, weights=self.clients_weight, k=1)[0]

        self.client = AzureOpenAI(
            azure_endpoint=selected_api['endpoints'],
            azure_ad_token_provider=self.token_provider,
            api_version="2024-04-01-preview",
            max_retries=0,
        )
        self.model = selected_api['model']


    def call(self, content, client_index=None, return_logits=False, max_tokens=512):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": f"{content}\n"},
            ]
             },
        ]

        client = self.client
        model = self.model

        max_retry = 5
        cur_retry = 0
        while cur_retry <= max_retry:
            try:
                if not return_logits:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=max_tokens,
                        frequency_penalty=0,
                        presence_penalty=0,
                        stop=None
                    )

                    # client.chat.completions.with_raw_response
                    results = completion.choices[0].message.content
                else:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        logprobs=True,
                        top_logprobs=20,
                        temperature=0.0,
                        max_tokens=max_tokens,
                        frequency_penalty=0,
                        presence_penalty=0,
                        stop=None
                    )

                    # client.chat.completions.with_raw_response
                    results = completion.choices[0].logprobs.content[-1].top_logprobs
                return results
            except openai.RateLimitError as e:
                time.sleep(1)
            except Exception as e:
                print(e)
                cur_retry += 1
        return ""

if __name__ == '__main__':
    oai_clients = Openai(
        apis=API_INFOS["OpenAI-GPT-4o-mini-0903-gen-4th"]
    )
    res = oai_clients.call("hello", return_logits=True, max_tokens=1)
    print(res)


    # for i in range(len(API_INFOS)):
    #     print(API_INFOS[i])
    #     oai_clients = Openai(
    #         apis=[API_INFOS[i]]
    #     )
    #     res = oai_clients.call("hello")
    #     # res = oai_clients.call(text)
    #     print(res)