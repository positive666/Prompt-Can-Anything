from ChatGPT.GPT import Chatbot
from ChatGPT.config.private import API_KEY,PROXIES
chatbot=Chatbot(api_key=API_KEY,proxy=PROXIES,engine="gpt-3.5-turbo")

for data in chatbot.ask_stream(prompt="Hello world"):
     print(data,end='',flush=True)

def check_caption(caption, pred_phrases, max_tokens=100, model="gpt-3.5-turbo"):
    object_list = [obj.split('(')[0] for obj in pred_phrases]
    object_num = []
    for obj in set(object_list):
        object_num.append(f'{object_list.count(obj)} {obj}')
    object_num = ', '.join(object_num)
    print(f"Correct object number: {object_num}")

    prompt = [
        {
            'role': 'system',
            'content': 'Revise the number in the caption if it is wrong. ' + \
                       f'Caption: {caption}. ' + \
                       f'True object number: {object_num}. ' + \
                       'Only give the revised caption: '
        }
    ]
    for data in chatbot.ask_stream(prompt=prompt):
          print(data,end='',flush=True)

          reply = data['choices'][0]['message']['content']
          caption = reply.split(':')[-1].strip()
          return caption