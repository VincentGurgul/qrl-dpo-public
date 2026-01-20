''' Utils package with telegram bot related functions. '''

import requests

bot_token = '<your_bot_token>'
api_url = f'https://api.telegram.org/bot{bot_token}/'
chat_id = '<your_chat_id>'


def send_message(content):
    method = 'sendMessage'
    params = {'chat_id': chat_id,
              'text': content}
    response = requests.post(api_url + method, params)
    if response.status_code == 200:
        print(f'Message sent: {content}')
    else:
        print(f'Something went wrong. ({response.status_code})')
