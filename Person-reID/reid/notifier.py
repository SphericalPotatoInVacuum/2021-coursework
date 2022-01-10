import notifiers


class Notifier:
    def __init__(self, token: str, chat_id: int):
        self.notifier = notifiers.get_notifier('telegram')
        self.token = token
        self.chat_id = chat_id

    def notify(self, *args, **kwargs):
        self.notifier.notify(*args, **kwargs, token=self.token, chat_id=self.chat_id)
