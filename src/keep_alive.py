import threading
import urllib.request
import os


def keep_alive():
    threading.Timer(60, keep_alive).start()
    try:
        url = os.environ.get('URL')
        if url is None:
            print('Missing url!')
        else:
            print('Requesting.')
            print(urllib.request.urlopen(url).read())
    except Exception as e:
        print('An error occurred!')
        print(str(e))


keep_alive()
