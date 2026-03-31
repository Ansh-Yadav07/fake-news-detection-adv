import urllib.request
import os
import ssl
from dotenv import load_dotenv

load_dotenv()
url = "https://api-inference.huggingface.co/pipeline/text-classification/anshy047/fake-news-detector-transformer"
data = b'{"inputs":"Sample news"}'
req = urllib.request.Request(url, data=data, headers={"Authorization": f"Bearer {os.environ.get('HF_TOKEN')}", "Content-Type": "application/json"})
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
try:
    with urllib.request.urlopen(req, context=ctx) as response:
        print(response.status)
        print(response.read().decode())
except urllib.error.HTTPError as e:
    print(e.code)
    print(e.read().decode())
