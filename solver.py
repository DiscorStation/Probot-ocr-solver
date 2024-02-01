import requests
import torch
from PIL import Image
from colorama import init, Fore
import time

init()

# Model
print(Fore.CYAN + 'Loading model')
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/best.pt')
print(Fore.GREEN + 'Model loaded')


def solveCaptcha(url, color = None) -> str:
  try:
    img = Image.open(requests.get(url, stream=True).raw)

    # Time taken to load image
    start_time = time.time()

    result = model(img)

    # Time taken to make predictions
    end_time = time.time()
    time_taken = end_time - start_time

    a = result.pandas().xyxy[0].sort_values('xmin')
    while len(a) > 5: #TODO: custom value for longer/shorter captchas
      lines = a.confidence
      linev = min(a.confidence)
      for line in lines.keys():
        if lines[line] == linev:
          a = a.drop(line)

    result = ""
    for _, key in a.name.items():
      result = result + key
    
    print(Fore.CYAN + f"[i | Solve] Captcha: {result}")
    print(Fore.MAGENTA + f"Time taken to solve captcha: {time_taken:.4f} seconds")

    return result
  except:
    print(Fore.RED + 'Failed to solve a captcha!')
    return None