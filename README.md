# stable-diffusion-webui-instruct-pix2pix
Extension for webui to run instruct-pix2pix models

Code Based on: https://www.timothybrooks.com/instruct-pix2pix/
(see LICENSE for more info)

![image](https://user-images.githubusercontent.com/26013475/214625822-2e60f5b1-fdc9-44ca-996d-6e7cddab8d67.png)

Download the instruct-pix2pix-00-22000 model manually from here: https://huggingface.co/timbrooks/instruct-pix2pix/tree/main

After loading the model you can use the extension.

![image](https://user-images.githubusercontent.com/26013475/215025842-f72dda01-0a33-4d32-b745-de9d9503ade2.png)

Model does not work with img2img or txt2img. Merging is possible but requires modification to extras.py (out of scope for now)

Embeddings for 1.5 work, as does emphasis.

ToDo: Button to send output back to input, scripts, proper batching, etc...

Feel free to use my modified Image Browser extension for native browsing support (and enhanced sort and filter features compared to normal repo):

https://github.com/Klace/stable-diffusion-webui-images-browser

![image](https://user-images.githubusercontent.com/26013475/214626966-50897959-7c7e-4a49-b92c-6609d7af1735.png)
