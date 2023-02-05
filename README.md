#**DEPRECATED**: stable-diffusion-webui-instruct-pix2pix

NOTE: This extension is no longer required. I have integrated the code into Automatic1111 img2img pipeline and the webUI now has Image CFG Scale for instruct-pix2pix models.

If there is anything from this extension you'd like please let me know (i.e. random CFG) and I can make a separate extension to extend webui img2img instead.

Enjoy!


Extension for webui to run instruct-pix2pix models

Code based on: https://www.timothybrooks.com/instruct-pix2pix/
(see LICENSE for more info)

Examples from original creator:
![image](https://user-images.githubusercontent.com/26013475/214625822-2e60f5b1-fdc9-44ca-996d-6e7cddab8d67.png)

Download the instruct-pix2pix-00-22000 model manually from here: https://huggingface.co/timbrooks/instruct-pix2pix/tree/main

After loading the model you can use the extension.

![image](https://user-images.githubusercontent.com/26013475/215627091-f8ee97f4-0e95-4845-8086-e77c413e0379.png)
 
Batch Input processing will unpack any animated .gif found in the directory and process all frames by default.
 
Model does not work with img2img or txt2img. Merging with other 1.5 models is possible but results are mixed at best.

Embeddings for 1.5 work, as does emphasis. Advanced prompting features from webui are not implemented.

Todo: Integrate with img2img instead of an extension :)

Feel free to use my modified Image Browser extension for native browsing support (and enhanced sort and filter features compared to normal repo):

https://github.com/Klace/stable-diffusion-webui-images-browser

![image](https://user-images.githubusercontent.com/26013475/214626966-50897959-7c7e-4a49-b92c-6609d7af1735.png)
