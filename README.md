# a trash attempt at rendering a 3d scene from videos

well wat are the chances tht someones gonna read this. might as well have fun while writing this haha. anyway its like saturday after week 2 of spring quarter 2020 and its stupid quarantining season and i have nothing to do other than schoolwork, so this is what ive been doing in quarantine. i miss the vibe and everybody at school and its so depressing to be at home 24/7. was also looking forward to interning in seattle this summer but im gonna be home :'(

so i guess this is supposed to be a summary or story. well if y'all ~recruiters~ people wanna know, i had a research lab interview at my school related to autonomous vehicles, and reading the description of what i needed to know, i saw SLAM. naturally i googled it and i came across it, and i googled a bunch of other stuff to make this project as well lmao.

## usage cuz i cant expect u to know

i use pipfiles because i really like using pipenv, but for all u guys and gals that dont, i have a requirements.txt in here for u. 
o btw heres the usage lmao:

```
python slam.py test.mp4
```

## some stuff you need
* python3 (maybe python2 works???) --> i did this in python3 and my pipfile says so, but test it out on python2 if you fancy that
* [pangolin](https://github.com/uoip/pangolin) --> this gave me pain
* opencv --> to analyze photos (god bless goodfeaturestotrack lmao)
* pysdl2 --> dont u wanna see the cool little feature dots on ur video

oh ye please note compile pangolin on linux, such as the chad ubuntu. i rlly like macOS or OS X, whatever the proper name is, but U CANT COMPILE PANGOLIN FOR ITS FREAKING LIFE. this problem caused me like 6 hours of my time so ima save all you boys and girls out there some time <3

## pictures cuz they cool
![pic1](https://github.com/thenry3/3D-Mapping-from-Video/raw/master/screenshots/screenshot.png)
![pic2](https://github.com/thenry3/3D-Mapping-from-Video/raw/master/screenshots/screenshot1.png)


