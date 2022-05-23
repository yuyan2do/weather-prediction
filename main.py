import numpy as np
from imageio import imread

def image_read(fname, factor = 35):
    # radar factor = 70
    # wind factor = 35
    # precip factor = 10
    img = np.array(imread(fname)/255*factor)
    #img = np.array(imread(fname))
    print(f"max value: {np.max(img)}")
    return img


if __name__ == "__main__":
    img = image_read("/mnt/yyua/data/qixiang/TestA/Wind/wind_31218.png")
    print(img)
