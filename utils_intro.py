import os
import gzip
import numpy as np
import pandas as pd

def load_citycam(domains, path="./data/citycam"):
    indexes = []
    for domain in domains:
        path_dom = os.path.join(path, domain)
        for r, d, f in os.walk(path_dom):
            if domain[:6] == "bigbus":
                d = [""]
            for direct in d:
                if "checkpoints" not in direct:
                    x_path = os.path.join(path_dom, direct, "X.npy")
                    xt_path = os.path.join(path_dom, direct, "Xt.npy")
                    y_path = os.path.join(path_dom, direct, "y.npy")
                    time_path = os.path.join(path_dom, direct, "time.npy")

                    Xti = np.load(xt_path)
                    try:
                        Xi = np.load(x_path)
                    except:
                        Xi = np.load(xt_path)
                    yi = np.load(y_path)
                    time_i = np.load(time_path)
                    
                    if len(yi) != len(Xi):
                        print(len(yi), len(Xi))
                        
                    indexes += [domain] * len(yi)

                    try:
                        X = np.concatenate((X, Xi))
                        Xt = np.concatenate((Xt, Xti))
                        y = np.concatenate((y, yi))
                        time = np.concatenate((time, time_i))
                    except:
                        X = np.copy(Xi)
                        Xt = np.copy(Xti)
                        y = np.copy(yi)
                        time = np.copy(time_i)
    return Xt, X, y, time, np.array(indexes)
    
    
def get_day(x):
    return x.split(" ")[0]
    

def citycam_weather_v2(state=None, return_indexes=False, path="./data/citycam"):
    cameras = ["164"]
    Xt, X, y, time, indexes = load_citycam(cameras, path=path)
    day = np.array(list(map(get_day, time)))
    
    Xt = Xt[day == "2016/02/23"]
    X = X[day == "2016/02/23"]
    y = y[day == "2016/02/23"]
    time = time[day == "2016/02/23"]
    indexes = indexes[day == "2016/02/23"]

    out = dict(
        Xt = Xt,
        X = X,
        y = y,
        time = time,
        indexes = indexes,
    )
    return out