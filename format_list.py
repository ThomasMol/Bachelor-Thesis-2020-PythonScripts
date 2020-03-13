import os
from natsort import natsorted, realsorted,  ns

asciiDict = {chr(i): i for i in range(129)}

with open("output.txt", "w") as a:
    for path, subdirs, files in os.walk(r'C:\Users\email\Documents\Universiteit\Informatiekunde 2019-2020\3. OZP\ChaLearn_DataSet\train-1\training80_12'):
       for filename in sorted(files, key=lambda d: d.lower().replace("_", ":")):
         f = os.path.join(path, filename)
         a.write(str(filename) + '\n') 
         

#sorted(files, key=lambda d: d.lower().replace("_", " "))
#natsorted(files, alg = ns.IGNORECASE)
#print(asciiDict)