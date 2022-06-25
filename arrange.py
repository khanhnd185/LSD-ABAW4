import os
import shutil

DATA_DIR = '../../../Data/ABAW4/synthetic_challenge/'
classes = ['ANGRER', 'DISGUST', 'FEAR', 'HAPPINESS', 'SADNESS', 'SURPRISE']

for dir in ['validation']:
    OUTPUT_DIR = DATA_DIR + dir + '/'
    INPUT_DIR = DATA_DIR + dir + '_set_real_images/'
    os.makedirs(OUTPUT_DIR, exist_ok = True)
    with open("{}/{}.txt".format(DATA_DIR, dir), "w") as file:
        for i, c in enumerate(classes):
            path = INPUT_DIR + c
            files = os.listdir(path)
            for f in files:
                newf = str(i) + f
                file.write("{},{}\n".format(newf, i))
                src = path + '/' + f
                dst = OUTPUT_DIR + newf
                shutil.copyfile(src, dst)
