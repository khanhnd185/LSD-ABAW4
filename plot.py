# importing package
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Plot learning curve')
parser.add_argument('--file', '-f', type=str, help='cfile to plot')
parser.add_argument('--factor', default='all', type=str, help="choose factor to plot")
args = parser.parse_args()

df = pd.read_csv(args.file)
i = np.arange(20)

print(df)  
for k in df.keys()[2:]:    
    if (args.factor in k) or (args.factor == 'all'):
        plt.plot(i, df[k], label=k)

plt.legend()
plt.show()