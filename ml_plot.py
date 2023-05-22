import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('metrics.csv')

plt.bar(df['build'], df['accuracy'])

plt.xlabel('Build')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Build')

plt.savefig('plot.png')
