# a bar plot with errorbars
import numpy as np
import matplotlib.pyplot as plt
N = 5
menMeans = (20, 35, 30, 35, 27)
womenMeans = (25, 32, 34, 20, 25)
menStd =   (2, 3, 4, 1, 2)
womenStd =   (3, 5, 2, 3, 3)
ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(ind, menMeans, width, color = 'r', yerr=menStd)
rects2 = ax.bar(ind+width, womenMeans, width, color='y', yerr=womenStd)
ax.legend( (rects1[0], rects2[0]), ('Men', 'Women') )

# add some text for labels, title and axes ticks
ax.set_ylabel('Family Votes')
ax.set_title('Votes by group and gender')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('G1', 'G2', 'G3', 'G4', 'G5') )

# attach some text labels
def autolabel(rects):
    for rect in rects:
         height = rect.get_height()
         ax.text(rect.get_x()+rect.get_width()/2., height+1, 
              int(height), ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.show()
