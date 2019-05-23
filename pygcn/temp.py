import pickle 
import matplotlib.pyplot as plt
import numpy as np



# Example data
people = ('TP', 'FP', 'FN', 'TN')
y_pos = np.arange(len(people))
performance = (64354, 2701, 17007, 2417826)


plt.barh(y_pos, performance, align='center',
        color='green', ecolor='black')
plt.set_yticks(y_pos)
plt.set_yticklabels(people)
plt.invert_yaxis()  # labels read top-to-bottom
plt.set_xlabel('Performance')
plt.set_title('Precission - 0.959 Recal - 0.790')

plt.show()
