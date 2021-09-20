#


#import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
#        0                                                                                                                                        16

odg_1 = [0.766, 0.767, 0.764, 0.770, 0.767]
odg_2 = [0.808, 0.810, 0.808, 0.807, 0.808]
odg_3 = [0.813, 0.818, 0.818, 0.819, 0.817]

odg_1 = [0.770, 0.767, 0.769, 0.762, 0.766]
odg_2 = [0.807, 0.810, 0.811, 0.804, 0.808]
odg_3 = [0.803, 0.805, 0.803, 0.804, 0.804]


mix = ['B0', 'B1', 'MT']

        
#print(mix)
plt.figure()

plt.plot(mix, [odg_1, odg_2, odg_3], label='', marker = "o", ls = '')
#plt.plot(mix, odg_2, label='ratio + wiener, iter = 2', marker = "o")
#plt.plot(mix, odg_3, label='binary + wiener', marker = "o")
plt.title('MACRO F1 score comparison')
plt.xlabel('')
plt.ylabel('F1 score')
plt.ylim([0.76, 0.83])
#plt.setp(lines,marker = "o") 
plt.grid(True)
plt.legend()
plt.savefig('F1_ind_cmp.png')
plt.show()



