import matplotlib.pyplot as plt

# window = [4, 5, 6, 7, 8]
# accw = [73.0, 73.5, 73.6, 73.9, 73.5]
plt.figure(figsize=(25, 15), dpi=200)
window = [4,5,6,7,8]
accw = [73.0, 73.5, 73.6, 73.9, 73.5]
####################
plt.plot(window, accw, c='red', linestyle='-.', label='EWS-VViT')
plt.scatter(window, accw, c='red')
plt.legend(loc='best')
########################
plt.yticks(range(72, 76, 1))
plt.xticks(range(3, 10, 1))
plt.grid(True, linestyle='--', alpha=1)
########################
plt.xlabel("(a) window size", fontdict={'size': 16})
plt.ylabel("Overal accuray(%)", fontdict={'size': 16})
#####################
plt.show()

