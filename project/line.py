import matplotlib.pyplot as plt

# window = [4, 5, 6, 7, 8]
# accw = [73.0, 73.5, 73.6, 73.9, 73.5]
plt.figure(figsize=(25, 15), dpi=200)
frame = [4, 8, 12, 16]
accf = [72.8, 73.9, 73.4, 73.1]
####################
plt.plot(frame, accf, c='red', linestyle='--', label='EWS-VViT')
plt.scatter(frame, accf, c='red')
plt.legend(loc='best')
########################
plt.yticks(range(72, 76, 1))
plt.xticks(range(0, 24, 4))
plt.grid(True, linestyle='--', alpha=1)
########################
plt.xlabel("(b) Sampling frame number", fontdict={'size': 16})
plt.ylabel("Overal accuray(%)", fontdict={'size': 16})
#####################
plt.show()

