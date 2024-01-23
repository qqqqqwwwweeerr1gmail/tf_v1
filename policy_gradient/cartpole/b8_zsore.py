from scipy.stats import zscore

discounted_acc_rewards = [5,3,1,-1,-3,-6]
print(discounted_acc_rewards)
z_d_rewards = zscore(discounted_acc_rewards)
print(z_d_rewards)



discounted_acc_rewards = [5,3,1,-1,-3,-6,-100]
print(discounted_acc_rewards)
z_d_rewards = zscore(discounted_acc_rewards)
print(z_d_rewards)















