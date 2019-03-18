from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt


# activate latex text rendering


env = 'Mountain'

def get_best_index(name,size):
	values = []
	for i in range(1,size+1):
		name_this = name + '_'+str(i)
		max_perform = genfromtxt(name_this+'.csv',delimiter=',')[:,2]
		values.append(np.max(max_perform))
	return (np.argsort(values)+1)#[::-1]


def plot_data(name,comment,color,ax,size,best=True):
	this_name = name + '_1'
	data = genfromtxt(this_name+'.csv',delimiter=',')
	time = data[:,0]

	if best:
		indecies = get_best_index(name,size)[:3]
	else:
		indecies = np.linspace(1,size,size,dtype=int)

	train_mean = np.zeros((len(time),len(indecies)))
	eval_mean = np.zeros((len(time),len(indecies)))
	print(indecies)
	index=0
	for i in indecies:
		print(i)
		name_this = name + '_'+str(i)
		data = genfromtxt(name_this+'.csv',delimiter=',')
		train_perform = data[:,1]
		max_perform = data[:,2]
		print(np.shape(train_perform))
		train_mean[:,index]=train_perform
		eval_mean[:,index]=max_perform
		index += 1
	mean_arr = np.mean(train_mean,axis=1)
	mean_max_arr = np.mean(eval_mean,axis=1)
	std = np.std(train_mean,axis=1)
	std_max = np.std(eval_mean,axis=1)

	upper = mean_arr + std
	lower = mean_arr - std

	upper_m = mean_max_arr+std_max
	lower_m = mean_max_arr-std_max

	#plt.plot(avg_data[:min_size,0],lower[:],label=comment,color=color)
	#plt.plot(avg_data[:min_size,0],upper[:],label=comment,color=color)

	ax.fill_between(time,upper,lower,alpha=.2,facecolor=color)
	ax.plot(time,mean_arr,label=comment+' training',color=color)

	ax.fill_between(time,upper_m,lower_m,alpha=.2,facecolor=color)
	ax.plot(time,mean_max_arr,label=comment+' evaluate',color=color,linestyle='--')



	#process_adv(name,color,1)

def process_adv(name,color,size):
	name_this = name + '_1'
	time = genfromtxt(name_this+'.csv',delimiter=',')[1:,0]
	for i in range(1,size+1):
		name_this = name + '_'+str(i)
		adv = genfromtxt(name_this+'v1.csv',delimiter=',')[1:]
		adv_mean = np.mean(adv,axis=1)
		adv_std = np.sqrt(np.var(adv,axis=1))

	upper = adv_mean+adv_std
	lower = adv_mean-adv_std

	plt.plot(time[:],adv_mean[:],color=color,linestyle='--')
	plt.fill_between(time[:],upper[:],lower[:],alpha=.2,facecolor=color)

def plot_detection_rate(name,ax,start,stop):
	data = genfromtxt(name+'.csv',delimiter=',')
	time = data[:,0]
	pol_1 = data[:,5]/1024
	pol_2 = data[:,6]/1024
	pol_r1= data[:,4]/1024
	pol_r2= 1-pol_r1
	ax.plot(time[start:stop],pol_1[start:stop],color='c',label='fraction nominal policy selected')
	ax.plot(time[start:stop],pol_2[start:stop],color='orange',label='fraction adversary policy selected')
	ax.plot(time[start:stop],pol_r2[start:stop],color='c',label='ground truth fraction nominal policy',linestyle='--',alpha=0.8)
	#ax.plot(time[start:stop],pol_r2[start:stop],color='orange',label='frequency pol 2 ground truth of 1024',linestyle='--',alpha=0.5)

#plot_data('./data/InvertedPendulum-v2_base_augment_False_batchsize_2048_100_200','base',color='g',size=5,best=True)
#plot_data('./data/InvertedPendulum-v2_bias_attack_augment_True_batchsize_2048_10_20','base adv',color='r',size=5,best=True)
#plot_data('./data/InvertedPendulum-v2_bias_attack_augment_True_batchsize_2048_1_1','base',color='g',size=5,best=False)


#plot_data('./data/InvertedPendulum-v2_bias_attack_augment_False_batchsize_2048_10_20','base',color='g',size=5,best=False)
#plot_data('./data/InvertedPendulum-v2_bias_attack_augment_False_batchsize_2048_10_20','base adv',color='r',size=5,best=False)
fig, ax1 = plt.subplots(1,1)

# plot_data('./data/GridWorld-v0_meta_augment_True_batchsize_1024','MLAH',color='c',ax=ax1,size=5,best=False)
# plot_data('./data/GridWorld-v0_nominal_augment_True_batchsize_1024','Vanilla PPO',color='r',ax=ax1,size=5,best=False)
# plot_detection_rate('./data/GridWorld-v0_meta_long_augment_True_batchsize_1024_1',ax2,0,300)

plot_data('./data/CartPole-v1_grad_oracle_no_delay_augment_True_batchsize_1024','Oracle MLAH',color='c',ax=ax1,size=5,best=False)
plot_data('./data/CartPole-v1_grad_baseline_no_delay_augment_True_batchsize_1024','Vanilla PPO',color='r',ax=ax1,size=5,best=False)
ax1.set_xlabel('iterations')
# ax2.set_xlabel('iterations')
# ax1.set_ylabel('average epsisode reward')
# ax2.set_ylabel('average epsisode reward')
# ax1.set_ylim(-100,140)
# ax1.set_ylim(-100,140)
#ax1.set_xlim(0,150)
#ax2.set_xlim(33,140)

ax1.legend(loc='best')
#ax2.legend(loc='lower right')
ax1.grid()
#ax2.grid()

ax1.set_xlabel('iterations')
#ax2.set_xlabel('iterations')
# ax[1].set_xlabel('iterations')
# ax[0].set_title('a')
# ax[1].set_title('b')
# ax[0].set_xlim([0,100])
# ax[1].set_xlim([0,100])
# ax[0].set_ylim([0,100])
# ax[1].set_ylim([0,600])
# plt.ylabel('meta policy selection frequency')
ax1.set_ylabel('average episode reward')
#ax2.set_ylabel('fraction of conditoin')
# ax1.set_xlim(0,80)
# ax2.set_xlim(340,420)
# ax1.set_ylim(-20/1024,(1024+20)/1024)
# ax2.set_ylim(-20/1024,(1024+20)/1024)
# scale = 1.0
# ax1.spines['right'].set_visible(False)
# ax2.spines['left'].set_visible(False)
# ax1.tick_params(right='off')
# ax2.tick_params(left='off')
# ax1.tick_params(labelright='off')
# ax2.tick_params(labelleft='off') # don't put tick labels at the top


# d = 1.5
# y_scale = 1000/80  # how big to make the diagonal lines in axes coordinates
# # arguments to pass to plot, just so we don't keep repeating them
# kwargs = dict(color='k', clip_on=False)
# ax1.plot((-d+80, d+80), ((-d*y_scale-20)/1024, (d*y_scale-20)/1024), **kwargs)        # top-left diagonal
# ax1.plot((-d+80, d+80), ((1024-d*y_scale+20)/1024, (1024+d*y_scale+20)/1024), **kwargs)  # top-right diagonal
# ax2.legend(loc='center')
# #ax1.legend(loc=2)
#  # switch to the bottom axess
# ax2.plot((-d*scale+340, d*scale+340), ((-d*y_scale-20)/1024, (d*y_scale-20)/1024), **kwargs)  # bottom-left diagonal
# ax2.plot((-d*scale+340, d*scale+340), ((1024-d*y_scale+20)/1024, (1024+d*y_scale+20)/1024), **kwargs)
# plt.subplots_adjust(wspace=0.1)
# ax1.set_ylabel('fraction of all selections')
# ax1.set_xlabel('iteration')
# ax2.set_xlabel('iteration')
plt.show()
