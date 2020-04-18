import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter


def plot_reward(a, b, name_a, name_b, hyperparam_dict, num=1):
    plt.rcParams.update({'font.size': 18})
    # Plot the running average results only for the different settings
    fig = plt.figure(num, figsize=(20,8))
    plt.plot(range(len(a)), a, color='grey', lw=2, label=name_a)
    
    if b != None:
        if 'epsilon' in hyperparam_dict.keys():
            plt.plot(range(len(b)), b, color='black', lw=2, label=name_b +', eps:'+str(hyperparam_dict['epsilon']))
        else:
            plt.plot(range(len(b)), b, color='black', lw=2, label=name_b)
        plt.grid()
        plt.xlabel('Episodes')
        plt.ylabel('Running average of Rewards')
        plt.legend(ncol=2)
        #plt.show()
    else:
        plt.grid()
        plt.xlabel('Episodes')
        plt.ylabel('Epsilon Thresold')
        plt.legend(ncol=2)
        #plt.show()
    
    fig.savefig(hyperparam_dict['name']+'/'+name_a+'_fig'+str(num)+'.png')

    
def multi_plot(lists_of_tuples, hyperparam_dict, num, title): 
    # Dangerous handling of the num figure
    #num = np.random.randint(0, 1000)
    plt.rcParams.update({'font.size': 18})
    fig, ax1 = plt.subplots(figsize=(20,8))
    # Colors plotting managing  
    #colors = [cmap(i) for i in np.linspace(0, 1, len(lists_of_tuples))]
    cmap = plt.get_cmap('gnuplot')
    colors = ['blue','black', 'orange', 'green', 'red']
    counter = 0
    for count, i in enumerate(lists_of_tuples):
        if i[3] == 'dash':
            dot = [1, 4, 1, 4, 1, 4]
        if i[3] != 'dash':
            dot=[]
        if i[2] == 0:
            ax1.plot(range(len(i[0])), i[0], lw=2, dashes = dot, label=i[1], color= colors[count])
        if i[2] == 1:
            if counter == 0:
                counter +=1
                ax2 = ax1.twinx()
                ax2.set_ylabel(i[1])  # we already handled the x-label with ax1
                ax2.plot(range(len(i[0])), i[0], lw=2, dashes = dot, label=i[1], color= 'orange')
                lines2, labels2 = ax2.get_legend_handles_labels()
            else:
                ax2.plot(range(len(i[0])), i[0], lw=2, dashes = dot, label=i[1], color= colors[count])
                lines2, labels2 = ax2.get_legend_handles_labels()
        if i[2]== 2:
            ax1.plot(np.linspace(0, hyperparam_dict['max_episodes'], len(i[0])), 
                     i[0], lw = 1, dashes = dot, label=i[1], color = colors[count])
            ax1.fill_between(np.linspace(0, hyperparam_dict['max_episodes'], len(i[0])), 
                             (np.array(i[0])-np.array(i[4])), 
                             (np.array(i[0]) + np.array(i[4])), 
                             color=colors[count], alpha = 0.25)
        if i[2]==4:
            ax1.plot(range(len(i[0])), i[0], lw=2, dashes = dot, label=i[1], color='black')
            ax1.fill_between(np.linspace(0, hyperparam_dict['max_episodes'], len(i[0])), 
                             (np.array(i[4])),
                             np.array(i[0]),
                             color='black', alpha = 0.25)
            
    
    lines, labels = ax1.get_legend_handles_labels()
    # One legend for both the plots
    # Putting the legend below, out of the plot, at centre
    #ax2.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                #fancybox=True, shadow=True, ncol=len(lists_of_tuples))
    if counter != 0:
        ax2.legend(lines + lines2, labels +labels2, loc=4, fancybox=True, shadow=False)
    if counter == 0:
        ax1.legend(lines, labels, loc=4, fancybox=True, shadow=False)

    ax1.grid()
    fig.suptitle(title, fontsize=20)
    ax1.set_ylabel('Rewards ')
    ax1.set_xlabel('Episodes')
    #plt.show()
    # Saving the png file
    fig.savefig(hyperparam_dict['name']+'/multiplot_fig'+str(num)+'.png')

def plot_mean_std(a, name_a, hyperparam_dict, type_im, num=1):
    plt.rcParams.update({'font.size': 18})
    # Plot the running average results only for the different settings
    fig = plt.figure(num, figsize=(20,8))
   
    if 'epsilon' in hyperparam_dict.keys():
        plt.plot(np.linspace(0, hyperparam_dict['max_episodes'], len(a)), a, color='blue', lw=2, label=name_a+',eps:'+str(hyperparam_dict['epsilon']))
    else:
        plt.plot(np.linspace(0, hyperparam_dict['max_episodes'], len(a)), a, color='blue', lw=2, label=name_a)
    
    plt.grid()
    plt.xlabel('Episodes')
    plt.ylabel('Value')
    plt.legend(ncol=2)
    #plt.show()
    fig.savefig(hyperparam_dict['name']+'/'+hyperparam_dict['name']+'_'+type_im+'_fig'+str(num)+'.png')

def subplot_three_one(a, b, c, d, name_a, name_b, name_c, name_d, max_episodes, filename):
    
    plt.rcParams.update({'font.size': 18})
    fig = plt.figure(1, figsize=(20,30))

    plt.subplot(3, 1, 1)
    
    plt.plot(range(len(a)), a, color='blue', lw=2, label=name_a)
    plt.plot(range(len(b)), b, color='black', lw=2, label=name_b)
    plt.grid()
    plt.xlabel('Episodes')
    plt.ylabel('Running average of Rewards')
    plt.legend(ncol=2)

    plt.subplot(3, 1, 2)

    plt.plot(np.linspace(0, max_episodes, len(c)), c, color='orange', lw=2, label=name_a)
    plt.grid()
    plt.xlabel('Episodes')
    plt.ylabel('Value')
    plt.legend(ncol=2)

    plt.subplot(3, 1, 3)

    plt.plot(np.linspace(0, max_episodes, len(d)), d, color='green', lw=2, label=name_a)
    plt.grid()
    plt.xlabel('Episodes')
    plt.ylabel('Value')
    plt.legend(ncol=2)
    
    fig.savefig(filename+'/'+filename+'__fig.png')

