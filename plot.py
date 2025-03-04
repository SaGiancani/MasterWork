import matplotlib
matplotlib.use('template')
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

    
def multi_plot(lists_of_tuples, hyperparam_dict, num, title, locality = 1, x='Episodes', y='Rewards'): 
    #locality = 1 upper right
    #locality = 4 lower right
    #locality = 0 best
    # Dangerous handling of the num figure
    #num = np.random.randint(0, 1000)
    plt.rcParams.update({'font.size': 18})
    fig, ax1 = plt.subplots(figsize=(20,8))
    # Colors plotting managing  
    #colors = [cmap(i) for i in np.linspace(0, 1, len(lists_of_tuples))]
    cmap = plt.get_cmap('gnuplot')
    colors = ['orange', 'blue','black', 'green', 'red']
    counter = 0
    for count, i in enumerate(lists_of_tuples):
        # The third position of the tuple is dedicated to the forma line of plotting: full line or dashed line
        if i[3] == 'dash':
            dot = [1, 4, 1, 4, 1, 4]
        if i[3] != 'dash':
            dot=[]
        # Simple plot
        if i[2] == 0:
            ax1.plot(range(len(i[0])), i[0], lw=2, dashes = dot, label=i[1], color= colors[count])
        # Plot with 2 y-axis and 1 x-axis
        if i[2] == 1:
            # If the number 1 is the second element of the i-th tuple, a new y axis is created, on the same 
            # figure: it shares the same x axis.
            # The check on the counter is due to the creation of the new y-axis
            if counter == 0:
                counter +=1
                ax2 = ax1.twinx()
                ax2.set_ylabel(i[1])  # we already handled the x-label with ax1
                ax2.plot(range(len(i[0])), i[0], lw=2, dashes = dot, label=i[1], color= 'orange')
                lines2, labels2 = ax2.get_legend_handles_labels()
            else:
                ax2.plot(range(len(i[0])), i[0], lw=2, dashes = dot, label=i[1], color= colors[count])
                lines2, labels2 = ax2.get_legend_handles_labels()
        # Plot with filling area between two different functions
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
    #loc = 1 upper right
    #loc = 4 lower right
    #loc = 0 best
    if counter != 0:
        ax2.legend(lines + lines2, labels +labels2, loc=locality, fancybox=True, shadow=False)
    if counter == 0:
        ax1.legend(lines, labels, loc=locality, fancybox=True, shadow=False)

    ax1.grid()
    fig.suptitle(title, fontsize=20)
    ax1.set_ylabel(y)
    ax1.set_xlabel(x)
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
    
def plot_imgs(lists_of_tuples, n_elements, title, size, filename, num):
    # Show n_elements images from the selected batch (original, reconstructed from 
    #the pure Convolutional autoencoder, reconstructed from the autoencoder)
    if (len(lists_of_tuples)==0):
        print('Warning: The list of tuples is empty!')
        return
    fig, axes = plt.subplots(nrows=len(lists_of_tuples), ncols=n_elements,figsize=(16,8))
    
    # input images on top row, reconstructions on bottom
    if (len(lists_of_tuples)>1):
        for images, row in zip(lists_of_tuples, axes):
            i = 0
            for img, ax in zip(images[0], row):
                ax.imshow(img.reshape(size,size), cmap='gray')
                ax.get_xaxis().set_visible(False)
                if i==0:
                    ax.set_ylabel(images[1], fontsize=16)
                else:
                    ax.get_yaxis().set_visible(False)
                i+=1
                
    if (len(lists_of_tuples)==1):
        for ax, img in zip(axes, lists_of_tuples[0][0]):
            ax.imshow(img.reshape(size,size), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)    

    fig.suptitle(title, fontsize=16)
    #pfig.tight_layout()
    fig.savefig(filename+'/'+filename+'imgs_'+str(num)+'_fig.png')


