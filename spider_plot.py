from math import pi
import matplotlib.pyplot as plt

 
def radar_chart(cl_pos):
    # Set data
    plt.figure(3)
    radar_df = cl_pos.reset_index().rename(columns={'index':'cluster'})
     
    # ------- PART 1: Create background
     
    # number of variable
    categories= radar_df.columns.tolist()[1:]
    # list(df)[1:]
    N = len(categories)
     
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
     
    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)
     
    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
     
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories)
     
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks(ticks=None, labels=None)
    # plt.ylim(0,40)
     
    
    # ------- PART 2: Add plots
     
    # Plot each individual = each line of the data
    # I don't make a loop, because plotting more than 3 groups makes the chart unreadable
    
    # Ind1
    for i in range(len(radar_df.iloc[:,0])):
        values=radar_df.loc[i].drop('cluster').values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle=None, label=f"Cluster {i}")
        ax.fill(angles, values, alpha=0.1)
    
    # # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Show the graph
    plt.show()