import matplotlib.pyplot as plt

def initialize_bar_plot():
    fig, ax = plt.subplots(1)

    # Get rid of grids and ticks
    ax.grid(b=False)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    # Remove the two sides of the bounding box
    spines_to_remove = ['top', 'right']
    for spine in spines_to_remove:
        ax.spines[spine].set_visible(False)

    return fig, ax
