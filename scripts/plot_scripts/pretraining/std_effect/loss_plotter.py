###############################################################################
# Author: Mohammad Mostafanejad                                               #
# Date: August 2025                                                           #
# Description:                                                                #
# This module will use the 'ds_effect.xls' raw data file to plot the          #
# dataset size effect on model performance.                                   #
###############################################################################

# import the necessary modules
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def main():
    # input and output file paths
    input_file = "./std_effect.xls"
    output_file = "./std_loss_plot.pdf"

    # load the dataset
    ds = pd.read_excel(input_file)

    # remove the diverged trainings with high perplexity (outliers)
    ds = ds[ds['perplexity'] < 10.0]

    # set the cosmetic plot parameters
    plt.rcParams["font.size"] = 9
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["figure.dpi"] = 150
    
    # set the legend font size
    plt.rcParams["legend.fontsize"] = 8

    # set the color palette to blue, red and green
    sns.set_palette(["darkblue", "red", "green"])

    # set the marker types
    markers = {"tiny": "o", "small": "s", "base": "^"}

    # create a dictionary for dashed line types
    dashes = {"tiny": (2, 2), "small": (3, 2), "base": (4, 2)}

    # plot t-loss and v-loss for tiny, small and base on the same plot
    fig, ax = plt.subplots(figsize=(3.6, 3.6))

    # plot t-losses
    # use solid lines for v-loss and dashed lines for t-loss
    sns.lineplot(
        data=ds,
        x="bin_idx",
        y="t-loss",
        hue="Name",
        style="Name",
        markers=markers,
        dashes=dashes,
        ax=ax,
    )

    # create a dictionary of legends.
    legends = {
        "tiny (t-loss)": ax.get_lines()[0],
        "small (t-loss)": ax.get_lines()[1],
        "base (t-loss)": ax.get_lines()[2],
    }

    # plot v-losses
    sns.lineplot(
        data=ds,
        x="bin_idx",
        y="v-loss",
        hue="Name",
        style="Name",
        markers=markers,
        dashes=False,
        ax=ax,
    )

    # add three more legends
    legends.update({
        "tiny (v-loss)": ax.get_lines()[6],
        "small (v-loss)": ax.get_lines()[7],
        "base (v-loss)": ax.get_lines()[8],
    })

    # add legends to the plot
    ax.legend(handles=legends.values(), labels=legends.keys())

    # change the y axis label to "loss"
    ax.set_ylabel("loss")

    # change the x axis label to $k$
    ax.set_xlabel("$k$")

    # increase the marker size for triangle
    for line in ax.get_lines():
        if line.get_marker() == "^":
            line.set_markersize(6)
        else:
            line.set_markersize(5)
        line.set_markeredgecolor("none")

    # adjust the layout
    fig.tight_layout()

    # save the figure
    fig.savefig(output_file, bbox_inches="tight", dpi=150)

if __name__ == "__main__":
    main()
