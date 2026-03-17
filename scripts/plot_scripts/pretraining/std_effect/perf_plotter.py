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
    input_file = "./ds_effect.xls"
    output_file = "./data_size_perf_plot.pdf"

    # load the dataset
    ds = pd.read_excel(input_file, sheet_name='pre-training-data-size-effect')

    # remove the diverged trainings with high perplexity (outliers)
    ds = ds[ds['perplexity'] < 10.0]

    # set the font size
    plt.rcParams["font.size"] = 9
    plt.rcParams["axes.titlesize"] = 9
    plt.rcParams["axes.labelsize"] = 9
    
    # set the legend font size
    plt.rcParams["legend.fontsize"] = 8

    # set the resolution
    plt.rcParams["figure.dpi"] = 150

    # grid of line plots with error bars for accuracy, weighted-f1 and perplexity
    properties = ["accuracy", "w-f1", "perplexity"]

    # set the color palette to blue, red and green
    sns.set_palette(["blue", "red", "green"])
    
    # set the circle, triangle, square markers for each Name
    markers = {"tiny": "o", "small": "s", "base": "^"}

    # create fig and axes objects
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(6.5, 3.3))

    # create a list of y axis labels
    labels = ["Accuracy", "Weighted-F1", "Pseudo-Perplexity"]

    # create the plots
    for i, prop in enumerate(properties):
        ax = axes[i]
        sns.lineplot(
            data=ds,
            x="bin_idx",
            y=prop,
            hue="Name",
            style="Name",
            err_style="band",
            markers=markers,
            dashes=False,
            errorbar=("ci", 95),
            ax=ax,
        )
        ax.set_ylabel(labels[i])
        ax.set_xlabel("$k$")
        ax.legend_.remove()    
        # increase the marker size for triangle
        # and remove the white edge around markers
        for line in ax.get_lines():
            if line.get_marker() == "^":
                line.set_markersize(6)
            else:
                line.set_markersize(5)
            line.set_markeredgecolor("none")
        ax.set_xticks(ds["bin_idx"].unique())

    # add legend only to the first plot
    axes[0].legend()
    fig.tight_layout()

    # save the figures
    fig.savefig(output_file, bbox_inches="tight", dpi=150)

if __name__ == "__main__":
    main()
