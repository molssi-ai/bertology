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
    ds = pd.read_excel(input_file, sheet_name="pre-training-data-size-effect")

    # remove the diverged trainings with high perplexity (outliers)
    ds = ds[ds["perplexity"] < 10.0]

    # set the font size
    plt.rcParams["font.size"] = 9
    plt.rcParams["axes.titlesize"] = 9
    plt.rcParams["axes.labelsize"] = 9

    # set the legend font size
    plt.rcParams["legend.fontsize"] = 8

    # set the resolution
    plt.rcParams["figure.dpi"] = 150

    # grid of line plots with error bars for accuracy, weighted-f1 and perplexity
    properties = ["v-loss", "accuracy", "w-f1", "perplexity"]

    # set the color palette to blue, red and green
    sns.set_palette(["blue", "red", "green"])

    # set the circle, triangle, square markers for each Name
    markers = {"tiny": "o", "small": "s", "base": "^"}

    # create fig and axes objects
    fig, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(6, 5), dpi=150, sharex=True, sharey=False
    )

    # create a list of y axis labels
    labels = ["V-Loss", "V-Acc", "V-wF1", "V-PPPL"]

    # plot titles
    plt_titles = ["(a)", "(b)", "(c)", "(d)"]

    # create the plots
    for i, prop in enumerate(properties):
        ax = axes[i // 2, i % 2]
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
        ax.set_title(plt_titles[i], loc="left", pad=10, fontweight="bold")
        # increase the marker size for triangle
        # and remove the white edge around markers
        for line in ax.get_lines():
            if line.get_marker() == "^":
                line.set_markersize(6)
            else:
                line.set_markersize(5)
            line.set_markeredgecolor("none")
        ax.set_xticks(ds["bin_idx"].unique())

        # secondary x axis with percent data on the top
        # only show it for the first and second plot
        if i in [0, 1]:
            ax2 = ax.secondary_xaxis("top")
            ax2.set_xticks(ds["bin_idx"].unique())
            ax2.set_xticklabels(["2.5%", "5%", "10%", "20%", "40%", "80%"])
        # make sure the y axis labels have two decimal places
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))

    # replace the legend labels with Tiny, Small and Base
    handles, _ = axes[0, 0].get_legend_handles_labels()
    labels = ["Tiny", "Small", "Base"]
    axes[0, 0].legend(handles, labels)
    # add legend only to the first plot
    # axes[0].legend()

    fig.tight_layout()

    # save the figures
    fig.savefig(output_file, bbox_inches="tight", dpi=150)


if __name__ == "__main__":
    main()
