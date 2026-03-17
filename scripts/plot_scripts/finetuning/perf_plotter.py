###############################################################################
# Author: Mohammad Mostafanejad                                               #
# Date: February 2026                                                         #
# Description:                                                                #
# This module will use the 'ds_effect.xls' raw data file to plot the          #
# dataset size effect on model performance.                                   #
###############################################################################

# import the necessary modules
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def main(properties, mode: str = None):
    if mode is not None and mode not in ["val", "test"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'val' or 'test'.")
    for prop in properties:
        if prop not in ["HLM", "hPPB", "SOL"]:
            raise ValueError(
                f"Invalid property: {prop}. Must be one of HLM, hPPB, SOL."
            )

        # input and output file paths
        input_file = "./sft_adme_datasize_effect.xls"
        output_file = f"./sft_{mode}_{prop}_ds_eff_perf_plot.pdf"

        # load the dataset
        ds = pd.read_excel(input_file, sheet_name=f"sft_{mode}_{prop}")

        # set the font size
        plt.rcParams["font.size"] = 9
        plt.rcParams["axes.titlesize"] = 9
        plt.rcParams["axes.labelsize"] = 9

        # set the legend font size
        plt.rcParams["legend.fontsize"] = 8

        # set the resolution
        plt.rcParams["figure.dpi"] = 150

        # grid of line plots with error bars for pearson_r, r2, rmse and mae
        metrics = [f"{mode}_pearson_r", f"{mode}_r2", f"{mode}_rmse", f"{mode}_mae"]

        # set the color palette to blue, red and green
        sns.set_palette(["blue", "red", "green"])

        # set the circle, triangle, square markers for each Name
        markers = {"tiny": "o", "small": "s", "base": "^"}

        # create fig and axes objects
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 5), dpi=150, sharex=True, sharey=False)

        # create a list of y axis labels
        labels = ["Pearson $R$", "$R^2$", "RMSE", "MAE"]

        # plot titles
        plt_titles = ["(a)", "(b)", "(c)", "(d)"]

        # create the plots
        if mode == "val":
            for i, metric in enumerate(metrics):
                row = i // 2
                col = i % 2
                sns.lineplot(
                    data=ds,
                    x="bin_idx",
                    y=metric,
                    hue="Name",
                    style="Name",
                    markers=markers,
                    ax=axes[row, col],
                    err_style="bars",
                    errorbar=("ci", 95),
                    err_kws={"capsize": 3, "capthick": 1, "elinewidth": 1},
                )
                axes[row, col].set_title(plt_titles[i], loc="left", pad=10, fontweight="bold")
                axes[row, col].set_ylabel(labels[i])
        else:
            # bar plots for the test mode without error bars
            for i, metric in enumerate(metrics):
                row = i // 2
                col = i % 2
                sns.barplot(
                    data=ds,
                    x="bin_idx",
                    y=metric,
                    hue="Name",
                    ax=axes[row, col],
                    errorbar=None,
                )
                axes[row, col].set_title(plt_titles[i], loc="left", pad=10, fontweight="bold")
                axes[row, col].set_ylabel(labels[i])

        for ax in axes.flatten():
            # set the x axis label to k
            ax.set_xlabel("$k$")
            # make sure the y axis numbers are formatted with 1 decimal place
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{x:.1f}")
            )
            if mode == "test":
                # set the x axis ticks to 0, 1, 2 and label them as 0, 3, 5 for test mode
                ax.set_xticks([0, 1, 2])
                ax.set_xticklabels([0, 3, 5])
                # set the y axis limits to 0, 0.7 for 
                ax.set_ylim(0, 0.7)
                

            # remove the lines connecting the markers in the line plots in val mode
            if mode == "val":
                for line in ax.lines:
                    line.set_linestyle("")
                    
            # for the second graph, remove the legend title
            ax.legend(title="")
            # compact the legend
            ax.legend(
                loc="best",
                frameon=True,
                ncol=1,
                borderaxespad=0.3,
                handletextpad=0.3,
                columnspacing=0.3,
                labelspacing=0.3,
            )
            # remove the legend for all but the second graph
            if ax != axes[0, 1]:
                ax.legend().remove()

        # add legend only to the first plot
        fig.tight_layout()

        # save the figures
        fig.savefig(output_file, bbox_inches="tight", dpi=150)

        # close the figure to free up memory
        plt.close(fig)


if __name__ == "__main__":
    properties = ["HLM", "hPPB", "SOL"]
    for mode in ["val", "test"]:
        main(properties=properties, mode=mode)
