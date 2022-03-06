import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from shutil import rmtree
from os.path import join as opj
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram

st.set_page_config(layout="wide")
warnings.filterwarnings("ignore")


@st.cache
def rendering_plots(outp, data, methods, metrics, cols):
    global enlarge
    global dp
    try:
        for method in methods:
            for metric in metrics:
                for cluster_n in range(1, len(cols)+1):
                    try:
                        plt.figure(figsize=(enlarge, enlarge))
                        labels = data.columns.values.tolist()
                        link = linkage(data.T.values, method=method, metric=metric)
                        if cluster_n != 1:
                            dendrogram(link, labels=labels, leaf_rotation=0, truncate_mode='lastp', p=cluster_n, color_threshold=0, above_threshold_color='k', show_contracted=True, orientation="left")
                        ax = plt.gca()
                        ax.axes.xaxis.set_ticklabels([])
                        ax.tick_params(axis=u'both', which=u'both', length=0)
                        plt.title(f"{metric} - {method}")
                        plt.tight_layout()
                        plt.savefig(opj(outp, f"d{cluster_n}_{method}_{metric}.png"), dpi=dp, bbox_inches="tight")
                        plt.close()
                    except ValueError:
                        pass
    except Exception as e:
        st.text(e)


def load_plots(metrics, methods, cols, output_folder):
    for mr in metrics:
        cluster_n = st.slider(f'Choose number of clusters for {mr}:', 1, len(cols), key=mr)
        s_columns = st.columns(len(methods))
        for s_col, method in zip(s_columns, methods):
            with st.expander(label="Expand / Hide"):
                try:
                    s_col.image(opj(output_folder, f"{cluster_n}_{method}_{mr}.png"))
                    s_col.image(opj(output_folder, f"d{cluster_n}_{method}_{mr}.png"))
                except Exception:
                    pass


@st.cache
def load_data(outp, file):
    data = pd.read_excel(file, index_col=0)
    cols = [i.split(' ')[0] if "Anis" not in i else "A. "+i.split(' ')[1] for i in data.columns.values.tolist()]
    return data, cols


@st.cache
def calculating_clusters(outp, data, cols, metrics, methods):
    # metrics = distance.__all__
    # methods = hierarchy.__all__
    global enlarge
    global dp
    if outp not in os.listdir():
        os.mkdir(outp)
    else:
        rmtree(outp)
        os.mkdir(outp)
    train_data = data.T.values
    for ic, metric in enumerate(metrics):
        for od, method in enumerate(methods):
            for cluster_n in range(1, len(cols)+1):
                try:
                    plt.figure(figsize=(enlarge, enlarge))
                    model = AgglomerativeClustering(n_clusters=cluster_n, linkage=method, affinity=metric)
                    result = model.fit_predict(train_data)
                    clusters = np.unique(result)
                    for cluster in clusters:
                        index = np.where(result == cluster)
                        if cluster_n > 10:
                            plt.scatter(train_data[index, 1], train_data[index, 0], color=['k'], alpha=.5)
                        else:
                            plt.scatter(train_data[index, 1], train_data[index, 0])
                        plt.xlabel(method)
                        plt.ylabel(metric)
                        for e, label in enumerate(cols):
                            plt.annotate(label, (train_data[e, 1], train_data[e, 0]), fontsize=8)
                    plt.title(f"{metric} - {method}\nN clusters = {cluster_n}", color=f"C{cluster_n-1}")
                    plt.tight_layout()
                    plt.savefig(opj(outp, f"{cluster_n}_{method}_{metric}.png"), dpi=dp, bbox_inches="tight")
                except ValueError:
                    pass


st.header('How does the clustering work?')
st.subheader('Introduction to hierarchical agglomerative clustering.')
st.write("According to Saul Dobilas's article from towardsdatascience.com, Hierarchical Agglomerative Clustering (HAC) is a clustering algorithm, it sits under the Unsupervised branch of Machine Learning. Clustering techniques are often used for segmentation analysis or as a starting point in more complex projects that require an understanding of similarities between data points.")
st.text("")
st.subheader("You can try it with your own data!")
file = st.file_uploader("Upload data", type=["xlsx"], accept_multiple_files=False)
if file is not None:
    output_folder = "data"
    data, cols = load_data(output_folder, file)
    st.subheader("Set parameters")
    l, lm, rm, r = st.columns((5, 2, 2, 2))
    metrics = [i for i in l.multiselect("Select metric(s):", ["canberra", "chebyshev", "correlation", "euclidean"])]
    methods = [i for i in l.multiselect("Select method(s):", ["average", "complete", "single", "ward"])]
    if len(metrics) == 0 or len(methods) == 0:
        l.write("Select distance metric(s) and clustering method(s).")
    else:
        enlarge = lm.radio("\nPlot size (default: 4)", options=list(range(2, 11, 2)), index=1)
        dp = rm.radio("\nPlot quality (default: 70)", options=[50, 70, 100, 120], index=1)
        r.write("Press Refresh each time the parameters change.")
        if r.button("Refresh"):
            calculating_clusters(output_folder, data, cols, metrics, methods)
            rendering_plots(output_folder, data, methods, metrics, cols)
            e = open(opj(output_folder, "empty.txt"), "w")
            e.close()
        if "empty.txt" in os.listdir(output_folder):
            load_plots(metrics, methods, cols, output_folder)
