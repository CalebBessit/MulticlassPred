import numpy as np
import matplotlib.pyplot as plt
import matplot2tikz

METHODS = ["first", "random", "random_weighted"]
models = ["Random Forest","MLP","XGBoost"]

styles = [
    {'color':'green',
     'marker':'^',
     'linestyle':'-'},
    {'color':'red',
     'marker':'o',
     'linestyle':'--'},
    {'color':'blue',
     'marker':'s',
     'linestyle':':'},
]

points = [100,200,300,400,500,600,750,1000,1500,2000]
points = [100,200,500,1000]
# points = [100,500,1000]
# points = [200]
nbhers = [10,20,30,50,75,100,150,200,500,1000]  
nbhers = [10,50,100,500,1000]
# nbhers = [10,50,100,200]

# , varying_points.npy
points_data, nbh_data = np.load("varying_points_pg.npy"), np.load("varying_nbh_pg.npy")

SETUP = 1


metrics = ["Accuracy","Balanced Accuracy","Precision","Recall","F1","Mean IoU","ROC AUC"]
metric_index = 5
metric = metrics[metric_index]


report_metrics = [0,1,5]


# for k, model in enumerate(models):
#     print(f"\n### {model} ###")
#     for i, method in enumerate(METHODS):
#         method_str = f"     + {method:^18s}: | "
#         for j, metric_value in enumerate(report_metrics):
#             subset = points_data[i]
#             value = subset[4,k,metric_value]
#             method_str += f" & {value:.3f}  "
#         print(method_str)

for method in ['random']:
    i=1
    plt.figure()

    plt.title(f"{metric} versus points with {method} point selection")
    # plt.xlabel(f"Cube root of total number of voxels")
    plt.xlabel(f"Number of neighbours used for feature computation (logarithmic scale)")
    plt.xscale('log')
    plt.ylabel(f"{metric}")
    plt.grid(alpha=0.5)
    
    if SETUP==0:
        subset = points_data[i]
        xdata = points
    else:
        subset = nbh_data[i]
        xdata = nbhers

    for j, model in enumerate(models):
        values = subset[:,j,metric_index]
        style = styles[j]
        plt.plot(
            xdata, values,
            label = models[j],
            color=style['color'],
            marker =style['marker'],
            linestyle=style['linestyle'],
            alpha=0.85
        )
    plt.legend()

    # if method=="random":
    matplot2tikz.save(f"../figures/ms_exp_{SETUP+1}_{method}_{metric}.tex")



plt.show()