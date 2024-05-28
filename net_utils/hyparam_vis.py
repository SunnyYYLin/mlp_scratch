import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def results_from_json(file):
    '''
    Load the results of a hyperparameter search from a json file.
    '''
    with open(file, 'r') as f:
        results = json.load(f)
    return results

def plot_all(dataframe, variable_param, by_param, target):
    color_keys = sorted(dataframe[by_param].unique())
    color_map = plt.cm.get_cmap('plasma', len(color_keys))
    colors = {key: color_map(i) for i, key in enumerate(color_keys)}
    # colors = {50:'r', 100:'g', 200:'b'}
    fixed_params = ['learning_rate', 'lr_decay', 'hidden_size', 'reg', 'iter_num']
    fixed_params.remove(variable_param)
    combinations = dataframe[list(fixed_params)].drop_duplicates()
    added_labels = set()  # 创建一个空集合用于存储已添加的标签
    
    plt.figure(figsize=(14, 7))
    
    # 为每个组合绘制一条曲线
    for _, combo in combinations.iterrows():
        
        filter_condition = pd.Series(True, index=dataframe.index)
        label_params = []
        
        # 构建筛选条件并准备图例标签
        for param in fixed_params:
            filter_condition &= (dataframe[param] == combo[param])
            label_params.append(f"{param}={combo[param]}")
        
        # 筛选数据
        filtered_df = dataframe[filter_condition]
        
        if not filtered_df.empty:
            sorted_df = filtered_df.sort_values(by=variable_param)
            label = by_param + ': ' + str(combo[by_param])
            if label in added_labels:
                label = None
            else:
                added_labels.add(label)
            plt.plot(sorted_df[variable_param], sorted_df[target], 
                        color = colors[combo[by_param]], label = label)
            
    plt.title(f'{target} vs. {variable_param}')
    plt.xlabel(variable_param)
    plt.ylabel(target)
    plt.legend(title='Combinations', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.xscale('log' if dataframe[variable_param].dtype in ['float64', 'int64'] else 'linear')
    plt.grid(True)
    plt.show()
    
def plot_mean(df, x, y):
    """
    Plot a line graph with error bars representing the mean and standard deviation of the y values grouped by x.
    
    Parameters:
    df (pd.DataFrame): The data frame containing the data.
    x (str): The name of the column to be used as the x-axis.
    y (str): The name of the column to be used as the y-axis.
    """
    # Group data by the x column and calculate mean and standard deviation of the y column
    grouped = df.groupby(x)[y].agg(['mean', 'std']).reset_index()
    
    # Extract grouped data
    x_values = grouped[x]
    mean_y = grouped['mean']
    std_y = grouped['std']
    
    # Create a line plot with error bars and logarithmic scale for the x-axis
    plt.figure(figsize=(10, 6))
    plt.errorbar(x_values, mean_y, yerr=std_y, fmt='-o', ecolor='r', capsize=5, capthick=2)
    plt.xscale('log')
    plt.title(f'Impact of {x} on {y}')
    plt.xlabel(f'{x} (Log Scale)')
    plt.ylabel(y)
    plt.grid(True, which='both', ls='--')
    plt.show()
    
def plot_mean_by_group(df, x, y, group=None):
    # 检查分组参数，并据此设置分组键
    if group is None:
        # 如果没有分组，计算整个数据集的平均值
        mean_val = df[y].mean()
        plt.figure(figsize=(10, 6))
        plt.axhline(y=mean_val, color='r', linestyle='-')
        plt.text(df[x].min(), mean_val, f'Average {y}: {mean_val:.2f}', va='center', ha='right', backgroundcolor='w')
        plt.title(f'Overall Average {y}')
        plt.xlabel(x)
        plt.ylabel(y)
    else:
        # 如果有分组，按照给定的分组键进行分组
        if isinstance(group, tuple) or isinstance(group, list):
            group_keys = list(group)
        else:
            group_keys = [group]
            
        # 获取组的唯一值并排序，为每个组分配颜色
        grouped = df.groupby(group_keys + [x])[y].mean().reset_index()
        unique_combinations = grouped[group_keys].drop_duplicates()
        sorted_combinations = unique_combinations.sort_values(by=group_keys)

        # 为每个唯一组合分配颜色
        color_map = plt.cm.get_cmap('rainbow', len(sorted_combinations))
        colors = {tuple(row): color_map(i) for i, row in enumerate(sorted_combinations.itertuples(index=False))}


        # 计算每个分组的平均值
        grouped = df.groupby(group_keys + [x])[y].mean().reset_index()

        # 转换为透视表以便绘图
        pivot_df = grouped.pivot_table(index=x, columns=group_keys, values=y)

        # 绘图
        plt.figure(figsize=(12, 8))
        for column in pivot_df.columns:
            plt.plot(pivot_df.index, pivot_df[column], marker='o', label=f'Group {column}', color=colors[tuple(column)])

        plt.title(f'{y} by {x} grouped by {group_keys}')
        plt.xlabel(x)
        plt.ylabel(f'Mean {y}')
        plt.legend(title='Groups')
        plt.grid(True)
        if (df[x] > 0).all() and (df[x].max() / df[x].min() > 100):
            plt.xscale('log')
    plt.show()
    
if __name__ == "__main__":
    results = results_from_json('results.json')
    results = pd.DataFrame(results)
    
    # plot_all(results, 'lr_decay', 'learning_rate', 'val_acc') # lr_lrd.png
    plot_mean_by_group(results, 'reg', 'val_acc', ('hidden_size', 'iter_num'))
    # plot_mean(results, 'reg', 'val_acc')