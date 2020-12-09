from sklearn import linear_model, cluster, datasets
import matplotlib.pyplot as plt
import pandas as pd


def linear():
    # load boston dataset
    boston = datasets.load_boston()
    # create boston dataset dataframe
    boston_df = pd.DataFrame(boston.data, columns=boston.feature_names.tolist())
    # run linear model
    lm = linear_model.LinearRegression().fit(boston_df, boston.target)
    # create df that has feature names, coefficient values, and absolute value of coefficient values (for sorting)
    coef_df = pd.DataFrame(columns=['feature', 'coefs'])
    coef_df['feature'] = boston.feature_names.tolist()
    coef_df['coefs'] = lm.coef_.tolist()
    coef_df['coefs_abs'] = abs(coef_df['coefs'])
    # print only the raw coefficients and the feature names sorted by the absolute value column. Sorting by absolute
    # value is necessary because we are looking for greatest impact, whether positive or negative as discussed in class
    print(coef_df.sort_values('coefs_abs', ascending=False).reset_index(drop=True)[['feature', 'coefs']])


def elbow():
    # load iris dataset
    iris = datasets.load_iris()
    # create iris dataset dataframe
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    sum_squared_error = []
    x = []
    for k in range(1, 15):
        kmeans = cluster.KMeans(n_clusters=k).fit(iris_df)
        # inertia_float (from sklearn docs)
        # Sum of squared distances of samples to their closest cluster center.
        sum_squared_error.append(kmeans.inertia_)
        x.append(k)

    # After inspecting the graph produced by plt.plot(x, sum_squared_error), it can be confirmed that 3 is the correct
    # number of clusters or populations to use. To make this more visually obvious I have modified my graph to
    # highlight 3 clusters on the graph
    plt.plot(x[0:3], sum_squared_error[0:3], 'g')
    plt.plot(x[2:], sum_squared_error[2:], 'r')
    plt.plot(x[2], sum_squared_error[2], 'bo')
    plt.title("Elbow Graph")
    plt.xlabel('# of Clusters')
    plt.ylabel('Sum of Squared Error')
    plt.show()


if __name__ == '__main__':
    linear()
    elbow()
