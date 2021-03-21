# Ridge Regression Prediction Model for Material Dielectric Constants

**Undergraduate Research Apprenticeship Program 2021**

Bella Crouch, under the supervision of Dr. Waqas Khalid

**Project Background**

This work seeks to enhance the energy storage of carbon nanotube (CNT) electrode arrays for use in biosensing applications. 

The ultra-high energy densities of CNT arrays can be attributed to the supercapacitance that can be achieved by CNT forests. This capacitance can be improved by integrating high-$\kappa$ dielectric materials into electrode arrays; in particular, such colossal dielectrics as hafnia, alumina, and titanium dioxide are frequently used in nanotech applications to improve CNT properties. There is now great interest in identifying other possible colossal dielectric materials to further the energy storage potential of electrodes.

The dielectric constant of a material is the ratio of the material’s permittivity to the permittivity of vacuum. Material polarization is a main driver of permittivity; hence, higher dielectric constants are present in materials with strong net dipoles. Polarization can be attributed to a broad range of factors, including bond polarity, band gap, and crystal system. Because these interactions are so complex, literature has yet to derive equations that can describe dielectric constants non-empirically. 

**Project Goal**

This project aims to predict unknown dielectric constants through data modeling, and, in doing so, identify potential colossal dielectrics for use in CNT electrode arrays. The model uses ridge regression trained on material parameters from the open-source [Materials Project](https://materialsproject.org) database.

**Exploratory Data Analysis and Visualization**

The Materials Project database provides polycrystalline dielectric constants and material parameters for 500 unique materials, five of which are displayed below. Initial analysis of the uncleaned data revealed a number of extreme outlying data points, as well as poor linearity between dielectric constants and potential explanatory variables. Dielectric constants are plotted against band gap and formation energy, the two variables that displayed the most visible associations prior to feature engineering.

#Import libraries and Materials Project API token
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pymatgen
import sklearn
import plotly
import plotly.express as px
import plotly.graph_objects as go
from IPython.core.display import display, HTML
from plotly.offline import init_notebook_mode, plot
init_notebook_mode(connected=True)

#Load data
ptable = pd.read_csv("Periodic Table of Elements.csv").set_index("Symbol")
mat_data = pd.read_csv("Materials Project Data.csv").drop(["Theoretical", "Count", "Has Bandstructure", "E Above Hull (eV)"], axis=1)
mat_data["Polycrystalline Dielectric Constant"] = pd.to_numeric(mat_data["Polycrystalline Dielectric Constant"].str.replace(",", ""))
mat_data.head(5)

#Initial EDA and visualization
overview = px.scatter_3d(mat_data, x='Band Gap (eV)', y='Formation Energy (eV)', 
                         z='Polycrystalline Dielectric Constant', color='Crystal System', 
                         title="Dielectric Constant as a Function of Unfeaturized <br> Formation Energy and Band Gap Data")
overview.update_traces(marker=dict(size=4, opacity=0.7), selector=dict(mode='markers'))
overview.update_layout(autosize=False, width=800, height=500, margin=dict(l=50, r=50, b=0, t=50),
                      scene_camera=dict(eye=dict(x=2.25, y=1.0, z=1.0), center=dict(x=0, y=0, z=-0.3)),
                      title_x=0.5, title_y = 0.9, legend_x=0.8, legend_y=0.9)
plot(overview, filename="overview.html")
display(HTML("overview.html"))

**Feature Engineering**

Several measures were taken to prepare the design matrix for regression fitting:
* Data points with outlying values of `Polycrystalline Dielectric Constant` were removed from the dataset. The response feature was then log-transformed to improve the scaling of the dataset.
* The features `Crystal System` and `Spacegroup` were one-hot encoded to produce numerical values for model fitting.
* All numerical features were standardized to reduce any overweighting of variables with higher average magnitudes.
* A new feature, `Max Electronegativity Difference`, was created to determine the maximum difference in electronegativity between any elements in the material (as a proxy for net dipole moment).

The resulting featurized dataset displayed a stronger association between the new predictor (`Log Polycrystalline Dielectric Constant`) and possible regressors.

#Define feature engineering utility functions
#to add: net dipole moment, breakdown voltage, deposition thickness, features from lit review
from sklearn.preprocessing import OneHotEncoder

def feature_pipeline(df, feature_mappings, y_col):
    """
    Input: 
        df (dataframe): dataframe to be transformed
        feature_mappings (dict): dictionary mapping feature functions to their input arguments eg {one_hot_encode:{"col_name":col}}
        y_col (str): name of column containing the response vector
    Output:
        X (dataframe): design matrix of engineered features with all non-numeric columns dropped
        y (series): response vector    
    """
    featurized = df.copy()
    for feat_function in feature_mappings:
        featurized = feat_function(featurized, **feature_mappings[feat_function])
    y = featurized[y_col]
    X = featurized.drop(columns=[y_col])._get_numeric_data()
    return X, y

def one_hot_encode(df, col_name):
    """
    Input:
        df (dataframe): dataframe to be transformed
        col_name (str): name of column to be encoded
    Output:
        original dataframe with the data in col_name one-hot-encoded (col_name is dropped) """
    output = df.copy()
    for col in col_name:
        ohe = OneHotEncoder(handle_unknown='ignore')
        ohe.fit(df[[col]])
        encoding = pd.DataFrame(ohe.transform(df[[col]]).todense(), columns = ohe.get_feature_names(), index = df.index)
        output = output.merge(encoding, left_index=True, right_index=True).drop(columns=[col])
    return output

def standardize(df, col_name):
    """
    Input:
        df (dataframe): dataframe to be transformed
        col_name (str): name of column to be standardized
    Output:
        original dataframe with data in col_name standardized to have 0 mean and 1 SD"""
    standardized = df.copy()
    for col in col_name:
        standardized[col] = (df[col]-np.mean(df[col]))/np.std(df[col])
    return standardized

def remove_outliers(df, col_name, min_lim=-np.inf, max_lim=np.inf):
    """
    Input:
        df (dataframe): dataframe to be transformed
        col_name (str): name of column containing outliers
        min_lim (int): lowest value of col_name to be contained in returned dataframe
        max_lim (int): highest value of col_name to be contained in returned dataframe
    Output:
        original dataframe filtered to only contain rows with min_lim <= col_name <= max_lim"""
    return df[(df[col_name]>=min_lim) & (df[col_name]<=max_lim)]

#Queries from Materials Project and takes a looooong time to run - USE SAVED `MAX ENEG DIFF` CSV INSTEAD
"""def eneg_transform(df, col_name):
    with_eneg = df.copy()
    def max_eneg_diff(mat_id):
        enegs = []
        for elem in m.query(criteria={"task_id": mat_id}, properties=["elements"])[0]["elements"]:
            enegs.append(ptable.loc[elem, "Electronegativity"])
        return max(enegs) - min(enegs)
    eneg_series = np.array([max_eneg_diff(entry) for entry in df[col_name]])
    with_eneg["Max Eneg Diff"] = eneg_series
    return with_eneg"""

def log_transform(df, col_name):
    """
    Input:
        df (dataframe): dataframe to be transformed
        col_name (str): name of column to be log-transformed
    Output:
        original dataframe with additional 'Log col_name' column"""
    transformed = df.copy()
    transformed["Log " + col_name] = np.log(df[col_name])
    return transformed.drop(columns=[col_name])

#Process data to create design matrix X and response vector y
mat_data["Max Eneg Diff"] = pd.read_csv("Max Eneg Diff")["Max Eneg Diff"]
X, y = feature_pipeline(mat_data, {one_hot_encode:{"col_name":["Crystal System", "Spacegroup"]},
                                           standardize:{"col_name":["Formation Energy (eV)", "Band Gap (eV)", "Volume", "Nsites", "Density (gm/cc)", "Max Eneg Diff"]},
                                           remove_outliers:{"col_name":"Polycrystalline Dielectric Constant", "max_lim":50},
                                           log_transform:{"col_name":"Polycrystalline Dielectric Constant"},},
                                           "Log Polycrystalline Dielectric Constant")

#Visualize featurized dataset
featurized_plot = X.copy()
featurized_plot["Log Polycrystalline Dielectric Constant"] = y
featurized_plot["Crystal System"] = remove_outliers(mat_data, "Polycrystalline Dielectric Constant", max_lim=50)["Crystal System"]
featurized = px.scatter_3d(featurized_plot, x='Band Gap (eV)', y='Formation Energy (eV)', 
                           z='Log Polycrystalline Dielectric Constant', color='Crystal System',
                          title="Dielectric Constant as a Function of Featurized <br> Formation Energy and Band Gap Data",
                          labels={"Band Gap (eV)":"Std Band Gap", "Formation Energy (eV)":"Std Formation Energy"})
featurized.update_traces(marker=dict(size=4, opacity=0.7), selector=dict(mode='markers'))
featurized.update_layout(autosize=False, width=800, height=500, margin=dict(l=50, r=50, b=0, t=50),
                      scene_camera=dict(eye=dict(x=2.25, y=1.0, z=1.0), center=dict(x=0, y=0, z=-0.3)),
                      title_x=0.5, title_y = 0.9, legend_x=0.8, legend_y=0.9)
plot(featurized, filename="featurized.html")
display(HTML("featurized.html"))
#featurized.show()

**Model Fitting and Performance**

A ridge regression model with cross-validated regularization penalty $\alpha = 0.2$ was fitted to a training set of 80% of the cleaned dataset. The predictions made by the model on the training set are displayed below, as are the true values of each of these data points.

#Split training and test sets
from sklearn.model_selection import train_test_split
np.random.seed(1337)
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

#Define model fitting utility functions
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.base import clone

def rmse(y, y_hat):
    return np.sqrt(np.mean((y - y_hat)**2))

def r2(y, y_hat):
    return np.var(y_hat)/np.var(y)

def cross_validate_alpha(model, X, y):
    alphas = np.linspace(0.5, 5, 50)
    cvmodel = model
    five_fold = KFold(n_splits=5)
    rmse_values = []
    for tr_ind, va_ind in five_fold.split(X):
        model.fit(X.iloc[tr_ind,:], y.iloc[tr_ind])
        rmse_values.append(rmse(y.iloc[va_ind], model.predict(X.iloc[va_ind,:])))
    return np.mean(rmse_values)

#Cross-validate ridge regression model and evaluate training performance
dielec_model = RidgeCV(np.linspace(0.1, 10, 1000), fit_intercept=False)
dielec_model.fit(X = xtrain, y = ytrain)

train_predictions = dielec_model.predict(xtrain)
delogged_train_predictions = np.exp(train_predictions)
delogged_train_y = np.exp(ytrain)

train_rmse = rmse(delogged_train_y, delogged_train_predictions)
train_r2 = dielec_model.score(xtrain, ytrain)

#print("Best alpha:", dielec_model.alpha_)
#print("Training RMSE:", train_rmse)
#print("Training R2:", train_r2)

#Visualize predictions
og = xtrain.copy()
og["Dielectric Constant"] = delogged_train_y
og["Is Predicted"] = ["Actual Dielectric Constant" for i in range(len(og))]

pred = xtrain.copy()
pred["Dielectric Constant"] = delogged_train_predictions
pred["Is Predicted"] = ["Predicted Dielectric Constant" for i in range(len(pred))]

pred_with_og = pd.concat([og, pred])
pred_with_og

pred_vis = px.scatter_3d(pred_with_og, x='Band Gap (eV)', y='Formation Energy (eV)', 
                           z='Dielectric Constant', color="Is Predicted",
                           title="Predicted and Actual Dielectric Constants as Functions <br> of Standardized Band Gap and Formation Energy",
                           labels={"Band Gap (eV)":"Std Band Gap", "Formation Energy (eV)":"Std Formation Energy", "Dielectric Constant":"Polycrytalline Dielectric Constant"})

pred_vis.update_traces(marker=dict(size=4, opacity=0.7), selector=dict(mode='markers'))
pred_vis.update_layout(autosize=False, width=800, height=500, margin=dict(l=50, r=50, b=0, t=50),
                      scene_camera=dict(eye=dict(x=2.25, y=1.0, z=1.0), center=dict(x=0, y=0, z=-0.3)),
                      title_x=0.5, title_y = 0.9, legend_x=0.8, legend_y=0.9, legend_title_text="")

plot(pred_vis, filename="pred_vis.html")
display(HTML("pred_vis.html"))

The optimized training root mean squared error of 5.07 corresponds to ± 23% error relative to the dataset's mean dielectric constant of 21.2. The $R^{2}$ value of 0.58 indicates moderate linear association with the original response variable; however, plots of residual values and the predicted dielectric constants against the actual values suggest that this relationship decays at higher values of $\kappa$.

#Visualize residuals of model

from plotly.subplots import make_subplots

residuals = make_subplots(cols=2, subplot_titles=("Plot of Residuals Against Predicted <br> Dielectric Constants", "Plot of Predicted Against Actual <br> Dielectric Constants"))
residuals.add_scatter(x=delogged_train_y, y=delogged_train_predictions, mode="markers", row=1, col=2, showlegend=False)

residuals.add_scatter(x=[10, 45], y=[10, 45], mode="lines", name="Perfect Correspondence <br> Between Predicted and <br> Actual Dielectric Constant", row=1, col=2)
residuals.add_scatter(x=delogged_train_predictions, y=delogged_train_y-delogged_train_predictions, mode="markers", row=1, col=1, showlegend=False)

residuals.update_layout(autosize=False, width=1000, height=400, margin=dict(l=0, r=0, b=0, t=50),
                      title_x=0.08, title_y = 0.95, xaxis= dict(tickvals = np.linspace(10, 45, 8)))
residuals.update_xaxes(title_text="Actual Dielectric Constant", row=1, col=2)
residuals.update_yaxes(title_text="Predicted Dielectric Constant", row=1, col=2)
residuals.update_xaxes(title_text="Predicted Dielectric Constant", row=1, col=1)
residuals.update_yaxes(title_text="Residual", row=1, col=1)

plot(residuals, filename="residuals.html")
display(HTML("residuals.html"))

**Next Steps**

After further refinement, the model will be used to produce predictions for the untouched test set as a final measure of accuracy. 

To do:
* Incorporate experimental values for net dipole moment
* Incorporate breakdown voltages
* Identify additional datasets that can be used for model training