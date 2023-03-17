import streamlit as st
from joblib import load
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import altair as alt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier as KNN
st.title("Predicting Well Facies")

st.write("Hi. This is a small project that I've visually represented in this web application. It's a little chaotic so lemme explain it: Basically, there are different facies (Layers) in the ground, each with their own properties. If we drill a hole in the ground, we'll get different samples at different depths, each with different properties. If we know the actual layers in the ground in one region and extract samples, we could theoretically predict geomorphology and lithographic history in another nearby region in which we don't know the facies by sampling it. This web app takes a large chunk of the project I've been working on and demonstrates some of the descisions I made through some graphs. If you want to read more, I have an amateur paper at the bottom of this page. The data for this project comes from a block in the Dutch sector of the North Sea. Our first step in creating an AI model is to take the data and clean it up")

st.write("After processing and labelling data, the training features were split from the target. The preliminary model uses linear regression and yields different accuracy depending on the parameters. Parameters of density, porosity, P-wave impedance, relative P-wave impedance, and gamma-ray were chosen. These parameters were chosen just by playing around and experimenting with combinations until one with the highest accuracy was found. You can experiment with these parameters and view how the predictions change depending on the factors you select.")
st.write("Select the factors to include in the model")

# Load data
data=pd.read_csv("regdata.csv")
X_cols = ['RHOB', 'DT', 'GR', 'AI', 'AIR', 'PHIE']
y_col = 'label'

# User selection of input factors
selected_cols = st.multiselect("Select factors to include", X_cols, default=X_cols)
X = data[selected_cols]
y = data[y_col]

# Define KNN pipeline
pipe = Pipeline([('scaler', StandardScaler()),
                 ('clf', LogisticRegression(multi_class='auto', solver='liblinear', 
                                            max_iter=1000, random_state=42))])



# Define stratified sampling CV 
cv = StratifiedKFold(5, shuffle=True)
st.write("To avoid overfitting on the F-01 well, cross validator was used in the form of a stratified k-fold from SKLearn. ")
# Cross-validation
scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
#st.write("Cross-validation accuracy scores:", scores)
st.write("Average cross-validation accuracy:", np.mean(scores))

# Fit model on full data
pipe.fit(X, y)

# Plot predicted and true facies
y_pred = pipe.predict(X)
facies_list = np.unique(y)
logs = X.columns

fig, ax = plt.subplots(figsize=(8,6))

ax.plot(y, data.DEPTH, color='k', lw=0.5)
F = np.vstack((y_pred,y_pred)).T
ax.imshow(F, aspect='auto', extent=[min(facies_list)-0.5, max(facies_list)+0.5, max(data.DEPTH), min(data.DEPTH)],
          cmap='viridis', alpha=0.4)

ax.set_xlabel('Facies')
ax.set_ylabel('Depth (m)')
ax.set_title('Well Facies Prediction')

# Show plot
st.pyplot(fig)
model = load("model.pkl")
data=pd.read_csv("graphdata.csv")
st.write("Well-log data has a high resolution because of its highest frequency in the range of 20 to 40 kHz. This usually is good, as this frequency can capture small contacts between two different lithofacies as accurately as 10-20 centimeters. But this accuracy level is not needed for our data, which contains facies thickness that ranges from 20-100 meters. Filtering out the high frequency will help accuracy levels. Filtering the high frequency and retaining the low frequency can be done by implementing a Butterworth filter, which depend on a cutoff value. Play around with to slider to see how the data is filtered. (Lower values = More filtered")
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def showgraph():
    data['AILP'] = butter_lowpass_filter(data.AI.values, cutoff, 1000/4, order=5) 
    data['AIRLP'] = butter_lowpass_filter(data.AIR.values, cutoff, 1000/4, order=5) 
    data['RHOBLP'] = butter_lowpass_filter(data.RHOB.values, cutoff, 1000/4, order=5) 
    data['GRLP'] = butter_lowpass_filter(data.GR.values, cutoff, 1000/4, order=5) 
    data['PHIELP'] = butter_lowpass_filter(data.PHIE.values, cutoff, 1000/4, order=5) 

    df = data[['DEPTH', 'RHOB', 'RHOBLP']]
    df = pd.melt(df, id_vars='DEPTH', value_vars=['RHOB', 'RHOBLP'], var_name='Type', value_name='Value')

    chart = alt.Chart(df).mark_line().encode(
        x='DEPTH',
        y='Value',
        color='Type'
    ).properties(
        width=800,
        height=300
    ).interactive()

    st.altair_chart(chart)
    

cutoff = st.slider("cutoff", 1, 10, value=1)
showgraph()
sdata = data 
st.write("While the standard regression model performed alright, we can do a lot better. By choosing a better model with our newly filtered data, we'll be able to achieve near perfect accuracy. To pick the best model I looked at the results from multiple models and picked the one with the highest accuracy. You can play around and see the accuracy levels for 5 AI prediction models. You will see that the KNN model performed the best, with an accuracy of 99%!")
sdata['AILP'] = butter_lowpass_filter(sdata.AI.values, 4, 1000/4, order=5) 
sdata['AIRLP'] = butter_lowpass_filter(sdata.AIR.values, 4, 1000/4, order=5) 
sdata['RHOBLP'] = butter_lowpass_filter(sdata.RHOB.values, 4, 1000/4, order=5) 
sdata['GRLP'] = butter_lowpass_filter(sdata.GR.values, 4, 1000/4, order=5) 
sdata['PHIELP'] = butter_lowpass_filter(sdata.PHIE.values, 4, 1000/4, order=5) 
import streamlit as st
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("regdata.csv")
X_train = sdata[['RHOBLP', 'AILP', 'GRLP', 'PHIELP', 'AIRLP']]
y_train = data['label']

# Define stratified sampling CV 
cv = StratifiedKFold(5, shuffle=True)

# Define models
models = {'Random Forest': RandomForestClassifier(),
          'SVM': svm.SVC(),
          'Naive Bayes': GaussianNB(),
          'Decision Tree': tree.DecisionTreeClassifier(),
          'KNN': KNeighborsClassifier(n_neighbors=4)}

# Streamlit App
st.write("Select the model to use")

# User selection of model
selected_model = st.selectbox("Select model to use", list(models.keys()))
if(selected_model == 'KNN'):
    st.success('sheesh this model is fire')

# Define pipeline
pipe = make_pipeline(StandardScaler(), models[selected_model])

# Cross-validation
cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='accuracy')
mean_cv_scores = np.mean(cv_scores)
st.write(f"Cross-validation accuracy scores for {selected_model}:", cv_scores)
st.write(f"Average cross-validation accuracy for {selected_model}:", mean_cv_scores)

# Fit model to training data
pipe.fit(X_train, y_train)

# Predict facies on training data
y_pred = pipe.predict(X_train)

# Plot predicted and true facies
facies_list = np.unique(y_train)
logs = X_train.columns

fig, ax = plt.subplots(figsize=(8,6))

ax.plot(y_train, data.DEPTH, color='k', lw=0.5)
F_true = np.vstack((y_train,y_train)).T
F_pred = np.vstack((y_pred,y_pred)).T
ax.imshow(F_true, aspect='auto', extent=[min(facies_list)-0.5, max(facies_list)+0.5, max(data.DEPTH), min(data.DEPTH)],
          cmap='viridis', alpha=0.4)
ax.imshow(F_pred, aspect='auto', extent=[min(facies_list)-0.5, max(facies_list)+0.5, max(data.DEPTH), min(data.DEPTH)],
          cmap='plasma', alpha=0.4)

ax.set_xlabel('Facies')
ax.set_ylabel('Depth (m)')
ax.set_title(f'{selected_model} Facies Prediction')

# Display plot
st.pyplot(fig)


st.write("I hope you thought this was as cool as I did. If you are curious and want to learn more, I go in more depth and predict an unlabeled well in the paper linked below. Its extremely amateur and I just wrote it for fun so dont get mad at me if you dont like the grammar or something. DGMAM")

st.write("paper: <insert later>")

