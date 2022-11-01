from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv('Crop_recommendation.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# ==========crop yield====================================================

crop_names = {"wheat": 1, "rice": 2, 'maize': 3, 'gram': 4}

dataset = pd.read_csv('cropyield.csv')
dataset.drop("District", axis=1, inplace=True)
dataset.drop("Season", axis=1, inplace=True)
# dataset.drop("Crop", axis=1, inplace=True)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#print(X_train)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)
crop = "wheat"
cropid = crop_names[crop]
area = 5
y_pred = regressor.predict([[cropid,area]])
#print(X_test)
print(y_pred)
# =========================================================================
app = Flask(__name__)
local_server = True


@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        '''   entry values '''
        state = request.form['state']
        district = request.form['district']
        n = request.form['n']
        p = request.form['p']
        k = request.form['k']
        temperature = request.form['temperature']
        humidity = request.form['humidity']
        ph = request.form['ph']
        rainfall = request.form['rainfall']

        # crop_yield (District,Season,Crop,Area)
        '''c_district = request.form['c_district']
        season = request.form['season']
        crop = request.form['crop']
        area = request.form['area']'''

        recom_crop = str(classifier.predict(sc.transform([[n, p, k, temperature, humidity, ph, rainfall]])))
        crop = recom_crop[2:-2]
        return "Recommended Crop is " + crop

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
