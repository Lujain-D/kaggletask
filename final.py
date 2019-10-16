import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from opencage.geocoder import OpenCageGeocode
import pickle

####the idea behind those functions is that I would break countries into longitude and latitude and use that for machine learning but that was not effective for this dataset
dump_file = "countries_mapped.txt"

def map_countries():
    country_list = {}
    key = 'key'
    geocoder = OpenCageGeocode(key)
    addressfile = 'countries.txt'
    try:
        with open(addressfile, 'r') as f:
            for line in f:
                address = line.strip()
                result = geocoder.geocode(address, no_annotations='1')

                if result and len(result):
                    longitude = result[0]['geometry']['lng']
                    latitude = result[0]['geometry']['lat']
                    print(u'%f;%f;%s' % (latitude, longitude, address))
                    country_list[address] = [latitude, longitude]
                else:
                    #sys.stderr.write("not found: %s\n" % address)
                    print("not found: %s\n" %address)
                    country_list[address] = [0, 0]

    except IOError:
        print('Error: File %s does not appear to exist.' % addressfile)


    filehandler = open(dump_file, 'wb')
    pickle.dump(country_list, filehandler)


def get_countries_mapping():
    filehandler = open(dump_file, 'rb')
    object = pickle.load(filehandler)
    return object;


def geocode(data_set):
    lat = []
    lon = []

    countries_mapping = get_countries_mapping()
    print(countries_mapping["Belarus"][0])

    for row in data_set["Country"]:

        try:
            # print (row)
            lat.append(countries_mapping[row][0])
            lon.append(countries_mapping[row][1])

        except:
            lat.append(np.NaN)
            lon.append(np.NaN)

    # Create two new columns from lat and lon
    data_set['latitude'] = lat
    data_set['longitude'] = lon

dataset = pd.read_csv('/Users/user/PycharmProjects/Test Project/tcdml1920-income-ind/tcd ml 2019-20 income prediction training (with labels).csv')
#dataset['Gender'].replace(to_replace={:'unknown', 0:'unknown'}, inplace=True)

## remove any other gender values that are not in this set
gender_allowed_vals = ["female","male","other", "unknown"]
dataset.loc[~dataset["Gender"].isin(gender_allowed_vals), "Gender"] = "unknown"

## remove any other University Degree values that are not in this set
degree_allowed_vals = ['Bachelor', 'Master', 'PhD', 'No', 'unknown']
dataset.loc[~dataset["University Degree"].isin(degree_allowed_vals), "University Degree"] = "unknown"

## columns chosen for one hot encoding
one_hot_columns = ["Gender", "Country", "Profession", "University Degree"]


list = {
    "Gender": [],
    "Country": [],
    "Profession": [],
    "University Degree":[]
}

## this was done in oder to remove the columns after one hot encoding of the test dataset
for column in one_hot_columns:
    for x in dataset[column]:
        if x not in list[column]:
            if isinstance(x, str):
                list[column].append(x)


dataset = pd.get_dummies(dataset, columns=one_hot_columns)
dataset = dataset.drop(columns=["Hair Color", "Wears Glasses", "Size of City", "Instance"])

## deal with missing data
dataset = dataset.fillna(method='ffill')

y = dataset['Income in EUR'].values
X = dataset.drop(columns=["Income in EUR"]).values

##preparing the dataset for unknown professions showing up in the test dataset before training
dataset["Profession_unknown"] = pd.Series([0 for x in range(len(dataset.index))], index=dataset.index)

## spliting the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
## fit a mulidemintional plane to the data
regressor.fit(X_train, y_train)

#predict the outcome of the test data
y_pred = regressor.predict(X_test)

##get the RMSE of the test data run
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))



######## get results for the full set:
full_dataset = pd.read_csv('/Users/user/PycharmProjects/Test Project/tcdml1920-income-ind/tcd ml 2019-20 income prediction test (without labels).csv')
##get the instance for the final output file
Instance_real = full_dataset['Instance']


##turn any data in the 4 columns in list into unknow if not encountered during training
for item in list:
    full_dataset.loc[~full_dataset[item].isin(list[item]), item] = "unknown"

###apply onehot
full_dataset = pd.get_dummies(full_dataset, columns=one_hot_columns)

##drop unwanted columns
full_dataset = full_dataset.drop(columns=["Hair Color", "Wears Glasses", "Size of City", "Instance", "Income"])
full_dataset = full_dataset.fillna(method='ffill')


X_full_test = full_dataset[[dataset.columns]].values


y_pred = regressor.predict( X_full_test)

combined = np.vstack((Instance_real, y_pred))

### the final submission file
np.savetxt("combined.csv", combined.T, delimiter=",", fmt='% 4d')


# # 81603.6296069847
# #one_hot_columns = ["Gender", "Country", "Profession"]
# # dataset['University Degree'].replace(to_replace=dict(No='0', Bachelor='1', Master = '2', PhD='3'), inplace=True)
#
#
#
# # 81372.03090418325
# # one_hot_columns = ["Gender", "Country", "Profession", "University Degree"]
