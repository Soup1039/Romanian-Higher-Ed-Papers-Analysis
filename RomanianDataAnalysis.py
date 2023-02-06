# Using numpy and pandas for analysis 
import numpy as np
import pandas as pd 
import datetime
# using matplot for plotting results
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#ALL FUNCTIONS 

# calculate years since publishing data
def YearsSincePublish(Years):
    currentYear = datetime.datetime.now().year
    return (currentYear - Years)

# Using K-Means clustering to analyze data
def createCenterOfData(k, data):
    # sample data randomly and return center
    centroids = data.sample(k)
    return centroids

# Now that the centers are created, calculate how far off they are from our actual data
def calcDistance(arr1, arr2):
    return np.square(np.sum((arr1-arr2)**2)) 

# Assign the position of our centroids and figure out their errors
def assignPositionOfCentroids(data, centroid):
    # get k from the number of rows (eg, 4 rows means k = 4, which means centroid num = 4)
    k = centroids.shape[0]
    n = data.shape[0]
    assignation = []
    assign_errors = []

    for obs in range(n):
        # Estimate error
        all_errors = np.array([])
        # for each centroid, look at the integer values and calculate the distance the centroid is from t
        for centroid in range(k):
            err = calcDistance(centroids.iloc[centroid, :], data.iloc[obs,:])
            all_errors = np.append(all_errors, err)

        # Get the nearest centroid and the error
        nearest_centroid =  np.where(all_errors==np.amin(all_errors))[0].tolist()[0]
        nearest_centroid_error = np.amin(all_errors)

        # Add values to corresponding lists
        assignation.append(nearest_centroid)
        assign_errors.append(nearest_centroid_error)

    return assignation, assign_errors

def adjustAlgorithm(mainData, k):
    # copying just in case the OG data gets messed up 
    copyOfData = mainData.copy()

    # Create some centroids and error tracking
    centroids = createCenterOfData(k, copyOfData)
    errorArr = []

    # Variables for future use
    x = 0
    continueOps = True
    while continueOps == True:
        print("Algorithm is being successfully run.")
        # assign pos of centroid to a column and add the error Amount to an array, so we can keep track
        copyOfData['Number of Centroids'], errorAmount = assignPositionOfCentroids(copyOfData, centroids)
        # have to sum the error amount because it's an array of current error tallys
        errorArr.append(sum(errorAmount))

        # once the algorithm assigns and updates the centroid pos using the function,update the position in our database
        # Have the mean error computed along the way to save code space
        # Because of the grouping, the indexes may get mixed up, so we reorganize
        centroids = copyOfData.groupby('Number of Centroids').agg('mean').reset_index(drop=True)
        
        # if we look at the errors and they're below our expected threshold (1E-4 in this case)
        # break out. We say if "x > 1" so the algorithm runs at least twice, if not more 
        if x>1:
            # Take the error difference to see how close the values are 
            # Looking for consistency amongst the error values 
            if errorArr[x - 1]- errorArr[x] <= (1E-4):
                continueOps = False
        x += 1
    # END WHILE 
    # Once the algorithm is run, make sure to update everything in the database
    copyOfData['Number of Centroids'], errorAmount = assignPositionOfCentroids(copyOfData, centroids)
    centroids = copyOfData.groupby('Number of Centroids').agg('mean').reset_index(drop = True)
    return copyOfData['centroid'], errorAmount, centroids

# FUNCTIONS END 
# MAIN START 
# VARIABLE DECLARATION
# Create a random seed so we can randomly sample better and grab the same results everytime
randSeed = np.random.seed(42)
numOfCentroids = 5
dataColors = ListedColormap(["purple", "mediumblue", "darkmagenta", "slategray", "midnightblue"])
centroidColors = ListedColormap(["green", "lime", "red", "gold", "cyan"])

# get all data needed, and slice it
alldata = pd.read_csv('romanianData.csv')
alldata["Years Since Publishing"] = alldata['Year'].apply(YearsSincePublish)

dataSelection = alldata[['Years Since Publishing','Cites']].copy()

#Create the centroids temp value
centroids = createCenterOfData(5, dataSelection)
print("Successfully sliced data. Beginning analysis now.")

#Run algorithm
dataSelection['Number of Centroids'], dataSelection['Error Amount'], centroids =  adjustAlgorithm(dataSelection[['Years Since Publishing','Cites']], 5)
print(dataSelection.head())

#MAIN END
# GRAPHING START
print("Generating graphs... Please be patient, this might take a minute!")

# Create the graph and assign the colors
figure, axises = plt.subplots(figsize=(15, 6))

# using the subplot function so we can super-impose the centroids on top of the data for easier viewing 
# using iloc to make sure the data is in integer form
plt.scatter(dataSelection.iloc[:,0], dataSelection.iloc[:,1],  marker = 'o', 
            c=dataSelection['centroidNumber'].astype('category'), 
            cmap = dataColors, s=80, alpha=0.5)
plt.scatter(centroids.iloc[:,0], centroids.iloc[:,1],  
            marker = 's', s=200, c=[0, 1, 2,3,4], 
            cmap = centroidColors)

# set aesthetic labels
axises.set_xlabel('Years Since Publishing', fontsize=15)
axises.set_ylabel('Citations', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()