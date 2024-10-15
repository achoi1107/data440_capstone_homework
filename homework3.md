# HW3 - Regularization and Variable Selection
### Andrew Choi
### DATA 440 Capstone Projects

# Part 1: 
# Create your own PyTorch class that implements the method of SCAD regularization and variable selection (smoothly clipped absolute deviations) for linear models. Your development should be based on the following references:
- https://andrewcharlesjones.github.io/journal/scad.html
- https://www.jstor.org/stable/27640214?seq=1
# Test your method on a real data set and determine a variable selection based on features' importance, according to SCAD.

## Defining the Libraries and Kernels


The screenshots below show the classes for scaling data as well as the proper device used:


<img src="./images3/scaler1.png" width="300"/>


<img src="./images3/scaler2.png" width="700"/>


<img src="./images3/scaler3.png" width="700"/>


Importing the proper device here allows for the calculations for this project to be done using the CPU and sets the proper data type to be a 64-bit floating-point number. Having these statements at the beginning of the workflow allows for proper formatting of what we want to use and run the rest of the project on. Moving forward with the two scaler classes, both of these classes are designed to help fit a scaling model to the data and transform that data based on the new scaled parameters. The StandardScaler class helps with standardizing the data to have a mean of 0 and a standard deviation of 1. After this class initializes its mean and standard deviation values, it transforms the inputted data using the mean and standard deviation that was calculated. This class at the end returns a tensor of the scaled data. Next with the MinMaxScaler class, this is different from the StandardScaler in that it scales the data based on a range of 0 to 1. So after initializing the min and max values, this scaler transforms the inputted data using the calulated minimum and maximum values along with its dimensions and stores these values. Similarily, this scaling class also returns a tensor of the scaled data. These classes are important as they help with the idea of scaling where having all of the features present contribute equally to the calculations. This is quite valuable when the input data have different units or range where scaling is quite important to use. 

## Defining the regularization classes


The screenshots below show the code for the regularization classes.


<img src="./images3/regular1.png" width="600"/>


<img src="./images3/regular2.png" width="600"/>


<img src="./images3/regular3.png" width="600"/>


<img src="./images3/regular4.png" width="600"/>


<img src="./images3/regular5.png" width="900"/>


The code here defines the three PyTorch classes of various types of regression models that are trying to regularize the data. The first one presented is the ElasticNet class. This class here implements a linear search where it combines both L1 (lasso) and L2 (ridge) penalties. If we were to go through the class, after initializing its constructors, there is a forward method function that performs a forward pass where it helps with predicting outputs based on input data. Next is the loss function which computes the loss by combining the MSE with L1 and L2 penalities. There is then the fit method that optimizes the model's parameters over multiple epochs using the Adam optimizer. Finally there are the predict and get coefficient methods where it helps perform the predictions and then returns the value of the weights respectively. The next class after that was defined is the Squareroot lasso class. This is a little different from ElasticNet where the SqrtLasso class only uses the L1 penalty to help with calculating penalty. So instead of combining the two regularization terms, it uses the one where it ultimately promotes feature selection by driving less important weights to zero. The final one pertains to question one where we created a SCADModel class. This class uses the idea of SCAD regularization, which aims to try and reduce the bias of L1 regularization as the SCAD here imposes a penalty that gradually reduces for larger coefficients. In this class we incorporate the scad_penalty function that calculates the SCAD regularization for the models coefficients. 

## Declaring the data and training


<img src="./images3/data1.png" width="900"/>


<img src="./images3/data2.png" width="900"/>


The screenshots here load the proper concrete dataset here as we plan to test our methods on this dataset. After loading the dataset, we prepare the target and features where the x variable represents the feature variables besides strength, while the y variable holds the strength which is the target variable. Following the declaration of the target and features, the target and features and converted to numpy arrays in the case of the next part where that is needed for training and eventually model fitting. 


## Visalizing the correlated features and determining proper variable selection


<img src="./images3/visual1.png" width="500"/>


<img src="./images3/visual2.png" width="1500"/>


<img src="./images3/visual3.png" width="500"/>


<img src="./images3/visual4.png" width="500"/>










