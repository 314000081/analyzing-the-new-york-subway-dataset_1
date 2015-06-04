# -*- coding: utf-8 -*-
#This program compute the coefficients theta and R^2 value for the linear model using Gradient Descent
"""

@author: 314000081 ( Abdulrazak Mulla)
"""

import numpy as np
import pandas
#code commented
#from ggplot import *



def normalize_features(array):
   """
   Normalize the features in the data set.
   """
   mu = array.mean()
   sigma = array.std()
   array_normalized = (array - mu)/sigma

   return array_normalized, mu, sigma

def compute_cost(features, values, theta):
    """
    Compute the cost function given a set of features / values, 
    and the values for our thetas.
    
    This can be the same code as the compute_cost function in the lesson #3 exercises,
    but feel free to implement your own.
    """
    
    # your code here
    m=len(values)
    sum_of_square_errors=np.square(np.dot(features,theta)-values).sum()
    cost=sum_of_square_errors/(2*m)
    #print str(theta) +"->"+ str(cost)
    
   
    return cost

def gradient_descent(features, values, theta, alpha, num_iterations):
    """
    Perform gradient descent given a data set with an arbitrary number of features.
    
    This can be the same gradient descent code as in the lesson #3 exercises,
    but feel free to implement your own.
    """
    
    m = len(values)
    cost_history = []

    for i in range(num_iterations):
        # your code here
        predicted_values=np.dot(features,theta)
        SST=((values-np.mean(values))**2).sum()
        SSReg=((values-predicted_values)**2).sum()
        r_squared=1-(SSReg/SST)
        print "theta:"
        print theta
        print "R^2:"
        print r_squared
        # Calculate R^2 ends
        theta= theta-alpha/m*np.dot((predicted_values-values),features)
        cost=compute_cost(features, values, theta)
        cost_history.append(cost)
    return theta, pandas.Series(cost_history)

def predictions():
     
    
    filename='C:\\ct\\turnstile_weather_v2.csv'
        
    dataframe=pandas.read_csv(filename)
    
    # Select Features (try different features!)
    #features = dataframe[['rain', 'fog', 'precipi']]
    #features = dataframe[['fog', 'hour','meantempi','precipi']]
    features = dataframe[['fog', 'hour','meantempi','precipi','rain']]
    
    # Add UNIT to features using dummy variables
    dummy_units = pandas.get_dummies(dataframe['UNIT'], prefix='unit')
    dummy_weekdays= pandas.get_dummies(dataframe['day_week'], prefix='day')
    dummy_conds = pandas.get_dummies(dataframe['conds'], prefix='conds')
    dummy_Time = pandas.get_dummies(dataframe['TIMEn'], prefix='time')
    
    #print dummy_units
  
    features = features.join(dummy_units)
    features = features.join(dummy_weekdays)
    features = features.join(dummy_conds)
    features = features.join(dummy_Time)

    
    # Values
    values = dataframe['ENTRIESn_hourly']
    m = len(values)

    features, mu, sigma = normalize_features(features)
    features['ones'] = np.ones(m) # Add a column of 1s (y intercept)
    
    # Convert features and values to numpy arrays
    features_array = np.array(features)
    values_array = np.array(values)
    # Set values for alpha, number of iterations.
    alpha = 0.1 # please feel free to change this value
    num_iterations =100 # please feel free to change this value

    # Initialize theta, perform gradient descent
    theta_gradient_descent = np.zeros(len(features.columns))
    theta_gradient_descent, cost_history = gradient_descent(features_array, 
                                                            values_array, 
                                                            theta_gradient_descent, 
                                                            alpha, 
                                                            num_iterations)
    
    plot = None
    # -------------------------------------------------
    # Uncomment the next line to see your cost history
    # -------------------------------------------------
    plot = plot_cost_history(alpha, cost_history)
    # 
    # Please note, there is a possibility that plotting
    # this in addition to your calculation will exceed 
    # the 30 second limit on the compute servers.
    
    predictions = np.dot(features_array, theta_gradient_descent)
    #print cost_history
    return predictions, plot


def plot_cost_history(alpha, cost_history):
   """This function is for viewing the plot of your cost history.
   You can run it by uncommenting this
   """ 
   #plot_cost_history(alpha, cost_history) 
   """
   call in predictions.
   
   If you want to run this locally, you should print the return value
   from this function.
   """
   cost_df = pandas.DataFrame({
      'Cost_History': cost_history,
      'Iteration': range(len(cost_history))
   })
   #Anu's code commented
   #print ggplot(cost_df, aes('Iteration', 'Cost_History')) + \
    #  geom_point() + ggtitle('Cost History for alpha = %.3f' % alpha )

if __name__=='__main__':
    predictions() 

