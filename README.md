# Logistic_Regression
Repo having easy to medium Logistic Regression models


The equation of Linear Regression is 
``` y' = wx + b ```

In <strong> Logistic Regression </strong> the idea is to add a probabilities to the model

``` 
            1      
y` = ------------------
      ( 1 + e ^ (-wx+b) )

```


### Error Calculation Method :
Cross Entropy 


## STEPS
> TRAINING
> - Initialize weights as zero
> - Initialize bias as zero


> GIVEN DATA POINT: 
> - Predict result y\`
> - Calculate error
> - Use gradient descent to figure out new weight and bias values
> - Repeat n times

> TESTING: 
> - Put values from the data point into equation 
> - Choose the label based on probability
