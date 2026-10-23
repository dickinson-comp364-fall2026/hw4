# Weather Station Latitude Prediction

## Objective

In this assignment, the objective is to use temperature data from weather stations in the United States to determine the location—or, more specifically, just the latitude—of a weather station from its temperatures alone. We use real data downloaded from the National Oceanic and Atmospheric Administration (NOAA).

The initial configuration of the assignment repository uses data from only 300 weather stations. However, you can increase this to 5,000 weather stations once your code and training are working satisfactorily.

## Dataset

Each row of data consists of an identifier for a particular weather station, the station's latitude, and 365 daily maximum temperatures—one high temperature for each day of the year, in order from January 1 through December 31. The latitude is the target variable that we will try to predict.

For every weather station, we have data spanning 2010 through 2024. We omit every leap year in its entirety to keep the dataset simple and ensure that every annual record has the same length of 365 temperatures.

## Training and Validation Split

This dataset requires an unusual approach to the split between training and validation data. Recall that each row comes from a specific weather station and that we have approximately a dozen rows for each station. If we randomly split the individual rows into training and validation sets, most weather stations will have some rows in each set. This would make the task too easy, because we expect the pattern of temperatures at exactly the same location to be similar from year to year.

Our goal is to predict the latitude of a weather station for which the model has never seen any data. Therefore, we first randomly assign each weather station to either the training set or the validation set. We then assign every row to the same set as its weather station. Thus, no weather station represented in the validation set is represented in the training set.

## Regression and Loss Function

Unlike the previous neural network assignment, we are not trying to classify the instances in the dataset. Instead, we are trying to predict a number that can vary continuously: the latitude. In other words, this is a **regression** task, not a classification task. Therefore, we use a mean squared error loss function rather than the cross-entropy loss functions used in previous assignments.

## One-Dimensional Convolutional Neural Network

Because the input for an instance consists of a sequence of 365 numbers with a meaningful ordering, it would typically be described as a **time series**. This is a good application for a one-dimensional convolutional neural network (1D conv net). The starter code provides the framework for creating this 1D conv net in PyTorch and training it.


## The task

To complete the assignment, add and/or alter code at the locations of the string `TODO` in the source file `one_d_conv_net.py`.
