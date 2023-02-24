# MAFAT-RSSI-Challenge (2022)
In this challenge the competitors were asked to detect and count occupants within rooms using WiFi signals (based on the mesured RSSI strength along time).  
My solution was derived from the observation that an increase in the number of occupants
in a room is likely to create stronger multipath effects, resulting in a greater variation
in power difference between the two antennas, especially if the people are in motion.
As shown in the figure below, the probability of a stronger power variance is much higher
for a room with three occupants (black label) compared to an empty room (blue label)

![CDF plot](https://github.com/zahilaty/MAFAT-RSSI-Challenge/blob/master/Figures/ExampleForMPImportance.png)

"This particular feature was among the inputs used to train ResNet50, alongside the average
power and other relevant data. The network was trained to predict the probability of each class,
which corresponds to the number of occupants in a given space, using L1 loss as the evaluation
criterion. This approach was necessary because we were dealing with an "ordered loss" scenario,
even though the predictions were based on classification rather than regression.  
A more detailed description will be provided at a later time