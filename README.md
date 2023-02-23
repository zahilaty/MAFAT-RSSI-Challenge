# MAFAT-RSSI-Challenge (2022)
In this challenge the competitors were asked to detect and count occupants within rooms using WiFi signals (based on the mesured RSSI strength along time).  
My solution was derived from the observation that an increase in the number of occupants
in a room is likely to create stronger multipath effects, resulting in a greater variation
in power difference between the two antennas, especially if the people are in motion.  
As shown in the figure below, the probability of a stronger power variance is much higher
for a room with three occupants (blue label) compared to an empty room (black label)

![CDF plot](https://github.com/zahilaty/MAFAT-RSSI-Challenge/blob/master/Figures/ExampleForMPImportance.png)

This feature was one of the features that were input into ResNet50.  
A more detailed description will be provided at a later time