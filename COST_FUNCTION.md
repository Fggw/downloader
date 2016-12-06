# Defining a Cost Function

When we considered how to appropriately score 
the models that we build for this project, we initially considered
testing setting the score as the accuracy of predicted inspection results
against a test set of observed inspections. Because we're dealing in this
case with time series data, we knew that our model selection process
had to be considerate of time: in other words, we couldn't test 
on data that occurred before the latest occurring data-point that
occurred in the pool of data used to the train the model. 

This got us as far as doing a train-test split on a training set. 
We might have stopped here and simply selected classification
performance in the last two months as our scoring quantity of interest. 

That, however, doesn't quite match up with the performance incentives 
of a system like this. The goal from the city of Chicago's perspective
is to maximize the number of inspections that result in discovering 
violations; one would reasonably argue, we believe, that inspections
that result in no violations are a waste of an inspector's time when
there are violations that might be discovered elsewhere. Inspections
are a constrained resource, so what we should really be interested 
is is not only whether or not the system can correctly identify 
when an establishment is likely to fail an inspection, but also
that the system correctly *tells* the inspectors that an establishment
should be an inspection priority. In other words, the model should
output a ordering ranking the possible inspections in order of 
failure probability, and score the quality of that ranking with
respect to the observations of the inspection outcomes.
We want to reward passed inspections that are put lower in the 
ranking and failed inspections that are put higher in the ranking, 
and penalize the opposite case for each of these things. 

This system is subtly different than accuracy scoring, because 
the score will rely not only on the predicted value, but in fact
the estimated probability of each outcome in comparison to the observed actual
value.


