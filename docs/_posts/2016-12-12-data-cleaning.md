---
layout: post
title:  "Data Collection and Cleaning"
published: true
page: 2
---

We began the project by deciding what data to use and where it could be sourced.
We identified the following data sets and their sources: 

|**Dataset**|**Source**|
Food Inspections | [City of Chicago Data Portal](https://data.cityofchicago.org/Health-Human-Services/Food-Inspections/4ijn-s7e5) | 
Business Licenses | [City of Chicago Data Portal](https://data.cityofchicago.org/Community-Economic-Development/Business-Licenses/r5kz-chrr) |
Crime Reports | [City of Chicago Data Portal](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2)
311 Service Requests -- Sanitation Code Complaints | [City of Chicago Data Portal](https://data.cityofchicago.org/Service-Requests/311-Service-Requests-Sanitation-Code-Complaints/me59-5fac)
Weather | [NOAA's Daily Summaries](https://www.ncdc.noaa.gov/cdo-web/datasets)

## Building the Dataset

To begin the data collection process, we collected all of the food inspections available
between Jan 4 2010 and Dec 2 2016. These
records formed the core of the dataset, and we used the businesses indicated
as inspected as the input to the next step.

The next step was to take the business referenced by its City of Chicago
license number and associate it with all of the licenses registered to
that business. In the original food inspection dataset,
*inspection* is related to exactly *one* license that a business has,
such as food preparation. However, previous research by the City
of Chicago found that other licenses held by a business, like 
tobacco or liquor sales, could also be predictive of inspection
outcomes. To build in this information, we queried the Business
Licenses dataset using the license number listed in the Food Inspections
dataset. We then found the account number associated with that license,
and then all of the *licenses* associated with that account number.
This gave us all of the relevant licenses for a business. After grouping
by account number and one-hot encoding all of the business license types,
the multiple licenses could be flattened into one row per business including
indicators for the types of licenses held. 

After having a complete picture of all of the businesses for which 
data was available (more on this later), we derived features about
the inspection history of each business based on the inspections
in the past that were present for each business. We
used some basic text processing on the comments
left by inspectors along with results of previous
inspections to derive the following variables:

|Variable|Meaning|
|`previous_count`| Number of previous inspections
|`previous_fraction` | Fraction of previous failures
|`previous_result` | Result of the previous inspection
|`license_age` | Time since license issue.
|`time_since_last_insp` | Time since previous inspection.
|`previous_violations` | Number of previous violations.
|`previous_citations`| Number of previous citations.
|`previous_critical` | Indicator for previous critical violations.
|`previous_serious`| Indicator for previous serious violation.
|`previous_minor`| Indicator for previous minor violation.
|`previous_corrected`| Indicator for previous collected violations. 

The data at this stage is available in [`data_built_features.csv`](https://www.dropbox.com/s/v791vnsrit050nv/data_built_features.csv?dl=0) on Dropbox. The work for this stage is available
in [`I_ChicagoFoodInspection_CS109a_Build_Features.ipynb`](https://github.com/Fggw/foodinspections/blob/master/I_ChicagoFoodInspection_CS109a_Build_Features.ipynb).

## Adding Crime and Sanitation Data

After building the features to describe each
inspection's business history, we aimed to
build a proxy for the socioeconomic picture
in the nearby area for each business. 

To develop this proxy, we decided to use
geographic count of crime and sanitation complaint
incident reports, the idea being that more
unsafe and less clean neighborhoods might help
predict inspection failures. Ideally, we would
associate every business with the count of 
crimes and sanitation reports in a fixed radius 
surrounding it, treating that radius as a 
hyperparameter that would be tuned in cross-validation.
The size of the datasets made this computationally
infeasible given the time constraints, however, so
we adopted a simplifying strategy. We built
a grid of "buckets" to divide the city and then
associated every crime and sanitation report
with the bucket that it fell into; for computational
simplicity, the "nearness" was calculated using 
rectangular boundaries, rather than circular radii, 
so that the appropriate bucket could be quickly
identified. We then took each bucket's count
of events for both crime and sanitation
and normalized by dividing by the total
number of observations of each type of 
report, so that the variables fell between
0 and 1. This resulted in a grid that looked
like this:

![Crime-Sanit-Map]({{site.baseurl}}/images/crime_sanit_map.png)

The code that collected this data, built the grid,
and then built the heatmaps is available in 
[`II_ChicagoFoodInspection_CS109a_Crime+Sanit Clean.ipynb`](https://github.com/Fggw/foodinspections/blob/master/II_ChicagoFoodInspection_CS109a_Crime%2BSanit%20Clean.ipynb).

Each inspection row, which includes a location
at which the inspection took place, was then associated
with the nearest bucket on the grid. After that
was done, the dataset had all of the features that
we intended to work with. The next step that we 
performed was imputing missing values. We 
discuss this process on the following page.





