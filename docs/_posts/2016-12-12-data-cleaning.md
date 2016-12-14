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

After identifying the datasets, we wrote Python code to download and collate parts of the data. We wrote one
script, available here, that would pull all of the inspections in a date range, identify all of the businesses
that were inspected in that range, find all of the licenses corresponding to that business (since an inspection
is related only to one license, but each business might have several), and then hot-encoded all of the categorical
variables.

We also wrote a script that retrieved crime and sanitation data and built a grid into which to bucket counts 
of crime and sanitation events. We made this decision for computational resource reasons; while ideally we would
have a set a radius and associated every inspection with the normalized number of crimes and sanitation complaints
within that radius, this was computationally infeasible given the number of inspections, crime reports, and sanitation
complaints in the date range of interest. As a result, we built a grid based on the range of latitude and longitudes, 
associated every crime and sanitation complaint with one rectangle on the grid, building counts in each rectangle on
the map. We then associated each inspection with its nearest rectangle center's crime and sanitation count. While
this system is imperfect, it was clear from visualization (see next page) that it was a useful proxy for 
the socioeconomic picture of a certain neighborhood. 

After completing the data cleaning process, a single observation in the dataset looked like this: 

The full list of variables that we built was: 

|**Variable Name**|**Meaning**|
Food Inspections | [City of Chicago Data Portal](https://data.cityofchicago.org/Health-Human-Services/Food-Inspections/4ijn-s7e5) | 
Business Licenses | [City of Chicago Data Portal](https://data.cityofchicago.org/Community-Economic-Development/Business-Licenses/r5kz-chrr) |
Crime Reports | [City of Chicago Data Portal](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2)
311 Service Requests -- Sanitation Code Complaints | [City of Chicago Data Portal](https://data.cityofchicago.org/Service-Requests/311-Service-Requests-Sanitation-Code-Complaints/me59-5fac)
Weather | [NOAA's Daily Summaries](https://www.ncdc.noaa.gov/cdo-web/datasets)


