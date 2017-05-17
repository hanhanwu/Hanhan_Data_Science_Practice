library(qicharts)

set.seed(7799)
y <- rnorm(24)   # create 24 random numbers
y
qic(y, chart = 'i')  # individuals chart
## Center line of the control chart represent the weighted mean, rather than median
## The other 2 lines represent the upper ad lower control limits

y[9] <- 7  # introduce an outlier
qic(y, chart = 'i')


## Simulate hospital data
# Setup parameters
m.beds       <- 300
m.stay       <- 4
m.days       <- m.beds * 7
m.discharges <- m.days / m.stay
p.pu         <- 0.08

# Simulate data
discharges  <- rpois(24, lambda = m.discharges)
patient_days <- round(rnorm(24, mean = m.days, sd = 100))
n.pu        <- rpois(24, lambda = m.discharges * p.pu * 1.5)  # counting defects, e.g. number of pressure ulcers, it should reflect Poisson Destribution
n.pat.pu    <- rbinom(24, size = discharges, prob = p.pu)     # counting defectives, e.g. number of patient with one or more pressure ulcers, it should reflect binomial distribution
week        <- seq(as.Date('2014-1-1'),
                   length.out = 24, 
                   by         = 'week') 

# Combine data into a data frame
d <- data.frame(week, discharges, patient_days,n.pu, n.pat.pu)
d


## C Chart - based on Poisson Distribution
## since n.pu should reflect Poisson distribution, the Count in this C chart is n.pu
qic(n.pu,
    x     = week,
    data  = d,
    chart = 'c',
    main  = 'Hospital acquired pressure ulcers (C chart)',
    ylab  = 'Count',
    xlab  = 'Week')


## U Chart - plot rate
qic(n.pu, 
    n        = patientdays,
    x        = week,
    data     = d,
    chart    = 'u',
    multiply = 1000,
    main     = 'Hospital acquired pressure ulcers (U chart)',
    ylab     = 'Count per 1000 patient days',
    xlab     = 'Week')


## P chart - plot proportion/percentage
qic(n.pat.pu,
    n        = discharges,
    x        = week,
    data     = d,
    chart    = 'p',
    multiply = 100,
    main     = 'Hospital acquired pressure ulcers (P chart)',
    ylab     = 'Percent patients',
    xlab     = 'Week')


## prime control chart - use when control limits for U, P charts are too narrow
qic(n.pat.pu, discharges, week, d,
    chart    = 'p',
    multiply = 100,
    main     = 'Prime P chart of patients with pressure ulcer',
    ylab     = 'Percent',
    xlab     = 'Week',
    prime    = TRUE)


## G Chart - geometric distribution, displays the number of cases between events
# Create vector of random values that have a geometric distribution
d <- c(NA, rgeom(100, 0.08))
d

qic(d,
    chart = 'g',
    main  = 'Patients between pressure ulcers (G chart)',
    ylab  = 'Count',
    xlab  = 'Discharge no.')


## T Chart - displays the time between events
dates  <- seq(as.Date('2017-1-1'), as.Date('2017-12-31'), by = 'day')
events <- sort(sample(dates, 100))
events
d <- c(NA, diff(events))
d

qic(d,
    chart = 't',
    main  = 'Days between pressure ulcers (T chart)',
    ylab  = 'Days',
    xlab  = 'Pressure ulcer no.')


## I and MR Charts - individual measures (I think it means individual feature)
# I chart is often accompained with MR chart, which measures the moving range (absolute difference between neughboring data)
# Vector of birth weights from 100 babies
y <- round(rnorm(100, mean = 3400, sd = 400))
y

qic(y,
    chart = 'i',
    main  = 'Birth weight (I chart)',
    ylab  = 'Grams',
    xlab  = 'Baby no.')
qic(y,
    chart = 'mr',
    main  = 'Pairwise differences in birth weights (MR chart)',
    ylab  = 'Grams',
    xlab  = 'Baby no.')


## Xbar and S chart
# Xbar - shows the average of a column
# S chart - shows the standard deviation of a column
# Vector of 100 subgroup sizes (average = 12)
sizes <- rpois(100, 12)
sizes

# Vector of dates identifying subgroups
date <- seq(as.Date('2017-1-1'), length.out = 100, by = 'day')
date <- rep(date, sizes)
date

# Vector of birth weights
y <- round(rnorm(sum(sizes), 3400, 400))
y

# Data frame of birth weights and dates
d <- data.frame(y, date)
d

qic(y, 
    x     = date, 
    data  = d,
    chart = 'xbar',
    main  = 'Average birth weight (Xbar chart)',
    ylab  = 'Grams',
    xlab  = 'Date')
qic(y, 
    x = date, 
    data = d,
    chart = 's',
    main = 'Standard deviation of birth weight (S chart)',
    ylab = 'Grams',
    xlab = 'Date')


## standardized a control chart - creates a standardised control chart, where points are plotted in standard deviation units along with a center line at zero and control limits at 3 and -3. Only relevant for P, U and Xbar charts.
qic(y, 
    x     = date, 
    data  = d,
    chart = 'xbar',
    standardised = T,
    main  = 'Average birth weight (Xbar chart)',
    ylab  = 'Grams',
    xlab  = 'Date')
