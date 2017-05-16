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


## G Chart
# Create vector of random values from a geometric distribution
d <- c(NA, rgeom(23, 0.08))
d
qic(d,
    chart = 'g',
    main  = 'Patients between pressure ulcers (G chart)',
    ylab  = 'Count',
    xlab  = 'Discharge no.')


# TO BE CONTINUED...
