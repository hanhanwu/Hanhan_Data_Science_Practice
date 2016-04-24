# WITH TIME SERIES OBJECT & CORRELATE ANALYSIS

cadairydata <- maml.mapInputPort(1)

# Create a new column as a POSIXct object
Sys.setenv(TZ = "PST8PDT")
cadairydata$Time <- as.POSIXct(strptime(paste(as.character(cadairydata$Year), "-", as.character(cadairydata$Month.Number), "-01 00:00:00", sep = ""), "%Y-%m-%d %H:%M:%S"))

# correlate analysis
ts.detrend <- function(ts, Time, min.length = 3){
  ## Function to detrend and standardize a time series. 
  
  ## Define some messages if they are NULL.  
  messages <- c('ERROR: ts.detrend requires arguments ts and Time to have the same length',
                'ERROR: ts.detrend requires argument ts to be of type numeric',
                paste('WARNING: ts.detrend has encountered a time series with length less than', as.character(min.length)),
                'ERROR: ts.detrend has encountered a Time argument not of class POSIXct',               
                'ERROR: Detrend regression has failed in ts.detrend',
                'ERROR: Exception occurred in ts.detrend while standardizing time series in function ts.detrend'
  )
  # Create a vector of zeros to return as a default in some cases.
  zerovec  <- rep(length(ts), 0.0)
  
  # The input arguments are not of the same length, return ts and quit.   
  if(length(Time) != length(ts)) {warning(messages[1]); return(ts)}
  
  # If the ts is not numeric, just return a zero vector and quit.   
  if(!is.numeric(ts)) {warning(messages[2]); return(zerovec)}
  
  # If the ts is too short, just return it and quit.   
  if((ts.length <- length(ts)) < min.length) {warning(messages[3]); return(ts)}
  
  ## Check that the Time variable is of class POSIXct.
  if(class(cadairydata$Time)[[1]] != "POSIXct") {warning(messages[4]); return(ts)}
  
  ## Detrent the time series using a linear model.
  ts.frame  <- data.frame(ts = ts, Time = Time)
  tryCatch({ts <- ts - fitted(lm(ts ~ Time, data = ts.frame))},
           error = function(e){warning(messages[5]); zerovec})
  
  tryCatch( {stdev <- sqrt(sum((ts - mean(ts))^2))/(ts.length - 1)
             ts <- ts/stdev}, 
            error = function(e){warning(messages[6]); zerovec}) 
  
  ts
}

# Apply the detrend.ts function to the variables of interest.
df.detrend <- data.frame(lapply(cadairydata[, 4:7], ts.detrend, cadairydata, cadairydata$Time))

# generate the pairwise scatterplot matrix
pairs(~ Cotagecheese.Prod + Icecream.Prod + Milk.Prod + N.CA.Fat.Price, data = df.detrend, main = "Pairwise Scatterplots of detrended standardized time series")


# A function to compute pairwise correlations from a 
## list of time series value vectors.
pair.cor <- function(pair.ind, ts.list, lag.max = 1, plot = FALSE){
  ccf(ts.list[[pair.ind[1]]], ts.list[[pair.ind[2]]], lag.max = lag.max, plot = plot)
}

## A list of the pairwaise indices.
corpairs <- list(c(1,2), c(1,3), c(1,4), c(2,3), c(2,4), c(3,4))

## Compute the list of ccf objects.
cadairycorrelations <- lapply(corpairs, pair.cor, df.detrend)  

## None of these correlation values is large enough to be significant. We can therefore conclude that we can model each variable independently.
cadairycorrelations

df.correlations <- data.frame(do.call(rbind, lapply(cadairycorrelations, '[[', 1)))

c.names <- c("correlation pair", "-1 lag", "0 lag", "+1 lag")
r.names  <- c("Corr Cot Cheese - Ice Cream",
              "Corr Cot Cheese - Milk Prod",
              "Corr Cot Cheese - Fat Price",
              "Corr Ice Cream - Mik Prod",
              "Corr Ice Cream - Fat Price",
              "Corr Milk Prod - Fat Price")

## Build a dataframe with the row names column and the
## correlation data frame and assign the column names
outframe <- cbind(r.names, df.correlations)
colnames(outframe) <- c.names
outframe
maml.mapOutputPort('outframe')  
