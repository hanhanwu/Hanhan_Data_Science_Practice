path<- "[YOUR PATH OF THE FOLDER]"  # change to your path of the folder!!
setwd(path)


# seasonal subseries plot
## columns from left to right: CO2, Year&Month, Year, Month
## To download the .txt file: https://www.itl.nist.gov/div898/handbook/datasets/MLCO2MON.DAT
fname = "MLCO2MON.txt"
m = matrix(scan(fname,skip=25),ncol=4,byrow=T)
print(dim(m))  # 161 rows, 4 columns
print(min(m[,2]))  # 1974.38
print(max(m[,2]))  # 1987.71
head(m)

## perform linear fir to detrend the data
# fit <- lm(m[,1]~m[,3])
fit <- lm(m[,1]~m[,2])
fit

## save residuals for each month
q <- ts(fit$residuals, start=c(1974, 5), frequency = 12)
q

## generate the seasonal subseries plot
par(mfrow=c(1,1))
monthplot(q, phase = cycle(q), base=mean, ylab = "CO2 Concentraions",
          main="Seasonal Subseries Plot of CO2 Concentrations", xlab="Month",
          labels = c("Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"))
