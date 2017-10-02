library(mlr)
library(data.table)
library(dplyr)
library(ggplot2)
library(plotly)
library(Hmisc)

path <- "[path]"
setwd(path)

work_order_info <- fread("Maintenance/Work_order_information.csv", na.strings = c("", " ", "?", "NULL", NA), stringsAsFactors = F)
oil_analysis_data <- fread("Suplemental/OilAnalysisDataDec2016-Aug2017.csv", na.strings = c("","-", " ", "?", "NULL", NA), stringsAsFactors = F)
exhaust_gas_data <- fread("MineCare/TruckMachineData.csv", na.strings = c("","-", " ", "?", "NULL", NA), stringsAsFactors = F)

fact_distribution_plot <- function(a){
  counts <- table(a)
  barplot(counts)
}

num_distribution_plot <- function(a, q){
  ggplot(data = q, aes(x= a, y=..density..)) + geom_histogram(fill="blue",color="red",alpha = 0.5,bins =100) + geom_density()
  ggplotly()
}

impute_NA <- function(x) {
  if(is.numeric(x) == T) {
    x[is.na(x)] <- median(x[which(!is.na(x))])
  }
  else if(is.factor(x) == T) {
    x[is.na(x)] <- names(sort(summary(x), decreasing = T))[1]
  }
  return(x)
}

data_scaling <- function(x){(x-min(x))/(max(x)-min(x))}

############################################### REMOVE COLUMNS ############################################
colnames(work_order_info)
remove_lst <- c(2,4,5,6,8,9,10,11,14,15,16,17,18,19,20,23,25,26,27)
work_order_info <- work_order_info[, -remove_lst, with=F]
summary(as.factor(work_order_info$compid))

colnames(oil_analysis_data)
remove_lst <- c(2,3,4,5,6,10,13,20)
oil_analysis_data <- oil_analysis_data[, -remove_lst, with=F]
summary(as.factor(oil_analysis_data$EquipmentNumber))
oil_analysis_data <- oil_analysis_data[EquipmentNumber != '8/1/17 UNK MARIGOLD' & EquipmentNumber != 'BULKOIL']
summary(as.factor(oil_analysis_data$EquipmentNumber))
head(oil_analysis_data)

colnames(exhaust_gas_data)
remove_lst <- c(1,2,4,8)
exhaust_gas_data <- exhaust_gas_data[, -remove_lst, with=F]
colnames(exhaust_gas_data)
head(exhaust_gas_data)
summary(as.factor(exhaust_gas_data$OEMParameterName))

# # export the data, I need to match the time in Pyhton
# write.csv(work_order_info, file = "work_order_info.csv")
# write.csv(oil_analysis_data, file = "oil_analysis_data.csv")
# write.csv(exhaust_gas_data, file = "exhaust_gas_data.csv")

# read matched_date
matched_time <- fread("matched_time.csv", na.strings = c("","-", " ", "?", "NULL", NA), stringsAsFactors = F)

############################################# DATE PREPROCESSING #############################################
# label preprocessing
work_order_info$ACCTDESC <- toupper(work_order_info$ACCTDESC)
work_order_info$ACCTDESC[which(is.na(work_order_info$ACCTDESC)==T)] <- 'MISSING'
levels(as.factor(work_order_info$ACCTDESC))
sort(summary(as.factor(work_order_info$ACCTDESC)))

levels(as.factor(work_order_info$PROBTYPE))
sort(summary(as.factor(work_order_info$ACCTDESC[which(work_order_info$PROBTYPE=='Breakdown')])))

# label 1 - 24V MECHANICAL
target_label_set <- work_order_info[which(work_order_info$ACCTDESC == '24V MECHANICAL' & work_order_info$PROBTYPE == 'Breakdown'),]
dim(target_label_set)
other_label_set <- work_order_info[which(work_order_info$ACCTDESC != '24V MECHANICAL' & work_order_info$PROBTYPE == 'Breakdown'),]
dim(other_label_set)
other_label_set$ACCTDESC <- 'Other'
levels(as.factor(other_label_set$ACCTDESC))
dm_data1 <- rbind(target_label_set, other_label_set)
dim(dm_data1)  # 4113    7
summary(as.factor(dm_data1$ACCTDESC))  # 871           3242

## link to other tables
colnames(dm_data1)
dim(dm_data1)  # (4113, 8)
head(dm_data1)
colnames(matched_time)
head(matched_time)
colnames(oil_analysis_data)
head(oil_analysis_data)
dm_data1 <- inner_join(dm_data1, matched_time, by = c("compid" = "compid", "DATE_WO" = "work_order_date"))
dm_data1$compid <- as.character(as.factor(dm_data1$compid))
head(dm_data1$compid)
dm_data1 <- inner_join(dm_data1, oil_analysis_data, by = c("compid" = "EquipmentNumber", "oil_sample_date" = "sampledate"))
dim(dm_data1)  # (10317, 90)
head(dm_data1)
summary(as.factor(dm_data1$ACCTDESC))  # 2006, 8311 (imbalanced)

data_backup <- data.frame(dm_data1)
## data explore & data preprocessing
summarizeColumns(dm_data1)
colnames(dm_data1)
dm_data1 <- data.table(dm_data1)
remove_lst <- c(1,2,3,4)
dm_data1 <- dm_data1[, -remove_lst, with=F]
summarizeColumns(dm_data1)

levels(as.factor(dm_data1$ACCTDESC))
dm_data1$ACCTDESC <- as.factor(dm_data1$ACCTDESC)

summary(as.factor(dm_data1$DEPARTMENT))
dm_data1$DEPARTMENT[which(is.na(dm_data1$DEPARTMENT)==T)] <- 'MISSING'
dm_data1$DEPARTMENT <- as.factor(dm_data1$DEPARTMENT)
summary(dm_data1$DEPARTMENT)

dm_data1$PRIORITY <- as.factor(dm_data1$PRIORITY)
summary(dm_data1$PRIORITY)

num_distribution_plot(dm_data1$ACTUAL, dm_data1)
quantile(dm_data1$ACTUAL)
num_distribution_plot(sqrt(dm_data1$ACTUAL), dm_data1)
boxplot(dm_data1$ACTUAL)
boxplot(sqrt(dm_data1$ACTUAL))  # can convert as factor

dm_data1$oil_sample_date <- as.factor(dm_data1$oil_sample_date)
summary(dm_data1$oil_sample_date)

head(dm_data1$oiltypeid)
dm_data1$oiltypeid <- as.factor(dm_data1$oiltypeid)
summary(dm_data1$oiltypeid)
dm_data1$oiltypeid <- as.character(dm_data1$oiltypeid)
dm_data1$oiltypeid[which(is.na(dm_data1$oiltypeid)==T)] <- 'MISSING' 
dm_data1$oiltypeid <- as.factor(dm_data1$oiltypeid)

dm_data1$oilgradeid[which(is.na(dm_data1$oilgradeid)==T)] <- 'MISSING' 
dm_data1$oilgradeid <- as.factor(dm_data1$oilgradeid)
summary(dm_data1$oilgradeid)

summary(as.factor(dm_data1$coolantid))
dm_data1[,coolantid:=NULL]

summary(as.factor(dm_data1$oiladded))
dm_data1[, oiladded:=NULL]

dm_data1$actual_fluid_hours <- impute_NA(dm_data1$actual_fluid_hours)
num_distribution_plot(dm_data1$actual_fluid_hours, dm_data1)
quantile(dm_data1$actual_fluid_hours)
bin_num <- 7
dm_data1[, actual_fluid_hours_bin := as.integer(cut2(dm_data1$actual_fluid_hours, g = bin_num))]
dm_data1$actual_fluid_hours_bin <- as.factor(dm_data1$actual_fluid_hours_bin)
summary(dm_data1$actual_fluid_hours_bin)
# dm_data1[,actual_fluid_hours:=NULL]

num_distribution_plot(dm_data1$meterread, dm_data1)
quantile(dm_data1$meterread)
bin_num <- 4
dm_data1[, meterread_bin := as.integer(cut2(dm_data1$meterread, g = bin_num))]
dm_data1$meterread_bin <- as.factor(dm_data1$meterread_bin)
summary(dm_data1$meterread_bin)
# dm_data1[,meterread:=NULL]


summary(as.factor(dm_data1$oilchanged))
dm_data1$oilchanged <- as.character(dm_data1$oilchanged)
dm_data1$oilchanged[which(is.na(dm_data1$oilchanged)==T)] <- 'MISSING'
dm_data1$oilchanged <- as.factor(dm_data1$oilchanged)

summary(as.factor(dm_data1$filterchanged))
dm_data1$filterchanged <- as.character(dm_data1$filterchanged)
dm_data1$filterchanged[which(is.na(dm_data1$filterchanged)==T)] <- 'MISSING'
dm_data1$filterchanged <- as.factor(dm_data1$filterchanged)
summary(dm_data1$filterchanged)

summary(as.factor(dm_data1$evalcode))
dm_data1$evalcode <- as.factor(dm_data1$evalcode)

summary(as.factor(dm_data1$`Problem Solved`))
dm_data1[, `Problem Solved`:=NULL]

summary(as.factor(dm_data1$Water))
dm_data1$Water[which(is.na(dm_data1$Water)==T)] <- 'MISSING'
dm_data1$Water <- as.factor(dm_data1$Water)
summary(dm_data1$Water)

dm_data1$V40 <- impute_NA(dm_data1$V40)
num_distribution_plot(dm_data1$V40, dm_data1)  # use bins
quantile(dm_data1$V40)
num_distribution_plot(log(dm_data1$V40), dm_data1)
num_distribution_plot(sqrt(dm_data1$V40), dm_data1)
bin_num <- 3
dm_data1[, V40_bin := as.integer(cut2(dm_data1$V40, g = bin_num))]
dm_data1$V40_bin <- as.factor(dm_data1$V40_bin)
summary(dm_data1$V40_bin)
# dm_data1[,V40:=NULL]

dm_data1$V100 <- impute_NA(dm_data1$V100)
num_distribution_plot(dm_data1$V100, dm_data1)  # use bins
quantile(dm_data1$V100)
num_distribution_plot(log(dm_data1$V100), dm_data1)
quantile(log(dm_data1$V100))  # use bins
num_distribution_plot(sqrt(dm_data1$V100), dm_data1)
bin_num <- 7
dm_data1[, V100_bin := as.integer(cut2(dm_data1$V100, g = bin_num))]
dm_data1$V100_bin <- as.factor(dm_data1$V100_bin)
summary(dm_data1$V100_bin)
# dm_data1[,V100:=NULL]

head(dm_data1$VI)
dm_data1$VI <- as.character(as.factor(dm_data1$VI))
dm_data1$VI <- impute_NA(dm_data1$VI)
dm_data1$VI <- as.factor(dm_data1$VI)
summary(dm_data1$VI)

# remove those with many empty values
dm_data1[, `GC Fuel`:=NULL]
colnames(dm_data1)
remove_lst <- c(17:35)
dm_data1 <- dm_data1[, -remove_lst, with=F]
dm_data1[, `Cap Inspect`:=NULL]
summarizeColumns(dm_data1)

dm_data1$Fe <- impute_NA(dm_data1$Fe)
num_distribution_plot(dm_data1$Fe, dm_data1)
quantile(dm_data1$Fe)    # use bins
num_distribution_plot(sqrt(dm_data1$Fe), dm_data1)
bin_num <- 4
dm_data1[, Fe_bin := as.integer(cut2(dm_data1$Fe, g = bin_num))]
dm_data1$Fe_bin <- as.factor(dm_data1$Fe_bin)
summary(dm_data1$Fe_bin)
# dm_data1[,Fe:=NULL]

dm_data1$Cu <- impute_NA(dm_data1$Cu)
num_distribution_plot(dm_data1$Cu, dm_data1)
quantile(dm_data1$Cu)    # use bins
bin_num <- 4
dm_data1[, Cu_bin := as.integer(cut2(dm_data1$Cu, g = bin_num))]
dm_data1$Cu_bin <- as.factor(dm_data1$Cu_bin)
summary(dm_data1$Cu_bin)
# dm_data1[,Cu:=NULL]

dm_data1$Pb <- impute_NA(dm_data1$Pb)
num_distribution_plot(dm_data1$Pb, dm_data1)
quantile(dm_data1$Pb) # use bins
bin_num <- 2
dm_data1[, Pb_bin := as.integer(cut2(dm_data1$Pb, g = bin_num))]
dm_data1$Pb_bin <- as.factor(dm_data1$Pb_bin)
summary(dm_data1$Pb_bin)
# dm_data1[,Pb:=NULL]

dm_data1$Sn <- impute_NA(dm_data1$Sn)
num_distribution_plot(dm_data1$Sn, dm_data1)
quantile(dm_data1$Sn) # use bins
bin_num <- 2
dm_data1[, Sn_bin := as.integer(cut2(dm_data1$Sn, g = bin_num))]
dm_data1$Sn_bin <- as.factor(dm_data1$Sn_bin)
summary(dm_data1$Sn_bin)
# dm_data1[,Sn:=NULL]

dm_data1$Cr <- impute_NA(dm_data1$Cr)
num_distribution_plot(dm_data1$Cr, dm_data1)
quantile(dm_data1$Cr) # use bins
bin_num <- 3
dm_data1[, Cr_bin := as.integer(cut2(dm_data1$Cr, g = bin_num))]
dm_data1$Cr_bin <- as.factor(dm_data1$Cr_bin)
summary(dm_data1$Cr_bin)
# dm_data1[,Cr:=NULL]

dm_data1$Ni <- impute_NA(dm_data1$Ni)
num_distribution_plot(dm_data1$Ni, dm_data1)
quantile(dm_data1$Ni) # use bins
bin_num <- 3
dm_data1[, Ni_bin := as.integer(cut2(dm_data1$Ni, g = bin_num))]
dm_data1$Ni_bin <- as.factor(dm_data1$Ni_bin)
summary(dm_data1$Ni_bin)
# dm_data1[,Ni:=NULL]

dm_data1$Ti <- impute_NA(dm_data1$Ti)
num_distribution_plot(dm_data1$Ti, dm_data1)
quantile(dm_data1$Ti) # use bins
bin_num <- 2
dm_data1[, Ti_bin := as.integer(cut2(dm_data1$Ti, g = bin_num))]
dm_data1$Ti_bin <- as.factor(dm_data1$Ti_bin)
summary(dm_data1$Ti_bin)
# dm_data1[,Ti:=NULL]

dm_data1$Al <- impute_NA(dm_data1$Al)
num_distribution_plot(dm_data1$Al, dm_data1)
quantile(dm_data1$Al) # use bins
bin_num <- 3
dm_data1[, Al_bin := as.integer(cut2(dm_data1$Al, g = bin_num))]
dm_data1$Al_bin <- as.factor(dm_data1$Al_bin)
summary(dm_data1$Al_bin)
# dm_data1[,Al:=NULL]

dm_data1$Si <- impute_NA(dm_data1$Si)
num_distribution_plot(dm_data1$Si, dm_data1)
quantile(dm_data1$Si) # use bins
bin_num <- 4
dm_data1[, Si_bin := as.integer(cut2(dm_data1$Si, g = bin_num))]
dm_data1$Si_bin <- as.factor(dm_data1$Si_bin)
summary(dm_data1$Si_bin)
# dm_data1[,Si:=NULL]

dm_data1$Na <- impute_NA(dm_data1$Na)
num_distribution_plot(dm_data1$Na, dm_data1)
quantile(dm_data1$Na) # use bins
bin_num <- 4
dm_data1[, Na_bin := as.integer(cut2(dm_data1$Na, g = bin_num))]
dm_data1$Na_bin <- as.factor(dm_data1$Na_bin)
summary(dm_data1$Na_bin)
# dm_data1[,Na:=NULL]

dm_data1$K <- impute_NA(dm_data1$K)
num_distribution_plot(dm_data1$K, dm_data1)
quantile(dm_data1$K) # use bins
bin_num <- 3
dm_data1[, K_bin := as.integer(cut2(dm_data1$K, g = bin_num))]
dm_data1$K_bin <- as.factor(dm_data1$K_bin)
summary(dm_data1$K_bin)
# dm_data1[,K:=NULL]

dm_data1$B <- impute_NA(dm_data1$B)
num_distribution_plot(dm_data1$B, dm_data1)
quantile(dm_data1$B) # use bins
bin_num <- 7
dm_data1[, B_bin := as.integer(cut2(dm_data1$B, g = bin_num))]
dm_data1$B_bin <- as.factor(dm_data1$B_bin)
summary(dm_data1$B_bin)
# dm_data1[, B:=NULL]

dm_data1$Ca <- impute_NA(dm_data1$Ca)
num_distribution_plot(dm_data1$Ca, dm_data1)
quantile(dm_data1$Ca) # use bins
bin_num <- 7
dm_data1[, Ca_bin := as.integer(cut2(dm_data1$Ca, g = bin_num))]
dm_data1$Ca_bin <- as.factor(dm_data1$Ca_bin)
summary(dm_data1$Ca_bin)
# dm_data1[,Ca:=NULL]


dm_data1$Mg <- impute_NA(dm_data1$Mg)
num_distribution_plot(dm_data1$Mg, dm_data1)
quantile(dm_data1$Mg) # use bins
bin_num <- 4
dm_data1[, Mg_bin := as.integer(cut2(dm_data1$Mg, g = bin_num))]
dm_data1$Mg_bin <- as.factor(dm_data1$Mg_bin)
summary(dm_data1$Mg_bin)
# dm_data1[,Mg:=NULL]


dm_data1$P <- impute_NA(dm_data1$P)
num_distribution_plot(dm_data1$P, dm_data1)
quantile(dm_data1$P) # use bins
bin_num <- 7
dm_data1[, P_bin := as.integer(cut2(dm_data1$P, g = bin_num))]
dm_data1$P_bin <- as.factor(dm_data1$P_bin)
summary(dm_data1$P_bin)
# dm_data1[,P:=NULL]

dm_data1$Zn <- impute_NA(dm_data1$Zn)
num_distribution_plot(dm_data1$Zn, dm_data1)
quantile(dm_data1$Zn) # use bins
bin_num <- 7
dm_data1[, Zn_bin := as.integer(cut2(dm_data1$Zn, g = bin_num))]
dm_data1$Zn_bin <- as.factor(dm_data1$Zn_bin)
summary(dm_data1$Zn_bin)
# dm_data1[,Zn:=NULL]

dm_data1$Mo <- impute_NA(dm_data1$Mo)
num_distribution_plot(dm_data1$Mo, dm_data1)
quantile(dm_data1$Mo) # use bins
bin_num <- 4
dm_data1[, Mo_bin := as.integer(cut2(dm_data1$Mo, g = bin_num))]
dm_data1$Mo_bin <- as.factor(dm_data1$Mo_bin)
summary(dm_data1$Mo_bin)
# dm_data1[,Mo:=NULL]

dm_data1$Li <- impute_NA(dm_data1$Li)
num_distribution_plot(dm_data1$Li, dm_data1)
quantile(dm_data1$Li) # use bins
bin_num <- 2
dm_data1[, Li_bin := as.integer(cut2(dm_data1$Li, g = bin_num))]
dm_data1$Li_bin <- as.factor(dm_data1$Li_bin)
summary(dm_data1$Li_bin)
# dm_data1[,Li:=NULL]

dm_data1$Sb <- impute_NA(dm_data1$Sb)
num_distribution_plot(dm_data1$Sb, dm_data1)
quantile(dm_data1$Sb) # use bins
bin_num <- 2
dm_data1[, Sb_bin := as.integer(cut2(dm_data1$Sb, g = bin_num))]
dm_data1$Sb_bin <- as.factor(dm_data1$Sb_bin)
summary(dm_data1$Sb_bin)
# dm_data1[,Sb:=NULL]

dm_data1$Ba <- impute_NA(dm_data1$Ba)
num_distribution_plot(dm_data1$Ba, dm_data1)
quantile(dm_data1$Ba) # use bins
bin_num <- 2
dm_data1[, Ba_bin := as.integer(cut2(dm_data1$Ba, g = bin_num))]
dm_data1$Ba_bin <- as.factor(dm_data1$Ba_bin)
summary(dm_data1$Ba_bin)
# dm_data1[,Ba:=NULL]

dm_data1$Cd <- impute_NA(dm_data1$Cd)
num_distribution_plot(dm_data1$Cd, dm_data1)
quantile(dm_data1$Cd) # use bins
bin_num <- 2
dm_data1[, Cd_bin := as.integer(cut2(dm_data1$Cd, g = bin_num))]
dm_data1$Cd_bin <- as.factor(dm_data1$Cd_bin)
summary(dm_data1$Cd_bin)
# dm_data1[,Cd:=NULL]

dm_data1$Mn <- impute_NA(dm_data1$Mn)
num_distribution_plot(dm_data1$Mn, dm_data1)
quantile(dm_data1$Mn) # use bins
bin_num <- 2
dm_data1[, Mn_bin := as.integer(cut2(dm_data1$Mn, g = bin_num))]
dm_data1$Mn_bin <- as.factor(dm_data1$Mn_bin)
summary(dm_data1$Mn_bin)
# dm_data1[,Mn:=NULL]

dm_data1$Ag <- impute_NA(dm_data1$Ag)
num_distribution_plot(dm_data1$Ag, dm_data1)
quantile(dm_data1$Ag) # use bins
bin_num <- 2
dm_data1[, Ag_bin := as.integer(cut2(dm_data1$Ag, g = bin_num))]
dm_data1$Ag_bin <- as.factor(dm_data1$Ag_bin)
summary(dm_data1$Ag_bin)
# dm_data1[,Ag:=NULL]

dm_data1$V <- impute_NA(dm_data1$V)
num_distribution_plot(dm_data1$V, dm_data1)
quantile(dm_data1$V) # use bins
bin_num <- 2
dm_data1[, V_bin := as.integer(cut2(dm_data1$V, g = bin_num))]
dm_data1$V_bin <- as.factor(dm_data1$V_bin)
summary(dm_data1$V_bin)
# dm_data1[,V:=NULL]

dm_data1$`4u` <- impute_NA(dm_data1$`4u`)
num_distribution_plot(dm_data1$`4u`, dm_data1)
quantile(dm_data1$`4u`) 
quantile(log(dm_data1$`4u`))
dm_data1$`4u` <- log(dm_data1$`4u`)  # use bins
bin_num <- 7
dm_data1[, `4u_bin` := as.integer(cut2(dm_data1$`4u`, g = bin_num))]
dm_data1$`4u_bin` <- as.factor(dm_data1$`4u_bin`)
summary(dm_data1$`4u_bin`)
# dm_data1[,`4u`:=NULL]

dm_data1$`6u` <- impute_NA(dm_data1$`6u`)
num_distribution_plot(dm_data1$`6u`, dm_data1)
quantile(dm_data1$`6u`) 
quantile(log(dm_data1$`6u`))
dm_data1$`6u` <- log(dm_data1$`6u`)  # use bins
bin_num <- 7
dm_data1[, `6u_bin` := as.integer(cut2(dm_data1$`6u`, g = bin_num))]
dm_data1$`6u_bin` <- as.factor(dm_data1$`6u_bin`)
summary(dm_data1$`6u_bin`)
# dm_data1[,`6u`:=NULL]

dm_data1$`14u` <- impute_NA(dm_data1$`14u`)
num_distribution_plot(dm_data1$`14u`, dm_data1)
quantile(dm_data1$`14u`) 
quantile(log(dm_data1$`14u`))
dm_data1$`14u` <- log(dm_data1$`14u`)  # use bins
bin_num <- 7
dm_data1[, `14u_bin` := as.integer(cut2(dm_data1$`14u`, g = bin_num))]
dm_data1$`14u_bin` <- as.factor(dm_data1$`14u_bin`)
summary(dm_data1$`14u_bin`)
# dm_data1[,`14u`:=NULL]

dm_data1$`21u` <- impute_NA(dm_data1$`21u`)
num_distribution_plot(dm_data1$`21u`, dm_data1)
quantile(dm_data1$`21u`) # use bins
bin_num <- 7
dm_data1[, `21u_bin` := as.integer(cut2(dm_data1$`21u`, g = bin_num))]
dm_data1$`21u_bin` <- as.factor(dm_data1$`21u_bin`)
summary(dm_data1$`21u_bin`)
# dm_data1[,`21u`:=NULL]

dm_data1$`38u` <- impute_NA(dm_data1$`38u`)
num_distribution_plot(dm_data1$`38u`, dm_data1)
quantile(dm_data1$`38u`) # use bins
bin_num <- 7
dm_data1[, `38u_bin` := as.integer(cut2(dm_data1$`38u`, g = bin_num))]
dm_data1$`38u_bin` <- as.factor(dm_data1$`38u_bin`)
summary(dm_data1$`38u_bin`)
# dm_data1[,`38u`:=NULL]

dm_data1$`70u` <- impute_NA(dm_data1$`70u`)
num_distribution_plot(dm_data1$`70u`, dm_data1)
quantile(dm_data1$`70u`) # use bins
bin_num <- 3
dm_data1[, `70u_bin` := as.integer(cut2(dm_data1$`70u`, g = bin_num))]
dm_data1$`70u_bin` <- as.factor(dm_data1$`70u_bin`)
summary(dm_data1$`70u_bin`)
# dm_data1[,`70u`:=NULL]

dm_data1$ISO[which(is.na(dm_data1$ISO)==T)] <- 'MISSING'
dm_data1$ISO <- as.factor(dm_data1$ISO)
summary(dm_data1$ISO)

dm_data1$Cutting <- impute_NA(dm_data1$Cutting)
num_distribution_plot(dm_data1$Cutting, dm_data1)
quantile(dm_data1$Cutting) # use bins
bin_num <- 4
dm_data1[, Cutting_bin := as.integer(cut2(dm_data1$Cutting, g = bin_num))]
dm_data1$Cutting_bin <- as.factor(dm_data1$Cutting_bin)
summary(dm_data1$Cutting_bin)
# dm_data1[,Cutting:=NULL]

dm_data1$Sliding <- impute_NA(dm_data1$Sliding)
num_distribution_plot(dm_data1$Sliding, dm_data1)
quantile(dm_data1$Sliding) # use bins
bin_num <= 4
dm_data1[, Sliding_bin := as.integer(cut2(dm_data1$Sliding, g = bin_num))]
dm_data1$Sliding_bin <- as.factor(dm_data1$Sliding_bin)
summary(dm_data1$Sliding_bin)
# dm_data1[,Sliding:=NULL]

dm_data1$Fatigue <- impute_NA(dm_data1$Fatigue)
num_distribution_plot(dm_data1$Fatigue, dm_data1)
quantile(dm_data1$Fatigue) # use bins
bin_num <- 7
dm_data1[, Fatigue_bin := as.integer(cut2(dm_data1$Fatigue, g = bin_num))]
dm_data1$Fatigue_bin <- as.factor(dm_data1$Fatigue_bin)
summary(dm_data1$Fatigue_bin)
# dm_data1[,Fatigue:=NULL]

dm_data1$`Non Metallic` <- impute_NA(dm_data1$`Non Metallic`)
num_distribution_plot(dm_data1$`Non Metallic`, dm_data1)
quantile(dm_data1$`Non Metallic`) # use bins
bin_num <- 7
dm_data1[, Non_Metallic_bin := as.integer(cut2(dm_data1$`Non Metallic`, g = bin_num))]
dm_data1$Non_Metallic_bin <- as.factor(dm_data1$Non_Metallic_bin)
summary(dm_data1$Non_Metallic_bin)
# dm_data1[,`Non Metallic`:=NULL]

dm_data1$Fibers <- impute_NA(dm_data1$Fibers)
num_distribution_plot(dm_data1$Fibers, dm_data1)
quantile(dm_data1$Fibers) # use bins
bin_num <- 7
dm_data1[, Fibers_bin := as.integer(cut2(dm_data1$Fibers, g = bin_num))]
dm_data1$Fibers_bin <- as.factor(dm_data1$Fibers_bin)
summary(dm_data1$Fibers_bin)
# dm_data1[,Fibers:=NULL]

dm_data1$`Total Part/ml` <- impute_NA(dm_data1$`Total Part/ml`)
num_distribution_plot(dm_data1$`Total Part/ml`, dm_data1)
quantile(dm_data1$`Total Part/ml`) 
quantile(log(dm_data1$`Total Part/ml`))  # use bins
dm_data1$`Total Part/ml` <- log(dm_data1$`Total Part/ml`)
bin_num <- 7
dm_data1[, ml_bin := as.integer(cut2(dm_data1$`Total Part/ml`, g = bin_num))]
dm_data1$ml_bin <- as.factor(dm_data1$ml_bin)
summary(dm_data1$ml_bin)
# dm_data1[,`Total Part/ml`:=NULL]

dm_data1$NO2 <- impute_NA(dm_data1$NO2)
num_distribution_plot(dm_data1$NO2, dm_data1)
quantile(dm_data1$NO2) # use bins
summary(as.factor(dm_data1$NO2))
dm_data1$NO2 <- as.factor(dm_data1$NO2)

dm_data1$`pqL Index` <- impute_NA(dm_data1$`pqL Index`)
num_distribution_plot(dm_data1$`pqL Index`, dm_data1)
quantile(dm_data1$`pqL Index`) # use bins
bin_num <- 7
dm_data1[, pql_bin := as.integer(cut2(dm_data1$`pqL Index`, g = bin_num))]
dm_data1$pql_bin <- as.factor(dm_data1$pql_bin)
summary(dm_data1$pql_bin)
# dm_data1[,`pqL Index`:=NULL]

dm_data1$Soot <- impute_NA(dm_data1$Soot)
num_distribution_plot(dm_data1$Soot, dm_data1)
quantile(dm_data1$Soot) # use bins
bin_num <- 2
dm_data1[, Soot_bin := as.integer(cut2(dm_data1$Soot, g = bin_num))]
dm_data1$Soot_bin <- as.factor(dm_data1$Soot_bin)
summary(dm_data1$Soot_bin)
# dm_data1[,Soot:=NULL]

dm_data1$OXI <- impute_NA(dm_data1$OXI)
num_distribution_plot(dm_data1$OXI, dm_data1)
quantile(dm_data1$OXI) # use bins
bin_num <- 4
dm_data1[, OXI_bin := as.integer(cut2(dm_data1$OXI, g = bin_num))]
dm_data1$OXI_bin <- as.factor(dm_data1$OXI_bin)
summary(dm_data1$OXI_bin)
# dm_data1[,OXI:=NULL]

dm_data1$NIT <- impute_NA(dm_data1$NIT)
num_distribution_plot(dm_data1$NIT, dm_data1)
quantile(dm_data1$NIT) # use bins
bin_num <- 4
dm_data1[, NIT_bin := as.integer(cut2(dm_data1$NIT, g = bin_num))]
dm_data1$NIT_bin <- as.factor(dm_data1$NIT_bin)
summary(dm_data1$NIT_bin)
# dm_data1[,NIT:=NULL]

dm_data1$Sulf <- impute_NA(dm_data1$Sulf)
num_distribution_plot(dm_data1$Sulf, dm_data1)
quantile(dm_data1$Sulf) # use bins
bin_num <- 4
dm_data1[, Sulf_bin := as.integer(cut2(dm_data1$Sulf, g = bin_num))]
dm_data1$Sulf_bin <- as.factor(dm_data1$Sulf_bin)
summary(dm_data1$Sulf_bin)
# dm_data1[,Sulf:=NULL]

dm_data1$AW <- impute_NA(dm_data1$AW)
num_distribution_plot(dm_data1$AW, dm_data1)
quantile(dm_data1$AW) # use bins
bin_num <- 4
dm_data1[, AW_bin := as.integer(cut2(dm_data1$AW, g = bin_num))]
dm_data1$AW_bin <- as.factor(dm_data1$AW_bin)
summary(dm_data1$AW_bin)
# dm_data1[,AW:=NULL]

dm_data1$`FT-IR Glycol` <- impute_NA(dm_data1$`FT-IR Glycol`)
num_distribution_plot(dm_data1$`FT-IR Glycol`, dm_data1)
quantile(dm_data1$`FT-IR Glycol`) # use bins
bin_num <- 4
dm_data1[, FTIR_bin := as.integer(cut2(dm_data1$`FT-IR Glycol`, g = bin_num))]
dm_data1$FTIR_bin <- as.factor(dm_data1$FTIR_bin)
summary(dm_data1$FTIR_bin)
# dm_data1[,`FT-IR Glycol`:=NULL]

dm_data1$`FT-IR Water` <- impute_NA(dm_data1$`FT-IR Water`)
num_distribution_plot(dm_data1$`FT-IR Water`, dm_data1)
quantile(dm_data1$`FT-IR Water`) # use bins
bin_num <- 4
dm_data1[, FTIRWATER_bin := as.integer(cut2(dm_data1$`FT-IR Water`, g = bin_num))]
dm_data1$FTIRWATER_bin <- as.factor(dm_data1$FTIRWATER_bin)
summary(dm_data1$FTIRWATER_bin)
# dm_data1[,`FT-IR Water`:=NULL]

# remove highly correlation data
library(caret)
fact_cols <- sapply(dm_data1, is.factor)
num_data <- data.table(subset(dm_data1, select = fact_cols==F))
fact_data <- data.table(subset(dm_data1, select = fact_cols==T))
ax <-findCorrelation(x = cor(num_data), cutoff = 0.7)   # 0.7 is the threshold here
sort(ax)
num_data <- num_data[, -ax, with=F]
scaled_num_data <- data.frame(sapply(num_data, data_scaling))  # normalize numerican data into [0,1]
dm_data1 <- cbind(scaled_num_data, fact_data)
rm(num_data)
rm(fact_data)

# remove 0 variance feature
variance_lst <- nearZeroVar(dm_data1, saveMetrics = T)
zero_variance_list <- names(subset(dm_data1, select = variance_lst$zeroVar==T))
zero_variance_list


## split into train and test data
idx <- createDataPartition(dm_data1$ACCTDESC, p=0.77, list=FALSE)
train_data <- dm_data1[idx,]
test_data <- dm_data1[-idx,]

## ROUND1,2 - feature selection
library(Boruta)
set.seed(410)
boruta_train <- Boruta(ACCTDESC~., data = train_data, doTrace = 2)
boruta_train
## plot feature importance
plot(boruta_train, xlab = "", xaxt = "n")
str(boruta_train)
summary(boruta_train$ImpHistory)
finite_matrix <- lapply(1:ncol(boruta_train$ImpHistory), 
                        function(i) boruta_train$ImpHistory[is.finite(boruta_train$ImpHistory[,i]), i])
names(finite_matrix) <- colnames(boruta_train$ImpHistory)
plot_labels <- sort(sapply(finite_matrix, median))
axis(side = 1, las = 2, labels = names(plot_labels), 
     at = 1:ncol(boruta_train$ImpHistory), cex.axis = 0.7)
## determine tentative features
new_boruta_train <- TentativeRoughFix(boruta_train)
new_boruta_train
plot(new_boruta_train, xlab = "", xaxt = "n")
finite_matrix <- lapply(1:ncol(new_boruta_train$ImpHistory), 
                        function(i) new_boruta_train$ImpHistory[is.finite(new_boruta_train$ImpHistory[,i]), i])
names(finite_matrix) <- colnames(new_boruta_train$ImpHistory) 
plot_labels <- sort(sapply(finite_matrix, median))
axis(side = 1, las = 2, labels = names(plot_labels), 
     at = 1:ncol(new_boruta_train$ImpHistory), cex.axis = 0.7)
feature_stats = attStats(new_boruta_train)
feature_stats
selected_cols <- getSelectedAttributes(new_boruta_train, withTentative = F)

## generate training, testing data based on selected features
selected_train <- subset(train_data, select = colnames(train_data) %in% selected_cols)
selected_train <- data.table(selected_train)
selected_train[, ACCTDESC:=train_data$ACCTDESC]
selected_test <- subset(test_data, select = colnames(test_data) %in% selected_cols)
selected_test <- data.table(selected_test)
selected_test[, ACCTDESC:=test_data$ACCTDESC]

## prediction
selected_train[, ISO:=NULL]  # too many levels, random forets won't be able to handle it
selected_test[, ISO:=NULL]
selected_train[, oil_sample_date:=NULL]
selected_test[, oil_sample_date:=NULL]
selected_train[, VI:=NULL]
selected_test[, VI:=NULL]
train_task <- makeClassifTask(data=data.frame(selected_train), target = "ACCTDESC", positive = "24V MECHANICAL")
test_task <- makeClassifTask(data=data.frame(selected_test), target = "ACCTDESC", positive = "24V MECHANICAL")
### random forest
set.seed(410)
getParamSet("classif.randomForest")
rf_learner <- makeLearner("classif.randomForest", predict.type = "response", par.vals = list(ntree = 200, mtry = 3))
rf_learner$par.vals <- list(importance = TRUE)
rf_param <- makeParamSet(
  makeIntegerParam("ntree",lower = 50, upper = 500),
  makeIntegerParam("mtry", lower = 3, upper = 10),
  makeIntegerParam("nodesize", lower = 10, upper = 50)
)
rancontrol <- makeTuneControlRandom(maxit = 50L)
cv_rf <- makeResampleDesc("CV",iters = 5L)
rf_tune <- tuneParams(learner = rf_learner, resampling = cv_rf, task = train_task, par.set = rf_param, control = rancontrol, measures = acc)
rf_tune$x
rf_tune$y
rf.tree <- setHyperPars(rf_learner, par.vals = rf_tune$x)
rf_model <- mlr::train(learner=rf.tree, task=train_task)
getLearnerModel(rf_model)
rfpredict <- predict(rf_model, test_task)
nb_prediction <- rfpredict$data$response
library(e1071)
dCM <- confusionMatrix(test_data$ACCTDESC, nb_prediction, positive = "24V MECHANICAL")
dCM    # round 1 - balanced accuracy:0.67655, Specificity : 0.81309, Sensitivity : 0.54000
      # round 2 (used bins) - balanced accuracy:0.63434, Specificity : 0.81822, Sensitivity : 0.45045
