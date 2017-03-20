path = 'DataAnalysis/DA2/all.csv'
#file all.csv contains all crimes from all folders
df = read.csv(path, sep = ',')
library(dplyr)

nums = df %>% group_by(CR) %>% summarise(TOTAL=sum(TOT))
print ('Top frequent cirmes')
print (nums[order(-nums$TOTAL),])

barplot(nums$TOTAL, names.arg = nums$CR, las=2, main = 'Amount of crimes by type')

av = nums$TOTAL/sum(nums$TOTAL)
plot(av, xaxt='n', xlab='CRIME', ylab='AVERAGE', main = 'Density of crimes by type')
lines(av)
axis(1, at = 1:nrow(nums), labels = nums$CR, las=2)

nums = df[c("YR","MO", "TOT")]
nums = nums %>% group_by(YR, MO) %>% summarise(TOTAL=sum(TOT))
xticks = paste(nums$YR, nums$MO, sep = '-')
barplot(nums$TOTAL, names.arg = xticks, las = 2, main = 'Amount of crimes by month')


