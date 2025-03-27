getwd()

no_diff_1 <- readRDS("cbc-no-diff-testing.rds")
no_diff_2 <- readRDS("cbc-no-diff-training.rds")
no_diff_3 <- readRDS("cbc-no-diff-validation.rds")
no_diff <- rbind(no_diff_1, no_diff_2, no_diff_3)

diff_1 <- readRDS("cbc-with-diff-testing.rds")
diff_2 <- readRDS("cbc-with-diff-training.rds")
diff_3 <- readRDS("cbc-with-diff-validation.rds")
diff <- rbind(diff_1, diff_2, diff_3)

write.csv(no_diff, "no_diff.csv")
write.csv(diff, "diff.csv")
          
a_1 <- as.integer(no_diff_1$wbit_error)
a_2 <- as.integer(no_diff_2$wbit_error)  
a_3 <- as.integer(no_diff_3$wbit_error)  

mean(a_1)
mean(a_2)
mean(a_3)
