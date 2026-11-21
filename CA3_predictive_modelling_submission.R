# =====================================================================
# Christopher McGlynn
#
# CA3 – Predictive modelling: Predicting aces at Wimbledon, 2022–2025
#
# Dataset : Jeff Sackmann, tennis_atp
#           https://github.com/JeffSackmann/tennis_atp
#           Files atp_matches_2022.csv to atp_matches_2025.csv
# Structure: one row = one match. Columns starting "winner_" / "w_"
#           for winner, "loser_" / "l_" for loser.
#
# RESEARCH QUESTION
# Can the number of aces a player serves in a Wimbledon match be predicted
# from the player's height, age, ranking and playing hand, together with
# how much they served, the match result and the round?
#
# This continues CA2, which tested one relationship at a time (Q1: do
# taller players serve more aces?). CA3 uses multiple linear regression
# to look at several variables together and predict aces.
#
#   y = b0 + b1x1 + b2x2 + ... + e
#   y  = aces (dependent variable)
#   x  = height, age, rank, service points (numeric)
#        hand, result, round (categorical)
#
# The script has two parts:
#   PART 1 – Data preparation (steps 1 to 8)
#            The same code as CA2, so both use the same 489 matches.
#            Lines added for CA3 are marked "CA3 addition".
#   PART 2 – Building and testing the model (steps 9 to 18)
# =====================================================================

#install.packages(c("psych", "e1071", "MASS", "lmtest", "faraway"))  # run once only
library(psych)     # pairs.panels() – correlation chart
library(e1071)     # skewness()
library(MASS)      # boxcox() – Box-Cox transformation
library(lmtest)    # dwtest() – Durbin-Watson test
library(faraway)   # vif() – variance inflation factor

# Each chart is drawn on its own, so every chart appears separately in
# the RStudio Plots pane.


# =====================================================================
# PART 1 – DATA PREPARATION (same as CA2)
# =====================================================================

# ---- Step 1: Read the dataset ----------------------------
all_matches <- read.csv("~/Desktop/DataScienceGIT/CA3/Data/atp_matches_2022_2025.csv")
dim(all_matches)
str(all_matches)   # shows the type of every column
cat("Matches in the full dataset:", nrow(all_matches), "\n") # 11,923 rows (matches)
cat("Columns in the full dataset:", ncol(all_matches), "\n") # 49 columns

# The file has no year column, so create one from tournament date
all_matches$year <- all_matches$tourney_date %/% 10000
table(all_matches$year)

# ---- Step 2: Keep Wimbledon matches only ----------------------------
wimbledon <- subset(all_matches, tourney_name == "Wimbledon")
dim(wimbledon)
table(wimbledon$year)  # 127 matches every year
cat("Wimbledon matches:", nrow(wimbledon), "\n") # 508 matches

# These three columns can be dropped as Wimbledon will have the same
# figures for each one and therefore no point including these
table(wimbledon$surface)
table(wimbledon$best_of)
table(wimbledon$tourney_level)

# ---- Step 3: Remove retirements and walkovers -----------------------
# A match that stopped early (RET = retired, W/O = walkover,
# DEF = default) would effect the results.
# grepl() returns TRUE for every score containing one of those words
retired <- grepl("RET|W/O|DEF", wimbledon$score)
sum(retired)
cat("Matches that did not finish:", sum(retired), "\n")  # 17 matches did not finish
wimbledon <- wimbledon[!retired, ]
nrow(wimbledon)
cat("Matches left after removing retirements and walkovers:", nrow(wimbledon), "\n")  # 491 matches left


# ---- Step 4: Keep only the variables needed --------------------------
# CA3 addition: the last line adds ranking and service points, which CA3
# uses as predictors. They have no missing values, so no extra matches
# are lost in step 5.
include_list <- names(wimbledon) %in% c("year", "round", "minutes",
                                        "winner_id", "winner_name", "winner_hand", "winner_ht", "winner_age",
                                        "loser_id", "loser_name", "loser_hand", "loser_ht", "loser_age",
                                        "w_ace", "l_ace", "w_df", "l_df",
                                        "winner_rank", "loser_rank", "w_svpt", "l_svpt")   # CA3 addition

wimbledon <- wimbledon[(include_list)]
str(wimbledon)


# ---- Step 5: Missing data ------------------------------------------
# Count the NAs in each column
colSums(is.na(wimbledon))            # one missing height for a winner and one for a loser
# Show the rows that have at least one NA
wimbledon[!complete.cases(wimbledon), ]

# Playing hand uses "U" for unknown instead of NA, so it is not
# picked up by is.na() so had to convert "U" to NA so R treats it as missing.
table(wimbledon$winner_hand)
table(wimbledon$loser_hand)
wimbledon$winner_hand[wimbledon$winner_hand == "U"] <- NA
wimbledon$loser_hand[wimbledon$loser_hand == "U"] <- NA

# na.omit() removes every row with an NA.
# Only 2 matches are affected
wimbledon <- na.omit(wimbledon)
nrow(wimbledon)                      # 489 matches - the same as CA2

# ---- Step 6: Rename and convert variable types ---------------------
# I changed minutes to match minutes
names(wimbledon)[names(wimbledon) == "minutes"] <- "match_minutes"

# Round is ORDINAL data - it has a natural order (first round -> final),
# so convert it to an ORDERED factor
wimbledon$round <- factor(wimbledon$round, order = TRUE,
                          levels = c("R128", "R64", "R32", "R16", "QF", "SF", "F"))
table(wimbledon$round)   # QF = 15, SF = 7, F = 4 – too few on their own

# Group the quarter-finals, semi-finals and final into one level, "QF+",
# so that every group has enough matches
wimbledon$round_group <- as.character(wimbledon$round)
wimbledon$round_group[wimbledon$round %in% c("QF", "SF", "F")] <- "QF+"
wimbledon$round_group <- factor(wimbledon$round_group, order = TRUE,
                                levels = c("R128", "R64", "R32", "R16", "QF+"))
table(wimbledon$round_group)

# CA3 note: a dummy variable for the final alone would rest on only 8
# player rows, so CA3 also uses round_group rather than round.

# Playing hand is NOMINAL data – the categories have no natural order –
# so convert it to a plain (unordered) factor
wimbledon$winner_hand <- factor(wimbledon$winner_hand, levels = c("R", "L"),
                                labels = c("Right", "Left"))
wimbledon$loser_hand  <- factor(wimbledon$loser_hand,  levels = c("R", "L"),
                                labels = c("Right", "Left"))

# (CA2 step 7 – age_diff and df_diff for the paired tests – is not needed in CA3)


# ---- Step 8: Restructure to one row per player per match -----------
# At the moment each row in the spreadsheet holds TWO players.
# I split every match into a winner row and a loser row with the same
# column names. This is essential here, because the model predicts the
# aces of ONE player.
#
# CA3 addition: every match gets a number (1 to 489). Both the winner row
# and the loser row get the same number, so the two players from one match
# can be kept together when the data is split in step 17.
wimbledon$match_id <- 1:nrow(wimbledon)

winners <- data.frame(match_id = wimbledon$match_id,                                         # CA3 addition
                      year = wimbledon$year, round = wimbledon$round,
                      height = wimbledon$winner_ht, age = wimbledon$winner_age,
                      hand = wimbledon$winner_hand, aces = wimbledon$w_ace,
                      double_faults = wimbledon$w_df, result = "Won",
                      player = wimbledon$winner_name, round_group = wimbledon$round_group,  # CA3 addition
                      rank = wimbledon$winner_rank, service_points = wimbledon$w_svpt)     # CA3 addition
losers  <- data.frame(match_id = wimbledon$match_id,                                         # CA3 addition
                      year = wimbledon$year, round = wimbledon$round,
                      height = wimbledon$loser_ht, age = wimbledon$loser_age,
                      hand = wimbledon$loser_hand, aces = wimbledon$l_ace,
                      double_faults = wimbledon$l_df, result = "Lost",
                      player = wimbledon$loser_name, round_group = wimbledon$round_group,  # CA3 addition
                      rank = wimbledon$loser_rank, service_points = wimbledon$l_svpt)      # CA3 addition

# rbind() stacks the winner and loser rows
players <- rbind(winners, losers)     # 489 x 2 = 978 rows

# result is a new categorical variable (Won / Lost) -> factor
players$result <- factor(players$result, levels = c("Won", "Lost"))
str(players)

# (CA2 steps 9 and 10 – descriptive statistics and charts for the five
#  hypothesis tests – are not repeated. The CA3 variables are described
#  in steps 10 to 12.)


# =====================================================================
# PART 2 – BUILDING AND TESTING THE MODEL
# =====================================================================

# ---- Step 9: Categorical variables for regression  ----
# lm() turns each factor into dummy (0 / 1) variables automatically.
# A variable with N categories needs N - 1 dummy variables; the category
# left out is the baseline.
#
# hand        – 2 categories, 1 dummy (Right = baseline)
# round_group – 5 categories, 4 dummies (R128 = baseline).
#               lm() does not make dummies from an ORDERED factor, so it
#               is converted to an unordered factor.
# result      – 2 categories, 1 dummy. "Lost" is listed first so it is
#               the baseline and the model shows the effect of WINNING.

players$round_group <- factor(players$round_group, ordered = FALSE)
players$result <- factor(players$result, levels = c("Lost", "Won"))
str(players)

# View the split of categorical variables to examine balance
table(players$hand)          # 835 right-handed, 143 left-handed
table(players$result)        # 489 lost, 489 won (balanced)
table(players$round_group)   # 498 first-round rows down to 52 for QF+

summary(players)


# ---- Step 10: Linearity --------------------------------------------
# ASSUMPTION: aces is linearly related to the independent variables.

model_variables <- players[, c("aces", "height", "age", "rank",
                               "service_points", "hand", "result", "round_group")]

# Correlation chart of every variable against every other.
# Be careful with the factor variables – R shows them as numbers.
pairs.panels(model_variables,
             smooth = FALSE,      # If TRUE, draws loess smooths
             scale = FALSE,       # If TRUE, scales the correlation text font
             density = TRUE,      # If TRUE, adds density plots and histograms
             ellipses = FALSE,    # If TRUE, draws ellipses
             method = "pearson",  # Correlation method (also "spearman" or "kendall")
             pch = 21,            # pch symbol
             lm = FALSE,          # If TRUE, plots linear fit rather than the LOESS (smoothed) fit
             cor = TRUE,          # If TRUE, reports correlations
             jiggle = FALSE,      # If TRUE, data points are jittered
             factor = 2,          # Jittering factor
             hist.col = 4,        # Histograms color
             stars = TRUE,        # If TRUE, adds significance level with stars
             ci = TRUE)           # If TRUE, adds confidence intervals

# Scatter plots with a smoothed line for each numeric variable.
# A steadily rising or falling line suggests a linear relationship.
scatter.smooth(x = players$height,
               y = players$aces,
               xlab = "Height (cm)",
               ylab = "Aces", main = "Correlation of aces ~ height")

scatter.smooth(x = players$age,
               y = players$aces,
               xlab = "Age (years)",
               ylab = "Aces", main = "Correlation of aces ~ age")

scatter.smooth(x = players$rank,
               y = players$aces,
               xlab = "ATP ranking",
               ylab = "Aces", main = "Correlation of aces ~ rank")

scatter.smooth(x = players$service_points,
               y = players$aces,
               xlab = "Service points played",
               ylab = "Aces", main = "Correlation of aces ~ service points")

# Box plots for the categorical variables
boxplot(aces ~ hand,        data = players, col = "blue", main = "Aces by playing hand", xlab = "Hand",   ylab = "Aces")
boxplot(aces ~ result,      data = players, col = "blue", main = "Aces by match result", xlab = "Result", ylab = "Aces")
boxplot(aces ~ round_group, data = players, col = "blue", main = "Aces by round",        xlab = "Round",  ylab = "Aces")

# Correlation between aces and each numeric variable.
# Between -0.2 and 0.2 = weak, the variable explains little of aces
paste("Correlation for aces and height: ",         round(cor(players$aces, players$height), 2))
paste("Correlation for aces and age: ",            round(cor(players$aces, players$age), 2))
paste("Correlation for aces and rank: ",           round(cor(players$aces, players$rank), 2))
paste("Correlation for aces and service points: ", round(cor(players$aces, players$service_points), 2))

# RESULT
# height = 0.51 (moderate), service points = 0.37 (weaker),
# age = -0.02 and rank = -0.03 (very low).
# Height agrees with CA2 Q1 (Spearman rho = 0.53).
# Age and rank are kept for the first model and judged on their p-values.

# Mean aces in each category
tapply(players$aces, players$hand, mean)          # Right 10.8, Left 9.4
tapply(players$aces, players$result, mean)        # Lost 9.1, Won 12.1
tapply(players$aces, players$round_group, mean)   # 9.8 to 11.8, no clear pattern


# ---- Step 11: Outliers ---------------------------------------------
# Any value more than 1.5 x IQR outside the box is an outlier.
# Outliers can pull the line of best fit towards them.
boxplot(players$aces,           main = "Aces")
boxplot(players$height,         main = "Height")
boxplot(players$age,            main = "Age")
boxplot(players$rank,           main = "Rank")
boxplot(players$service_points, main = "Service points")

# Use boxplot.stats() to list the outliers
# Aces – 29 outliers, all 28 aces or more
outlier_values <- boxplot.stats(players$aces)$out
paste("Aces outliers: ", paste(sort(outlier_values), collapse = ", "))

# Height – 170cm, 208cm and 211cm
outlier_values <- boxplot.stats(players$height)$out
paste("Height outliers: ", paste(sort(unique(outlier_values)), collapse = ", "))

# Age – 37.8 to 40.7 years
outlier_values <- boxplot.stats(players$age)$out
paste("Age outliers: ", paste(sort(unique(outlier_values)), collapse = ", "))

# Rank – 55 outliers (counted rather than listed)
outlier_values <- boxplot.stats(players$rank)$out
paste("Number of rank outliers: ", length(outlier_values))

# Service points – 1 outlier (211)
outlier_values <- boxplot.stats(players$service_points)$out
paste("Service points outliers: ", paste(sort(outlier_values), collapse = ", "))
players[players$service_points %in% outlier_values, c("year", "player", "aces", "service_points", "result")]

# DECISION: all outliers are KEPT
# The unfinished matches (the only errors) were removed in step 3.
# 211 service points is David Goffin's five-set win in 2022, a real match.
# The height, age and rank outliers are real players – removing them would
# remove the tallest servers, who matter most to the research question.
# The skew in aces is dealt with by transformation in step 14.
nrow(players)    # still 978 rows


# ---- Step 12: Normality --------------------------------------------
# The regression assumption is that the RESIDUALS are normal (step 16).
# Each variable is still checked, because a very
# skewed variable – especially aces – can cause non-normal residuals.
#
# skewness of < -1 or > 1 = highly skewed
# -1 to -0.5 and 0.5 to 1 = moderately skewed
# Skewness of -0.5 to 0.5 = approx symmetrical

plot(density(players$aces),
     main = "Density plot : Aces",
     ylab = "Frequency", xlab = "Aces",
     sub = paste("Skewness : ", round(e1071::skewness(players$aces), 2)))
polygon(density(players$aces), col = "red")

plot(density(players$height),
     main = "Density plot : Height",
     ylab = "Frequency", xlab = "Height",
     sub = paste("Skewness : ", round(e1071::skewness(players$height), 2)))
polygon(density(players$height), col = "red")

plot(density(players$age),
     main = "Density plot : Age",
     ylab = "Frequency", xlab = "Age",
     sub = paste("Skewness : ", round(e1071::skewness(players$age), 2)))
polygon(density(players$age), col = "red")

plot(density(players$rank),
     main = "Density plot : Rank",
     ylab = "Frequency", xlab = "Rank",
     sub = paste("Skewness : ", round(e1071::skewness(players$rank), 2)))
polygon(density(players$rank), col = "red")

plot(density(players$service_points),
     main = "Density plot : Service points",
     ylab = "Frequency", xlab = "Service points",
     sub = paste("Skewness : ", round(e1071::skewness(players$service_points), 2)))
polygon(density(players$service_points), col = "red")

paste("Skewness for aces : ",           round(e1071::skewness(players$aces), 2))
paste("Skewness for height : ",         round(e1071::skewness(players$height), 2))
paste("Skewness for age : ",            round(e1071::skewness(players$age), 2))
paste("Skewness for rank : ",           round(e1071::skewness(players$rank), 2))
paste("Skewness for service points : ", round(e1071::skewness(players$service_points), 2))

# RESULT
# aces 1.35 => highly skewed
# height 0.30 => approx symmetrical
# age 0.61 => moderately skewed
# rank 3.63 => highly skewed
# service points 0.45 => approx symmetrical

# Q-Q plots
qqnorm(players$aces,           main = "QQ plot : Aces");           qqline(players$aces, col = "red")
qqnorm(players$height,         main = "QQ plot : Height");         qqline(players$height, col = "red")
qqnorm(players$age,            main = "QQ plot : Age");            qqline(players$age, col = "red")
qqnorm(players$rank,           main = "QQ plot : Rank");           qqline(players$rank, col = "red")
qqnorm(players$service_points, main = "QQ plot : Service points"); qqline(players$service_points, col = "red")

# Shapiro-Wilk tests
# H0: Data is normally distributed
# H1: Data is not normally distributed
shapiro.test(players$aces)            # p < 2.2e-16 -> reject H0 -> not normal
shapiro.test(players$height)          # p = 1.25e-12 -> not normal
shapiro.test(players$age)             # p = 1.25e-14 -> not normal
shapiro.test(players$rank)            # p < 2.2e-16 -> not normal
shapiro.test(players$service_points)  # p = 1.15e-12 -> not normal

# Every variable fails, but with nearly 1,000 rows Shapiro-Wilk picks up
# very small differences, so the skewness values are also used.
# Aces, the dependent variable, is highly skewed: most players serve
# 5 to 14 aces but a few serve 30 to 54.


# ---- Step 13: Initial build of the MLR model -----------------------
# Dummy variables created automatically by R
model_1 <- lm(aces ~
                height +
                age +
                rank +
                service_points +
                hand +
                result +
                round_group,
              data = players)

model_1
summary(model_1)

# For each variable:
# H0: the coefficient is zero (no effect on aces)
# H1: the coefficient is not zero
# p <= 0.05 -> reject H0 -> the variable is significant
#
# RESULT
# height, service_points and result are significant (p < 0.001)
# age (0.42), rank (0.19) and hand (0.69) are not
# Only 1 of the 4 round dummies is below 0.05 (QF+, p = 0.025)
# Adjusted R-squared = 0.404

# H0: Residuals are normally distributed
shapiro.test(residuals(model_1))
paste("Skewness of model_1 residuals : ", round(e1071::skewness(residuals(model_1)), 2))
# p = 1.22e-13, skewness 0.79 -> reject H0 -> residuals not normal
# The skewed dependent variable is the cause, so aces is transformed.


# ---- Step 14: Box-Cox transformation of aces -----------------------
# Box-Cox finds the lambda that makes a variable closest to normal:
#      (y^lambda - 1) / lambda
# It only works on positive values. 14 rows have 0 aces, so 1 is added
# to aces first.
sum(players$aces == 0)

box_cox_transform <- boxcox(lm((aces + 1) ~
                                 height +
                                 age +
                                 rank +
                                 service_points +
                                 hand +
                                 result +
                                 round_group,
                               data = players))

# Pick the lambda at the top of the curve
lamda <- box_cox_transform$x[which.max(box_cox_transform$y)]
lamda                                      # 0.34

normalised_aces <- ((players$aces + 1)^lamda - 1) / lamda

# Histograms before and after
hist(players$aces,     col = "blue", main = "Aces",               xlab = "Aces")
hist(normalised_aces,  col = "blue", main = "Aces after Box-Cox", xlab = "Transformed aces")

paste("Skewness for transformed aces : ", round(e1071::skewness(normalised_aces), 2))
shapiro.test(normalised_aces)

# RESULT
# Skewness 1.35 -> 0.06 (approx symmetrical)
# Shapiro-Wilk p < 2.2e-16 -> 0.0019, a large improvement

players$aces_new <- normalised_aces

# Rank is also skewed but is left as it is: it has almost no relationship
# with aces (r = -0.03) and is judged on its p-value in step 15.


# ---- Step 15: Rebuild the model with the transformed aces ----------
model_2 <- lm(aces_new ~
                height +
                age +
                rank +
                service_points +
                hand +
                result +
                round_group,
              data = players)

summary(model_2)

# RESULT
# height, service_points and result are significant (p < 0.001)
# age (0.079), rank (0.30), hand (0.95) and all 4 round dummies
# (QF+ closest at 0.055) are not -> fail to reject H0 -> removed.
# Hand not being significant agrees with CA2 Q3.
# Adjusted R-squared = 0.416

# model_3 – only the significant variables
model_3 <- lm(aces_new ~
                height +
                service_points +
                result,
              data = players)

model_3
summary(model_3)

# RESULT
# All three variables p < 2e-16 -> reject H0 for each
# Adjusted R-squared = 0.414 – almost the same as model_2 with 4 fewer
# variables. The model explains about 42% of the variation in the
# TRANSFORMED aces.
#
# aces_new = -17.215 + 0.0995 * height + 0.0156 * service_points + 0.6122 * Won

# 95% confidence intervals – none include zero
confint(model_3)

# AIC and BIC – the model with the LOWEST score is preferred.
# model_1 uses aces rather than aces_new, so it cannot be compared.
paste("AIC of model_1", AIC(model_1))
paste("AIC of model_2", AIC(model_2))
paste("AIC of model_3", AIC(model_3))
paste("BIC of model_1", BIC(model_1))
paste("BIC of model_2", BIC(model_2))
paste("BIC of model_3", BIC(model_3))

# RESULT
#            AIC      BIC
# model_2   2956.1   3014.7
# model_3   2952.3   2976.7
# DECISION: model_3 has the lowest AIC and BIC -> final model


# ---- Step 16: Check the model assumptions --------------------------
# Diagnostic plots
plot(model_3, which = 1)   # Residuals vs Fitted   – linearity (flat red line)
plot(model_3, which = 2)   # Q-Q Residuals         – normality (points on the line)
plot(model_3, which = 3)   # Scale-Location        – constant variance
plot(model_3, which = 5)   # Residuals vs Leverage – influential rows

## residuals normally distributed
# H0: Residuals are normally distributed
shapiro.test(residuals(model_3))
paste("Skewness of model_3 residuals : ", round(e1071::skewness(residuals(model_3)), 2))
# p = 0.025, skewness -0.19 -> just under 0.05, but a large improvement
# on model_1 (1.22e-13) and close to normal on the plots

hist(residuals(model_3), col = "blue", main = "Residuals of model_3", xlab = "Residual")
qqnorm(residuals(model_3), main = "QQ plot : residuals of model_3"); qqline(residuals(model_3), col = "red")

## residuals different from zero
# H0: Mean of residuals equal to zero
t.test(residuals(model_3), mu = 0)       # p = 1 -> fail to reject H0

# randomness of residuals (autocorrelation) -- Durbin-Watson test
# H0: there is no autocorrelation (DW close to 2)
dwtest(model_3)                          # DW = 1.92, p = 0.101 -> fail to reject H0

# Check for multi-collinearity
# VIF close to 1 = no correlation between predictors; above 5 is a concern
v1 <- vif(model_3)
v1                                       # height 1.02, service_points 1.01, result 1.02

# LIMITATION: 978 rows come from only 237 players, so rows from the same
# player are not fully independent. Durbin-Watson does not pick this up.
length(unique(players$player))
nrow(players)


# ---- Step 17: Model validation – train and test split --------------
# 80% training data to build the model, 20% testing data to test it.
# The split is done by MATCH, so the winner and loser of a match are
# always on the same side and the model never sees half of a test match.

# set.seed() so the same matches are picked every run
set.seed(1)

# Choose 80% of the 489 match numbers
match_numbers <- unique(players$match_id)
training_matches <- sample(match_numbers, 0.8 * length(match_numbers))

# model training data
training_data <- subset(players, match_id %in% training_matches)
nrow(training_data)    # 782 rows (391 matches)

# test data
testing_data <- subset(players, !(match_id %in% training_matches))
nrow(testing_data)     # 196 rows (98 matches)

# Build the model on training data
mlr_model_1 <- lm(aces_new ~
                    height +
                    service_points +
                    result,
                  data = training_data)

mlr_model_1
summary(mlr_model_1)
# All p < 0.001, adjusted R-squared = 0.409 – close to model_3, so stable

# predict aces from testing data, then reverse the Box-Cox formula
# to turn the prediction back into a number of aces:
#   aces = (lamda * aces_new + 1)^(1 / lamda) - 1
predicted_new  <- predict(mlr_model_1, testing_data)
aces_predicted <- (lamda * predicted_new + 1)^(1 / lamda) - 1

# make actuals_predicteds dataframe
actuals_preds <- data.frame(cbind(actuals   = testing_data$aces,
                                  predicted = aces_predicted))
head(actuals_preds)

# correlation accuracy
correlation_accuracy <- cor(actuals_preds)
correlation_accuracy                 # 0.669

# Min - max accuracy (1 = perfect)
min_max_accuracy <- mean(apply(actuals_preds, 1, min) / apply(actuals_preds, 1, max))
min_max_accuracy                     # 0.653

# MAPE -- Mean Absolute Percentage Error
# Cannot divide by zero, so the 5 rows with 0 aces are left out
sum(actuals_preds$actuals == 0)
non_zero <- subset(actuals_preds, actuals > 0)
mape <- mean(abs(non_zero$predicted - non_zero$actuals) / non_zero$actuals)
mape                                 # 0.729 – high because small counts give large % errors

# MSE -- Mean Squared Error (in squared aces)
# RMSE (its square root) puts the error back into aces
mse <- mean((actuals_preds$predicted - actuals_preds$actuals)^2)
mse                                  # 33.4
sqrt(mse)                            # 5.8 – a typical prediction is about 6 aces out

# Predicted against actual aces – a perfect model would sit on the red line
plot(actuals_preds$actuals, actuals_preds$predicted, pch = 19, col = "blue",
     xlab = "Actual aces", ylab = "Predicted aces",
     main = "Test data : predicted ~ actual aces (80:20 split)")
abline(0, 1, col = "red")
# The model under-predicts the biggest serving days

# 70:30 split – checks the result does not depend on one split
set.seed(1)
training_matches_2 <- sample(match_numbers, 0.7 * length(match_numbers))
training_data_2 <- subset(players, match_id %in% training_matches_2)
testing_data_2  <- subset(players, !(match_id %in% training_matches_2))

mlr_model_2 <- lm(aces_new ~
                    height +
                    service_points +
                    result,
                  data = training_data_2)
summary(mlr_model_2)

predicted_new_2 <- predict(mlr_model_2, testing_data_2)
actuals_preds_2 <- data.frame(cbind(actuals   = testing_data_2$aces,
                                    predicted = (lamda * predicted_new_2 + 1)^(1 / lamda) - 1))

cor(actuals_preds_2)                                                       # 0.624
mean(apply(actuals_preds_2, 1, min) / apply(actuals_preds_2, 1, max))      # 0.655
non_zero_2 <- subset(actuals_preds_2, actuals > 0)
mean(abs(non_zero_2$predicted - non_zero_2$actuals) / non_zero_2$actuals)  # 0.693
mean((actuals_preds_2$predicted - actuals_preds_2$actuals)^2)              # 32.3
sqrt(mean((actuals_preds_2$predicted - actuals_preds_2$actuals)^2))        # 5.7
# Similar to the 80:20 split


# ---- Step 18: Model forecasting ------------------------------------
# Six devised tests. Each changes one variable from the typical (median)
# values of 188cm and 113 service points.
# These are illustrative checks; the validation is step 17.
# Service points and the result are only known after a match, so these
# estimate aces for a completed match rather than forecasting before play.
summary(players[, c("height", "service_points")])

forecast_tests <- data.frame(
  test           = c("1 Typical player, won",
                     "2 Typical player, lost",
                     "3 Short player (170cm), won",
                     "4 Tall player (198cm), won",
                     "5 Short match (80 points), won",
                     "6 Long match (160 points), won"),
  height         = c(188, 188, 170, 198, 188, 188),
  service_points = c(113, 113, 113, 113,  80, 160),
  result         = factor(c("Won", "Lost", "Won", "Won", "Won", "Won"),
                          levels = c("Lost", "Won")))

# Predict, convert back to aces and round to 1 decimal place
forecast_new <- predict(mlr_model_1, forecast_tests)
forecast_tests$predicted_aces <- round((lamda * forecast_new + 1)^(1 / lamda) - 1, 1)
forecast_tests

# Compare each forecast with the real average for similar players
# Tests 1 and 2 – typical height (186-190cm), won and lost
mean(players$aces[players$height >= 186 & players$height <= 190 & players$result == "Won"])
mean(players$aces[players$height >= 186 & players$height <= 190 & players$result == "Lost"])

# Test 3 – short players (175cm or less)
mean(players$aces[players$height <= 175])

# Test 4 – tall players (196-200cm)
mean(players$aces[players$height >= 196 & players$height <= 200])

# Test 5 – short matches (90 service points or fewer)
mean(players$aces[players$service_points <= 90])

# Test 6 – long matches (150 service points or more)
mean(players$aces[players$service_points >= 150])

# RESULT (predicted aces, then real average)
# 1 Typical player, won           10.9   11.7
# 2 Typical player, lost           7.9    8.5
# 3 Short player (170cm), won      4.0    4.4
# 4 Tall player (198cm), won      16.6   16.0
# 5 Short match (80 points), won   8.6    7.8
# 6 Long match (160 points), won  14.7   15.1
#
# CONCLUSION: every forecast moves in the expected direction and is close
# to the real average for similar players. Height, service points and the
# match result predict aces well on average, but a single match can be
# about 6 aces out (RMSE 5.8). Age, ranking, hand and round did not help,
# so the research question is partly supported.

# save the model and the cleaned data
saveRDS(mlr_model_1, "./aces_model.rds")
write.csv(players, "wimbledon_player_matches.csv", row.names = FALSE)
