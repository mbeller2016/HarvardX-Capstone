#Setup code pre-provided
##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#-Start Analysis here

#--Preprocessing
memory.limit(size=10000)

#--Split into training and testing sets
set.seed(99, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
#edx_test_index <- createDataPartition(
#  y = edx_preprocess$rating, times = 1, p = 0.2, list = FALSE)
#edx_train <- edx_preprocess[-edx_test_index,]
#edx_test <- edx_preprocess[edx_test_index,]
edx_test_index <- createDataPartition(
  y = edx$rating, times = 1, p = 0.2, list = FALSE)
edx_train <- edx[-edx_test_index,]
edx_test <- edx[edx_test_index,]

#-Define loss function (RMSE)
RMSE_f <- function(predicted,actual){
  RMSE <- sqrt(mean((actual - predicted)^2))
}

# Split title field into actual title and release year
#Took multiple iterations to get the regex logic right
#pptitle <- str_sub(edx_train$title, start = 1, 
#                   end = (str_locate(edx_train$title,"\\((19|20)")[,1])-1)
#ppyear <- str_sub(edx_train$title, start = str_locate(edx_train$title,"\\((19|20)")[,1])
#ppyear <- str_sub(ppyear,start = 2, end = nchar(ppyear)-1)

#Check for problems with the year
#length(which(is.na(as.numeric(ppyear))))

# Add the actual title and release year to the edx data frame
#edx_train <- edx_train %>% mutate(
#  nametitle = pptitle,
#  relyear = as.numeric(ppyear))

#rm(pptitle)   #cleanup
#rm(ppyear)

# Create a list of all single genres (from movies that have only one genre)
genre_tab <- edx_train %>% filter(str_detect(genres,"\\|") == FALSE) %>%
                         group_by(genres) %>%
                         summarize(count = n())
genre_list <- genre_tab$genres

# Verify that there are no multiple-genre movies that contain genres that weren't in
# the single-genre list
#all_genre <- edx_train %>% group_by(genres) %>%
#  summarize(count = n())

#ind <- seq(1,nrow(all_genre))
#for (x in ind) {
#  gcheck <- str_detect(all_genre$genres[x],genre_list)
#  if(mean(gcheck) == 0)
#    {print(all_genre$genres[x])}
#  }

# Build movie/genre table
build_movie_genres <- function (movietab) {
  movgen <- data.frame(movieId = 0, genre = "Dummy")
  for (x in 1:nrow(movietab)) {
    mg <- str_split(movietab$genres[x],"\\|",simplify = TRUE)
    mrow <- ""
    for (y in 1:dim(mg)[2]) {
      mrow <- c(mrow,mg[y])
    }
    mtab <- data.frame(movieId = movietab$movieId[x],genre = mrow)
    mtab <- mtab[-1,]
    movgen <- bind_rows(movgen,mtab)
  }
  movgen
}

moviesum <- edx_train %>%
              group_by(movieId, genres) %>%
              summarize()
movie_genres <- build_movie_genres(moviesum)
movie_genres <- movie_genres[-1,]
#--Preprocessing completed


#- Function for correctly rounding a predicted value
# If the value is within .4 of a whole-number rating, use the whole-number rating.
# Otherwise, use the nearest half-number rating
#
#pred_round <- function (prediction) {
#  preddec <- prediction - floor(prediction)
#  predcalc <- case_when(
#    prediction < 0.4 ~ 0.5,
#    preddec < 0.4 ~ floor(prediction),
#    preddec >= 0.4 & preddec <= 0.6 ~ floor(prediction) + 0.5,
#    preddec > 0.6 ~ ceiling(prediction),
#    TRUE ~ prediction)
#  predcalc
#}

#rm(edx_preprocess)   #cleanup

#-- Begin data analysis

#-Rating analysis
#rating_sum <- edx_train %>% group_by(rating) %>% summarize(count=n())
#rating_sum <- rating_sum %>% mutate(prop = count/nrow(edx_train))
#sum(rating_sum$prop[(rating_sum$rating-floor(rating_sum$rating)) != 0])

#-Overall mean rating (default)
overallmean <- mean(edx_train$rating)

#-Movie analysis
movie_effects <- edx_train %>%
  group_by(movieId) %>%
  summarize(mcount = n(), meanrating = mean(rating)) %>%
  ungroup() %>%
  mutate(moviediff = meanrating - overallmean)
#movie_effects %>% filter(mcount < 100) %>%
#  ggplot(aes(mcount)) + geom_histogram(binwidth=5)

#-Add year effect data to movie table (better performance)
#movie_effects <- left_join(movie_effects, year_effect, by="relyear")
#movie_effects$ydiff[which(is.na(movie_effects$ydiff))] <- 0

#-User analysis
user_effects <- edx_train %>%
                  group_by(userId) %>%
                  summarize(ucount = n(),meanrating = mean(rating)) %>%
                  ungroup() %>%
                  mutate(userdiff = meanrating - overallmean)
user_effects %>% ggplot(aes(ucount)) + geom_histogram(binwidth = 20) +
  scale_y_log10() + xlim(0,4000) + xlab("User Review Count")

# Compare means of number of user reviews
user_effectsq <- quantile(user_effects$ucount)
lowuser <- user_effects %>% filter(ucount <= user_effectsq[2])
meduser <- user_effects %>% filter(ucount > user_effectsq[2] & ucount <= user_effectsq[3])
highuser <- user_effects %>% filter(ucount > user_effectsq[3] & ucount <= user_effectsq[4])
vhighuser <- user_effects %>% filter(ucount > user_effectsq[4])
user_ratingsq <- data.frame(quantile = c("Low","Medium","High","Very High"),
                            meanrating = c(mean(lowuser$meanrating),
                                           mean(meduser$meanrating),
                                           mean(highuser$meanrating),
                                           mean(vhighuser$meanrating)))

#-Genre analysis
#Get means by genre
genre_means <- inner_join(movie_genres, edx_train,by="movieId") %>%
                 group_by(genre) %>%
                 summarize(gcount = n(), grating = mean(rating))

#Get movie means by averaging its genres' means
moviegenre_means <- inner_join(movie_genres, genre_means, by = "genre") %>%
                    group_by(movieId) %>%
                    summarize(mgcount = n(), mgrating = mean(grating))

#Add movie genre effects to movie_effects table
movie_effects <- inner_join(movie_effects,moviegenre_means,by="movieId") %>%
                   mutate(gdiff = mgrating - overallmean)

#Now calculate movie quality in movie effects table
#movie_effects <- movie_effects %>%
#                   mutate(qdiff = moviediff - ydiff - gdiff)

#Does this even matter?????  Isn't it really about user effects????
# May matter for new movies for which we don't have a quality rating

#-Build user/genre effects table
usergenre_effects <- inner_join(edx_train, movie_genres, by="movieId") %>%
                     group_by(userId,genre) %>%
                     summarize(ucount = n(), ugrating = mean(rating)) %>%
                     ungroup()
#Link to genre means to calculate difference by genre
usergenre_effects <- inner_join(usergenre_effects, genre_means, by="genre") %>%
                     mutate(ugdiff = ugrating - grating)

#Analysis
usergenre_analysis <- usergenre_effects %>%
                        group_by(genre) %>%
                        summarize(gcount = n(), gdiff = mean(ugdiff),gsd = sd(ugdiff),
                                  gstderr = gsd/sqrt(gcount))
#                                  gerrlow = gdiff - gstderr,
#                                  gerrhigh = gdiff + gstderr)


usergenre_analysis %>% filter(genre != "(no genres listed)") %>%
                       ggplot(aes(genre,gdiff)) + geom_point(color="blue") +
                         theme(axis.text.x = element_text(angle=90)) +
                         geom_errorbar(aes(ymin=gdiff-gsd,
                                           ymax=gdiff+gsd,),color="red")

#--Predictions
# Goal is the following model
# Prediction = overall_mean + movie_effect + user_effect
# The total "movie effect" as quantified in course 8 is a composite of the following:
# 1. The overall quality of the movie
# 2. The year in which it was released
# 3. The genre of the movie
#
# Any difference between the total movie effect and the overall mean is due to the
#   total user effect, which is a composite of the following:
# 1. The overall "pickiness" of the user
# 2. The user's preference/lack thereof for a particular genre
#
# So a more detailed statement of the prediction equation is:
# Prediction = overall_mean + movie_quality_effect + releaseyear_effect + genre_effect +
#                userpickiness_effect + user/genre_effect
#
# We know that the total movie effect for a given prediction is the difference between
# the movie's mean rating and the overall mean.
# We know that the total user effect for a given prediction is the RMSE from predicting
# a rating using the overall mean and total movie effect.
# The question for both is how do we ultimately break these down into their respective
# components (3 for movie, 2 for user), so that we can use these to predict more 
# accurately at the component level.
#
# For movie effects, we can calculate the effect by release year, and the effect by
# genre.  The quality effect should be the difference between these two.

#-Base prediction A (Overall mean predicts all movie/user combinations)
predA <- overallmean  
predA_RMSE <- RMSE_f(predA,edx_test$rating)
pred_results <- data.frame(prednum = "A", predtype = "Overall Mean", RMSE = predA_RMSE)

#-Base prediction B (Mean rating of movie predicts ratings for all users)
predB <- left_join(edx_test, movie_effects, by = "movieId")
predB$moviediff[which(is.na(predB$moviediff))] <- 0
predB <- predB %>% mutate(predrating = overallmean + moviediff)

predB_RMSE <- RMSE_f(predB$predrating,predB$rating)
pred_results <- bind_rows(pred_results,
                  data_frame(prednum = "B", predtype = "Movie Mean", RMSE = predB_RMSE))
rm(predB)    #cleanup

#-Base prediction C (Mean rating of user predicts ratings for all movies)
predC <- left_join(edx_test, user_effects, by = "userId")
predC$userdiff[which(is.na(predC$userdiff))] <- 0
predC <- predC %>% mutate(predrating = overallmean + userdiff)

predC_RMSE <- RMSE_f(predC$predrating,predC$rating)
pred_results <- bind_rows(pred_results,
                  data_frame(prednum = "C", predtype = "User Mean", RMSE = predC_RMSE))
rm(predC)    #cleanup

#Prediction 4: Movie and user effect breakdown 
#Movie effect breakdown may be useful for predicting movies that aren't in the train set
# Compare residuals from movie prediction with amounts of user effects

#Get residuals from movie effects
resid_effect <- left_join(edx_train, movie_effects,by="movieId") %>%
                  mutate(resid = rating - overallmean - moviediff)
resid_effect <- left_join(resid_effect, movie_genres, by="movieId")
resid_effect <- left_join(resid_effect, usergenre_effects, by=c("userId","genre"))
#resid_effect$ugdiff[which(is.na(resid_effect$ugdiff))] <- 0
resid_effect <- resid_effect %>%
                  group_by(userId,movieId,rating,resid) %>%
                  summarize(ugdiff = mean(ugdiff)) %>%
                  ungroup()
userpicky_effects <- resid_effect %>%
                       mutate(updiff = ifelse((rating + ugdiff - resid > 5) |
                                              (rating + ugdiff - resid < 0.5),
                                               0, resid - ugdiff)) %>%
                       group_by(userId) %>%
                       summarize(updiff = mean(updiff)) %>%
                       ungroup()
#The above mutate accounts for ratings above 5 or below 0.5

#Adjust for users who give everything the same rating?

#Predict
pred1 <- left_join(edx_test, movie_effects, by = "movieId")
pred1$moviediff[which(is.na(pred1$moviediff))] <- 0
pred1 <- left_join(pred1,userpicky_effects,by="userId")
pred1$updiff[which(is.na(pred1$updiff))] <- 0
pred1 <- left_join(pred1, movie_genres, by="movieId")
pred1 <- left_join(pred1, usergenre_effects, by=c("userId","genre"))
pred1$ugdiff[which(is.na(pred1$ugdiff))] <- 0
pred1 <- pred1 %>%
           group_by(userId,movieId,rating,moviediff,updiff) %>%
           summarize(ugdiff = mean(ugdiff)) %>%
           ungroup()
pred1 <- pred1 %>% mutate(predrating = overallmean + moviediff + updiff + ugdiff)
pred1$predrating[which(pred1$predrating>5)] <- 5
pred1$predrating[which(pred1$predrating<0.5)] <- 0.5
pred1 <- pred1 %>% mutate(RMSE = (rating - predrating)^2)

pred1_RMSE <- RMSE_f(pred1$predrating,pred1$rating)
pred_results <- bind_rows(pred_results,
                          data_frame(prednum = "1", predtype = "OMR + ME + UGE + UPE",
                          RMSE = pred1_RMSE))
rm(pred1)    #cleanup

#Check missing
missing <- anti_join(edx_test,movie_effects,by="movieId")
print(pred4[pred4$movieId %in% missing$movieId,],n=30)

#-Regularization
# Look at penalizing more extreme estimates.
# Apply to movie effects, user effects, and user genre effects

#Regularize movie effects
kfold <- 10
movie_cvs <- data.frame(fold = 0, bestlambda = 0, bestRMSE = 0)
for (K in 1:kfold) {
  #Create a data partition of the correct size
  set.seed(200+K, sample.kind="Rounding")
  cv_test_index <- createDataPartition(
    y = edx_train$rating, times = 1, p = 0.2, list = FALSE)
  edx_train_k <- edx_train[-cv_test_index,]
  edx_test_k <- edx_train[cv_test_index,]

  #Look at potential lambda penalty terms
#  m_lambda <- seq(1,10,1)
#  m_lambda <- seq(1.5,3.5,0.2)
  m_lambda <- seq (2,2.6,0.1)
  movie_cv <- data.frame(fold = 0, lambda = 0, RMSE = 99999)
  for (L in m_lambda) {
    movie_r <- edx_train_k %>%
                 group_by(movieId) %>%
                 summarize(mdiffr = sum(rating - overallmean)/(n()+L)) %>%
                 ungroup()
    #Use inner join instead of left join --> if a movie doesn't exist in the kfold train,
    #we don't want to consider it for lambda testing
    predk <- inner_join(edx_test_k, movie_r, by = "movieId")
    predk$mdiffr[which(is.na(predk$mdiffr))] <- 0
    predk <- predk %>% mutate(predrating = overallmean + mdiffr)
    predk_RMSE <- RMSE_f(predk$predrating,predk$rating)
    movie_cv <- bind_rows(movie_cv, data_frame(fold = K, lambda = L, 
                                               RMSE = predk_RMSE))
  }
  movie_cvs <- bind_rows(movie_cvs, data_frame(fold = K,
                 bestlambda = movie_cv$lambda[which.min(movie_cv$RMSE)],
                 bestRMSE = movie_cv$RMSE[which.min(movie_cv$RMSE)]))
}
movie_cvs <- movie_cvs[-1,]

# Regularize movie effects using best movie lambda of 2.3
m_lambda <- 2.3
movie_effectsr <- edx_train %>%
  group_by(movieId) %>%
  summarize(mdiffr = sum(rating - overallmean)/(n()+m_lambda)) %>%
  ungroup()
movie_effects <- inner_join(movie_effects,movie_effectsr,by="movieId")
rm(movie_effectsr)  #cleanup

#-Prediction 2 (Full, regularized movie)
pred2 <- left_join(edx_test, movie_effects, by = "movieId")
pred2$mdiffr[which(is.na(pred2$mdiffr))] <- 0
pred2 <- left_join(pred2,userpicky_effects,by="userId")
pred2$updiff[which(is.na(pred2$updiff))] <- 0
pred2 <- left_join(pred2, movie_genres, by="movieId")
pred2 <- left_join(pred2, usergenre_effects, by=c("userId","genre"))
pred2$ugdiff[which(is.na(pred2$ugdiff))] <- 0
pred2 <- pred2 %>%
  group_by(userId,movieId,rating,mdiffr,updiff) %>%
  summarize(ugdiff = mean(ugdiff)) %>%
  ungroup()
pred2 <- pred2 %>% mutate(predrating = overallmean + mdiffr + updiff + ugdiff)
pred2$predrating[which(pred2$predrating>5)] <- 5
pred2$predrating[which(pred2$predrating<0.5)] <- 0.5

pred2_RMSE <- RMSE_f(pred2$predrating,pred2$rating)
pred_results <- bind_rows(pred_results,
                          data_frame(prednum = "2", predtype = "OMR + MEr + UGE + UPE",
                                     RMSE = pred2_RMSE))
rm(pred2)    #cleanup

#-Regularize user genre means
#kfold <- 1
#userg_cvs <- data.frame(fold = 0, bestlambda = 0, bestRMSE = 0)
#for (K in 1:kfold) {
#Create a data partition for lambda cross-validation
set.seed(400+K, sample.kind="Rounding")
cv_test_index <- createDataPartition(
  y = edx_train$rating, times = 1, p = 0.2, list = FALSE)
edx_train_k <- edx_train[-cv_test_index,]
edx_test_k <- edx_train[cv_test_index,]
  
#Look at potential lambda penalty terms
#  ug_lambda <- seq(1,10,1)
#  ug_lambda <- seq(3.5,4.5,0.2)
  ug_lambda <- seq(4.2,4.4,0.1)
  userg_cv <- data.frame(fold = 0, lambda = 0, RMSE = 9999)
  for (L in ug_lambda) {
    usergenre_effectsr <- inner_join(edx_train_k, movie_genres, by="movieId")
    usergenre_effectsr <- inner_join(usergenre_effectsr, genre_means, by="genre") %>%
      group_by(userId,genre) %>%
      summarize(ucount = n(), ugdiffr = sum(rating - grating)/(n()+L)) %>%
      ungroup()
    #Use inner join instead of left join --> if a movie doesn't exist in the kfold train,
    #we don't want to consider it for lambda testing
    predk <- inner_join(edx_test_k, movie_genres, by="movieId")
    predk <- inner_join(predk, usergenre_effectsr, by=c("userId","genre"))
    predk <- predk %>%
      group_by(userId,movieId,rating,ugdiffr) %>%
      summarize(ugdiffr = mean(ugdiffr)) %>%
      ungroup()
    predk <- predk %>% mutate(predrating = overallmean + ugdiffr)
    predk_RMSE <- RMSE_f(predk$predrating,predk$rating)
    userg_cv <- bind_rows(userg_cv, data_frame(fold = K, lambda = L, 
                                               RMSE = predk_RMSE))
  }
#  userg_cvs <- bind_rows(userg_cvs, data_frame(fold = K,
#                 bestlambda = userg_cv$lambda[which.min(userg_cv$RMSE)],
#                 bestRMSE = userg_cv$RMSE[which.min(userg_cv$RMSE)]))
#}
#userg_cvs <- userg_cvs[-1,]


#Regularize user/genre effects using best lambda of 4.3
ug_lambda <- 4.3
usergenre_effectsr <- inner_join(edx_train, movie_genres, by="movieId")
usergenre_effectsr <- inner_join(usergenre_effectsr, genre_means, by="genre") %>%
  group_by(userId,genre) %>%
  summarize(ucount = n(), ugdiffr = sum(rating - grating)/(n()+ug_lambda)) %>%
  ungroup()
usergenre_effects <- inner_join(usergenre_effects,usergenre_effectsr,
                                by=c("userId","genre"))
rm(usergenre_effectsr)    #Cleanup

#-Prediction 3: Full, but with regularized movie and user/genre effects
pred3 <- left_join(edx_test, movie_effects, by = "movieId")
pred3$mdiffr[which(is.na(pred3$mdiffr))] <- 0
pred3 <- left_join(pred3,userpicky_effects,by="userId")
pred3$updiff[which(is.na(pred3$updiff))] <- 0
pred3 <- left_join(pred3, movie_genres, by="movieId")
pred3 <- left_join(pred3, usergenre_effects, by=c("userId","genre"))
pred3$ugdiffr[which(is.na(pred3$ugdiffr))] <- 0
pred3 <- pred3 %>%
  group_by(userId,movieId,rating,mdiffr,updiff) %>%
  summarize(ugdiffr = mean(ugdiffr)) %>%
  ungroup()
pred3 <- pred3 %>% mutate(predrating = overallmean + mdiffr + updiff + ugdiffr)
pred3$predrating[which(pred3$predrating>5)] <- 5
pred3$predrating[which(pred3$predrating<0.5)] <- 0.5

pred3_RMSE <- RMSE_f(pred3$predrating,pred3$rating)
pred_results <- bind_rows(pred_results,
                          data_frame(prednum = "3", predtype = "OMR + MEr + UGEr + UP",
                                     RMSE = pred3_RMSE))
rm(pred3)    #cleanup
#pred3check <- pred3 %>% mutate(se = (rating - predrating)^2) %>%
#                   select(movieId,userId,rating,predrating,mdiffr,updiff,ugdiffr,se) %>%
#                   arrange(desc(se))
#pred3q <- quantile(pred3check$se)

#Adjust user pickyness to account for regularized user/genre effects and movie effects
resid_effectr <- left_join(edx_train, movie_effects,by="movieId") %>%
  mutate(residr = rating - overallmean - mdiffr)
resid_effectr <- left_join(resid_effectr, movie_genres, by="movieId")
resid_effectr <- left_join(resid_effectr, usergenre_effects, by=c("userId","genre"))
resid_effectr$ugdiffr[which(is.na(resid_effectr$ugdiffr))] <- 0
resid_effectr <- resid_effectr %>%
                   group_by(userId,movieId,rating,residr) %>%
                   summarize(ugdiffr = mean(ugdiffr)) %>%
                   ungroup()
userpicky_effectsr <- resid_effectr %>%
                   mutate(updiffa = ifelse((rating + ugdiffr - residr > 5) |
                                      (rating + ugdiffr - residr < 0.5),
                                      0, residr - ugdiffr)) %>%
                   group_by(userId) %>%
                   summarize(updiffa = mean(updiffa)) %>%
                   ungroup()
userpicky_effects <- inner_join(userpicky_effects,userpicky_effectsr,by="userId")

#-Prediction 4: Full, but with regularized movie and user/genre effects, and adjusted
#  user pickyness residual to account for regularization of the above
pred4 <- left_join(edx_test, movie_effects, by = "movieId")
pred4$mdiffr[which(is.na(pred4$mdiffr))] <- 0
pred4 <- left_join(pred4,userpicky_effects,by="userId")
pred4$updiffa[which(is.na(pred4$updiffa))] <- 0
pred4 <- left_join(pred4, movie_genres, by="movieId")
pred4 <- left_join(pred4, usergenre_effects, by=c("userId","genre"))
pred4$ugdiffr[which(is.na(pred4$ugdiffr))] <- 0
pred4 <- pred4 %>%
           group_by(userId,movieId,rating,mdiffr,updiffa) %>%
           summarize(ugdiffr = mean(ugdiffr)) %>%
           ungroup()
pred4 <- pred4 %>% mutate(predrating = overallmean + mdiffr + updiffa + ugdiffr)
pred4$predrating[which(pred4$predrating>5)] <- 5
pred4$predrating[which(pred4$predrating<0.5)] <- 0.5

pred4_RMSE <- RMSE_f(pred4$predrating,pred4$rating)
pred_results <- bind_rows(pred_results,
                          data_frame(prednum = "4", predtype = "OMR + MEr + UGEr + UPEa",
                                     RMSE = pred4_RMSE))
rm(pred4)
#pred7check <- pred7 %>% mutate(se = (rating - predrating)^2) %>%
#  select(movieId,userId,rating,predrating,mdiffr,updiffa,ugdiffr,se) %>%
#  arrange(desc(se))
#pred7q <- quantile(pred7check$se)

#missing <- anti_join(edx_test,movie_effects,by="movieId")
#print(pred7check[pred7check$movieId %in% missing$movieId,],n=30)


#-Treat year effect as alternative to overall mean
# Start by calculating average ratings by year
year_effects <- edx_train %>%
                 group_by(relyear) %>%
                 summarize(ycount = n(), ymean = mean(rating))
yearq <- quantile(year_effects$relyear,seq(0,1,0.1))
year_effects <- year_effects %>%
                  mutate(yq = case_when(
                    relyear < yearq[2] ~ 1,
                    relyear < yearq[3] ~ 2,
                    relyear < yearq[4] ~ 3,
                    relyear < yearq[5] ~ 4,
                    relyear < yearq[6] ~ 5,
                    relyear < yearq[7] ~ 6,
                    relyear < yearq[8] ~ 7,
                    relyear < yearq[9] ~ 8,
                    relyear < yearq[10] ~ 9,
                    TRUE ~ 10))
yearq_effects <- year_effects %>%
                  group_by(yq) %>%
                  summarize(yqcount = sum(ycount),
                            yqmean = sum(ymean*ycount)/sum(ycount))
year_effects <- inner_join(year_effects, yearq_effects, by = "yq")


# Treat year effect as alternative to overall mean
movie_effects <- left_join(edx_train,year_effects,by="relyear") %>%
  group_by(movieId) %>%
  summarize(mcount = n(), meanrating = mean(rating), yqmean = mean(yqmean)) %>%
  ungroup()
movie_effects$yqmean[which(is.na(movie_effects$yqmean))] <- overallmean
movie_effects <- movie_effects %>% mutate(moviediff = meanrating - yqmean)

#Add movie genre effects to movie_effects table
movie_effects <- inner_join(movie_effects,moviegenre_means,by="movieId") %>%
  mutate(gdiff = mgrating - yqmean)


# May matter for new movies for which we don't have a quality rating

#users who give everything the same rating????


