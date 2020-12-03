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

##########################################################
# Begin project code by mpbeller
##########################################################

#-Increase memory limit given the size of the data set
memory.limit(size=10000)

##########################################################
# Basic setup
##########################################################

#-Split the edx dataset into training and test partitions
set.seed(99, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
edx_test_index <- createDataPartition(
  y = edx$rating, times = 1, p = 0.2, list = FALSE)
edx_train <- edx[-edx_test_index,]
edx_test <- edx[edx_test_index,]

#-Define loss function (RMSE)
RMSE_f <- function(predicted,actual){
  RMSE <- sqrt(mean((actual - predicted)^2))
}

##########################################################
# Build movie/genre and movie effects table
##########################################################

#-Calculate overall mean rating
overallmean <- mean(edx_train$rating)

#-Build movie effects table
movie_effects <- edx_train %>%
  group_by(movieId) %>%
  summarize(mcount = n(), meanrating = mean(rating)) %>%
  ungroup() %>%
  mutate(moviediff = meanrating - overallmean)

#-Get list of all single genres (from movies that have only one genre)
genre_tab <- edx_train %>% filter(str_detect(genres,"\\|") == FALSE) %>%
  group_by(genres) %>%
  summarize(count = n())
genre_list <- genre_tab$genres

#-Build movie/genre table
build_movie_genres <- function (movietab) {
  movgen <- data.frame(movieId = 0, genre = "Dummy")
  #Loop through list of movies
  for (x in 1:nrow(movietab)) {
    mg <- str_split(movietab$genres[x],"\\|",simplify = TRUE)
    mrow <- ""
    #Loop through genres and create rows
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

#-Get means by genre
genre_means <- inner_join(movie_genres, edx_train,by="movieId") %>%
  group_by(genre) %>%
  summarize(gcount = n(), grating = mean(rating))

#-Get movie means by averaging its genres' means
moviegenre_means <- inner_join(movie_genres, genre_means, by = "genre") %>%
  group_by(movieId) %>%
  summarize(mgcount = n(), mgrating = mean(grating))

#-Add movie genre effects to movie_effects table
movie_effects <- inner_join(movie_effects,moviegenre_means,by="movieId") %>%
  mutate(gdiff = mgrating - overallmean)

#-Regularize movie effects using lambda = 2.3
m_lambda <- 2.3
movie_effectsr <- edx_train %>%
  group_by(movieId) %>%
  summarize(mdiffr = sum(rating - overallmean)/(n()+m_lambda)) %>%
  ungroup()
movie_effects <- inner_join(movie_effects,movie_effectsr,by="movieId")
rm(movie_effectsr)  #cleanup

##########################################################
# Build user/genre effects table
##########################################################

#-Build user/genre effects table
usergenre_effects <- inner_join(edx_train, movie_genres, by="movieId") %>%
  group_by(userId,genre) %>%
  summarize(ucount = n(), ugrating = mean(rating)) %>%
  ungroup()

#-Link to genre means to calculate difference by genre
usergenre_effects <- inner_join(usergenre_effects, genre_means, by="genre") %>%
  mutate(ugdiff = ugrating - grating)

#-Regularize user/genre combinations using lambda = 4.3
ug_lambda <- 4.3
usergenre_effectsr <- inner_join(edx_train, movie_genres, by="movieId")
usergenre_effectsr <- inner_join(usergenre_effectsr, genre_means, by="genre") %>%
  group_by(userId,genre) %>%
  summarize(ucount = n(), ugdiffr = sum(rating - grating)/(n()+ug_lambda)) %>%
  ungroup()
usergenre_effects <- inner_join(usergenre_effects,usergenre_effectsr,
                                by=c("userId","genre"))
rm(usergenre_effectsr)    #Cleanup

##########################################################
# Build user pickiness table
##########################################################

#-Calculate user pickiness from residuals
resid_effectr <- left_join(edx_train, movie_effects,by="movieId") %>%
                 mutate(residr = rating - overallmean - mdiffr)
resid_effectr <- left_join(resid_effectr, movie_genres, by="movieId")
resid_effectr <- left_join(resid_effectr, usergenre_effects, by=c("userId","genre"))
resid_effectr$ugdiffr[which(is.na(resid_effectr$ugdiffr))] <- 0
resid_effectr <- resid_effectr %>%
  group_by(userId,movieId,rating,residr) %>%
  summarize(ugdiffr = mean(ugdiffr)) %>%
  ungroup()
userpicky_effects <- resid_effectr %>%
  mutate(updiffa = ifelse((rating + ugdiffr - residr > 5) |
                            (rating + ugdiffr - residr < 0.5),
                          0, residr - ugdiffr)) %>%
  group_by(userId) %>%
  summarize(updiffa = mean(updiffa)) %>%
  ungroup()

##########################################################
# Make Predictions
##########################################################

#-Get regularized movie effects
pred_val <- left_join(validation, movie_effects, by = "movieId")
#-If a movie from the validation set does not exist in the training set, set its
#   regularized movie effects parameter to zero.
pred_val$mdiffr[which(is.na(pred_val$mdiffr))] <- 0

#-Get adjusted user pickiness effects
pred_val <- left_join(pred_val,userpicky_effects,by="userId")
#-If a user from the test set does not exist in the training set, set its adjusted user 
#  pickiness effects parameter to zero.
pred_val$updiffa[which(is.na(pred_val$updiffa))] <- 0

#-Get movie genres
pred_val <- left_join(pred_val, movie_genres, by="movieId")
#-Get regularized user/genre effects
pred_val <- left_join(pred_val, usergenre_effects, by=c("userId","genre"))
#-If a user/genre combination from the test set does not exist in the training set, set
#   its regularized user/genre effects parameter to zero.
pred_val$ugdiffr[which(is.na(pred_val$ugdiffr))] <- 0

#-Sum up user/genre effects
pred_val <- pred_val %>%
  group_by(userId,movieId,rating,mdiffr,updiffa) %>%
  summarize(ugdiffr = mean(ugdiffr)) %>%
  ungroup()

#-Calculate predicted rating
pred_val <- pred_val %>% mutate(predrating = overallmean + mdiffr + updiffa + ugdiffr)
#-Adjust predictions that fall outside the 0.5 - 5 range
pred_val$predrating[which(pred_val$predrating>5)] <- 5
pred_val$predrating[which(pred_val$predrating<0.5)] <- 0.5

#Calculate RMSE
predval_RMSE <- RMSE_f(pred_val$predrating,pred_val$rating)
predval_RMSE