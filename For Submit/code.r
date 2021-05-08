library(caret)
library(tidyverse)
################################################################################
#  ing. Wojciech Pribula, 2021, wojtek.pribula.cz                              #
#  CONTENT:                                                                    #
#           1. DATA LOADING    - loads data and does some pre-processing       #
#           2. PARAMETER TUNING FUNCTIONS  (run from TRAINING FUNCTION)        #
#           3. TRAINING FUNCTION           (run from BUILD MODEL)              #
#           4. BUILD MODEL     - uses other modules to build actual model      #
#           5. VALIDATE MODEL  - creates final RMSE                            #
#           6. PLOTS           - plots for report                              #
#                                                                              #
################################################################################


################################################################################
#                                                                              #
#                               DATA LOADING                                   #
#                                                                              #
################################################################################

#-------------------------------------------------------------------------------
#           Create edx set, validation set (final hold-out test set)
#-------------------------------------------------------------------------------

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

#-------------------------------------------------------------------------------
#                             Load/Save Data
#-------------------------------------------------------------------------------
# # Save data
# save(edx,file = 'edx.rdata')
# save(validation,file = 'validation.rdata')
# # Load data
# load('edx.rdata')
# load('validation.rdata')

#-------------------------------------------------------------------------------
#                            DATA PREPROCESSING
#-------------------------------------------------------------------------------
#                      >>> Separate title and year <<<
edx <- edx %>% extract(title, c("title", "year"), regex = "(.*)\\s\\((\\d{4})\\)$", remove = TRUE)
validation <- validation %>% extract(title, c("title", "year"), regex = "(.*)\\s\\((\\d{4})\\)$", remove = TRUE)
#                      >>> Separate genres <<<
edx_genres <- edx %>% separate_rows(genres, sep = "\\|") 
#-------------------------------------------------------------------------------


################################################################################
#                                                                              #
#                       PARAMETER TUNING FUNCTIONS                             #
#                                                                              #
#  - functions which are responsible for tuning parameter alpha                #
#                   E = sum[i=0..n]{(y_i - miu) / (n + allpha)}                #
#  - it is basically average of errors lowered by parameter alpha              #
#                                                                              #
################################################################################
#-------------------------------------------------------------------------------
#                      >>> Alpha for movie bias <<<
#-------------------------------------------------------------------------------
getAlphaForMoviesBias <- function(maxAlpha, step, test, train_data, y_hat){
  # define alpha sequence
  alpha <- seq(0,maxAlpha,step)
  # test all alphas
  result <- sapply(alpha, function(alpha){
    # Print progress
    cat(alpha, "\\", maxAlpha, "\n")
    # generate bias for alpha
    movies <- train_data %>% group_by(movieId) %>% summarise(n = n(), movie_bias = sum(rating_spread)/(alpha+n()))
    # calculate predictions - add movie bias
    y_hat_movie <- y_hat + test %>% select(movieId) %>% 
      left_join(movies, "movieId") %>% 
      replace_na(list(n = 0, movie_bias = 0)) %>% 
      .$movie_bias
    c(alpha, RMSE(y_hat_movie,test$rating))
  })
  # process result
  result <- as.data.frame(t(result))
  colnames(result) <- c("alpha","RMSE")
  # generate data for report
  alpha_plots <<- rbind(alpha_plots, result %>% mutate(Type = "Movie", Cycle = cycles))
  # print plot for progress check
  plot <- result %>%
    ggplot(aes(alpha, RMSE)) + 
    geom_point() +
    ggtitle("Calculate alpha for movies bias") + 
    xlab("Alpha") + 
    ylab("RMSE")
  print(plot)
  # Print and retun best alpha
  alpha <- result[(which.min(result$RMSE)),]$alpha
  cat("Alpha for movie is:", alpha, "\n")
  alpha
}
#-------------------------------------------------------------------------------
#                      >>> Alpha for user bias <<<
#-------------------------------------------------------------------------------
getAlphaForUserBias <- function(maxAlpha, step, test, train_data, y_hat){
  # define alpha sequence
  alpha <- seq(0,maxAlpha,step)
  # test all alphas
  result <- sapply(alpha, function(alpha){
    # Print progress
    cat(alpha, "\\", maxAlpha, "\n")
    # generate bias for alpha
    users <- train_data %>% group_by(userId) %>% summarise(n = n(), user_bias = sum(rating_spread)/(alpha+n()))
    # calculate predictions - add users bias
    y_hat_user <- y_hat + test %>% select(userId) %>% 
      left_join(users, "userId") %>% 
      replace_na(list(n = 0, user_bias = 0)) %>% 
      .$user_bias
    c(alpha, RMSE(y_hat_user,test$rating))
  })
  # process result
  result <- as.data.frame(t(result))
  colnames(result) <- c("alpha","RMSE")
  # generate data for report
  alpha_plots <<- rbind(alpha_plots, result %>% mutate(Type = "User", Cycle = cycles))
  # print plot for progress check
  plot <- result %>%
    ggplot(aes(alpha, RMSE)) + 
    geom_point() +
    ggtitle("Calculate alpha for user bias") + 
    xlab("Alpha") + 
    ylab("RMSE")
  print(plot)
  # Print and retun best alpha
  alpha <- result[(which.min(result$RMSE)),]$alpha
  cat("Alpha for user is:", alpha, "\n")
  alpha
}
#-------------------------------------------------------------------------------
#                      >>> Alpha for year bias <<<
#-------------------------------------------------------------------------------
getAlphaForYearBias <- function(maxAlpha, step, test, train_data, y_hat){
  # define alpha sequence
  alpha <- seq(0,maxAlpha,step)
  # test all alphas
  result <- sapply(alpha, function(alpha){
    # Print progress
    cat(alpha, "\\", maxAlpha, "\n")
    # generate bias for alpha
    years <- train_data %>% group_by(year) %>% summarise(year_bias = sum(rating_spread)/(alpha+n()))
    # calculate predictions - add year bias
    y_hat_year <- y_hat + test %>% select(year) %>% 
      left_join(years, "year") %>% 
      replace_na(list(n = 0, year_bias = 0)) %>% 
      .$year_bias
    c(alpha, RMSE(y_hat_year,test$rating))
  })
  # process result
  result <- as.data.frame(t(result))
  colnames(result) <- c("alpha","RMSE")
  # generate data for report
  alpha_plots <<- rbind(alpha_plots, result %>% mutate(Type = "Year", Cycle = cycles))
  # print plot for progress check
  plot <- result %>%
    ggplot(aes(alpha, RMSE)) + 
    geom_point() +
    ggtitle("Calculate alpha for year bias") + 
    xlab("Alpha") + 
    ylab("RMSE")
  print(plot)
  # Print and retun best alpha
  alpha <- result[(which.min(result$RMSE)),]$alpha
  cat("Alpha for year is:", alpha, "\n")
  alpha
}
#-------------------------------------------------------------------------------
#                   >>> Alpha for user & genre bias <<<
#-------------------------------------------------------------------------------
getAlphaForUserGenreBias <- function(maxAlpha, step, test, base_test, base_train, y_hat){
  # define alpha sequence
  alpha <- seq(0,maxAlpha,step)
  # generate data for all alphas
  result <- sapply(alpha, function(alpha){
    # Print progress
    cat(alpha, "\\", maxAlpha, "\n")
    # generate bias for alpha
    base_train %>% summarise(user_genre_bias = sum(rating_spread)/(alpha+n())) %>% 
      ungroup()
  })
  # pre-process data for y_hat calculation
  N <- length(result[3,])
  print("Processing result...")
  result <- data.frame(result[1,1], result[2,1],matrix(unlist(result[3,]), ncol = N))
  # calculate predictions - add user&genre bias
  y_hat_user_genre <- y_hat + base_test %>%
    left_join(result, c("userId", "genres")) %>% 
    group_by(userId, movieId) %>% 
    summarise_at(vars(6:(N+5)), mean) %>% 
    ungroup() %>% 
    select(3:(N+2)) %>% 
    as.matrix()
  y_hat_user_genre <- replace_na(y_hat_user_genre,0)
  result <- data.frame(alpha = alpha, RMSE = apply(y_hat_user_genre, 2, function(x){RMSE(x,test$rating)}))
  colnames(result) <- c("alpha","RMSE")
  # generate data for report
  alpha_plots <<- rbind(alpha_plots, result %>% select(alpha, RMSE) 
                        %>% mutate(Type = "User&Genre", Cycle = cycles))
  # print plot for progress check
  plot <- result %>%
    ggplot(aes(alpha, RMSE)) + 
    geom_point() +
    ggtitle("Calculate alpha for user & genre bias") + 
    xlab("Alpha") + 
    ylab("RMSE")
  print(plot)
  # Print and retun best alpha
  alpha <- result[(which.min(result$RMSE)),]$alpha
  cat("Alpha for user/genre is:", alpha, "\n")
  alpha
}
#-------------------------------------------------------------------------------
#                      >>> Alpha for genre bias <<<
#-------------------------------------------------------------------------------
getAlphaForGenreBias <- function(maxAlpha, step, test, base_test, base_train, y_hat){
  # define alpha sequence
  alpha <- seq(0,maxAlpha,step)
  # generate data for all alphas
  result <- sapply(alpha, function(alpha){
    # Print progress
    cat(alpha, "\\", maxAlpha, "\n")
    # generate bias for alpha
    base_train %>% summarise(genre_bias = sum(rating_spread)/(alpha+n()))
  })
  # pre-process data for y_hat calculation
  N <- length(result[2,])
  print("Processing result...")
  result <- data.frame(result[1,1], matrix(unlist(result[2,]), ncol = N))
  # calculate predictions - add genre bias
  y_hat_genre <- y_hat + base_test %>%
    left_join(result, c("genres")) %>% 
    group_by(userId, movieId) %>% 
    summarise_at(vars(6:(N+5)), mean) %>% 
    ungroup() %>% 
    select(3:(N+2)) %>% 
    as.matrix()
  y_hat_genre <- replace_na(y_hat_genre,0)
  # process result
  result <- data.frame(alpha = alpha, RMSE = apply(y_hat_genre, 2, function(x){RMSE(x,test$rating)}))
  colnames(result) <- c("alpha","RMSE")
  # generate data for report
  alpha_plots <<- rbind(alpha_plots, result %>% select(alpha, RMSE) 
                        %>% mutate(Type = "Genre", Cycle = cycles))
  # print plot for progress check
  plot <- result %>%
    ggplot(aes(alpha, RMSE)) + 
    geom_point() +
    ggtitle("Calculate alpha for genre bias") + 
    xlab("Alpha") + 
    ylab("RMSE")
  print(plot)
  # Print and retun best alpha
  alpha <- result[(which.min(result$RMSE)),]$alpha
  cat("Alpha for genre is:", alpha, "\n")
  alpha
}

################################################################################
#                                                                              #
#                             TRAINING FUNCTION                                #
#                                                                              #
#  - functions responsible for training one model on training edX              #
#  - edX data are randomly divided into test and train data (1:4)              #
#                                                                              #
#          Y = miu + movie_bias + user_bias + year_bias +                      #
#                  + sum(user&genre_bias) + sum(genre)                         #
#                                                                              #
################################################################################
build_model <- function(){
  #-----------------------------------------------------------------------------
  #              >>> Divide data into test and train sets <<<
  #-----------------------------------------------------------------------------
  test_indexes <- createDataPartition(edx$rating, times = 1, p = 0.2, list = FALSE)
  test <- edx[test_indexes,]
  train_data <- edx[-test_indexes,]
  
  #-----------------------------------------------------------------------------
  #                   >>> Follow model building cycles <<<
  #-----------------------------------------------------------------------------
  cycles <<- cycles + 1
  #-----------------------------------------------------------------------------
  #           >>> Make train smaller for developping purposes <<<
  #!!!!!!!!!!!!!!!!!!!!! COMMENT FOR FULL TRAINING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  #-----------------------------------------------------------------------------
  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  # #make train smaller for developping purposes
  # develop_indexes <- createDataPartition(train_data$rating, times = 1, p = 0.01, list = FALSE)
  # train_data <- train_data[develop_indexes,]
  # develop_indexes <- createDataPartition(test$rating, times = 1, p = 0.01, list = FALSE)
  # test <- test[develop_indexes,]
  # # load('edx.rdata')
  # # train_data <- edx
  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
  #-----------------------------------------------------------------------------
  #            >>> Assign global average as starting prediction <<<
  #-----------------------------------------------------------------------------
  cat("Training global mean, cycle:", cycles, "\\", model_training_passes, "\n")
  # Average = miu
  miu = mean(train_data$rating)
  # Add spread of rating from global average miu
  train_data <- train_data %>% mutate(rating_spread = rating - miu)
  # Build predictions vector
  y_hat <- rep(miu, length(test$rating))
  
  #-----------------------------------------------------------------------------
  #                           >>> Movie bias <<<
  #-----------------------------------------------------------------------------
  cat("Training movie bias, cycle:", cycles, "\\", model_training_passes, "\n")
  # Tune alpha
  alpha = getAlphaForMoviesBias(10, 1, test, train_data, y_hat)
  # Build biases for movies
  movies <- train_data %>% group_by(movieId) %>% summarise(n = n(), movie_bias = sum(rating_spread)/(alpha+n()))
  # Correct rating spread with new bias
  train_data <- train_data %>% left_join(movies %>% select(movieId,movie_bias), "movieId") %>% 
    mutate(rating_spread = rating_spread - movie_bias) %>% select(-movie_bias)
  # Adjust predictions vector
  y_hat <- y_hat + test %>% select(movieId) %>% 
    left_join(movies, "movieId") %>% 
    replace_na(list(n = 0, movie_bias = 0)) %>% 
    .$movie_bias
  
  #-----------------------------------------------------------------------------
  #                           >>> User bias <<<
  #-----------------------------------------------------------------------------
  cat("Training user bias, cycle:", cycles, "\\", model_training_passes, "\n")
  # Tune alpha
  alpha = getAlphaForUserBias(10, 1, test, train_data, y_hat)
  # Build biases for users
  users <- train_data %>% group_by(userId) %>% summarise(n = n(), user_bias = sum(rating_spread)/(alpha+n()))
  # Correct rating spread with new bias
  train_data <- train_data %>% left_join(users %>% select(userId, user_bias), "userId") %>% 
    mutate(rating_spread = rating_spread - user_bias) %>% select(-user_bias)
  # Adjust predictions vector
  y_hat <- y_hat + test %>% select(userId) %>% 
    left_join(users, "userId") %>% 
    replace_na(list(n = 0, user_bias = 0)) %>% 
    .$user_bias
  
  #-----------------------------------------------------------------------------
  #                             >>> Year bias <<<
  #-----------------------------------------------------------------------------
  cat("Training year bias, cycle:", cycles, "\\", model_training_passes, "\n")
  # Tune alpha
  alpha = getAlphaForYearBias(3000, 30, test, train_data, y_hat)
  # Build biases for years
  years <- train_data %>% group_by(year) %>% summarise(year_bias = sum(rating_spread)/(alpha+n()))
  # Correct rating spread with new bias
  train_data <- train_data %>% left_join(years %>% select(year, year_bias), "year") %>% 
    mutate(rating_spread = rating_spread - year_bias) %>% select(-year_bias)
  # Adjust predictions vector
  y_hat <- y_hat + test %>% select(year) %>% 
    left_join(years, "year") %>% 
    replace_na(list(n = 0, year_bias = 0)) %>% 
    .$year_bias
  
  #-----------------------------------------------------------------------------
  #                       >>> User & Genre bias <<<
  #-----------------------------------------------------------------------------
  cat("Training user&genre bias, cycle:", cycles, "\\", model_training_passes, "\n")
  # Separate genres for movies in test and train data sets
  base_test <- test %>% separate_rows(genres, sep = "\\|")
  base_train <- train_data %>% separate_rows(genres, sep = "\\|") %>% select(userId, movieId, genres, rating_spread) %>% group_by(userId, genres)
  
  # Tune alpha
  alpha = getAlphaForUserGenreBias(40, 2, test, base_test, base_train, y_hat)
  # Build biases for users $ genres
  user_genre <- base_train %>% select(userId, genres, rating_spread) %>%
    summarise(user_genre_bias = sum(rating_spread)/(alpha+n())) %>%
    ungroup()
  # Correct rating spread with new bias
  tmp <- base_train %>%
    left_join(user_genre, c("userId", "genres")) %>%
    group_by(userId, movieId) %>%
    summarise(user_genre_bias = mean(user_genre_bias))
  train_data <- train_data %>% left_join(tmp, c("userId","movieId")) %>% 
    mutate(rating_spread = rating_spread - user_genre_bias) %>% select(-user_genre_bias)
  # Adjust predictions vector
  y_hat <- y_hat + base_test %>% select(userId, movieId, genres) %>%
    left_join(user_genre, c("userId", "genres")) %>%
    group_by(userId, movieId) %>%
    summarise(user_genre_bias = mean(user_genre_bias)) %>%
    replace_na(list(user_genre_bias = 0)) %>%
    .$user_genre_bias
  
  #-----------------------------------------------------------------------------
  #                           >>> Genre bias <<<
  #-----------------------------------------------------------------------------
  cat("Training genre bias, cycle:", cycles, "\\", model_training_passes, "\n")
  # Separate genres for movies in test and train data sets
  base_test <- test %>% separate_rows(genres, sep = "\\|")  #create base division to genres of train
  base_train <- train_data %>% separate_rows(genres, sep = "\\|") %>% select(userId, movieId, genres, rating_spread) %>% group_by(genres)
  
  # Tune alpha
  alpha = getAlphaForGenreBias(100000, 5000, test, base_test, base_train, y_hat)
  # Build biases for genres
  genres <- base_train %>% select(genres, rating_spread) %>% 
    summarise(genre_bias = sum(rating_spread)/(alpha+n()))
  # Correct rating spread with new bias
  tmp <- base_train %>%
    left_join(genres, c("genres")) %>%
    group_by(userId, movieId) %>%
    summarise(genre_bias = mean(genre_bias))
  train_data <- train_data %>% left_join(tmp, c("userId","movieId"))
  
  #-----------------------------------------------------------------------------
  #                      >>> Return all predictors <<<
  #-----------------------------------------------------------------------------
  c(miu = miu, movies = movies, users = users, years = years, user_genre = user_genre, genres = genres)
}



################################################################################
#                                                                              #
#                                 BUILD MODEL                                  #
#                                                                              #
#  - Model is build for equation:                                              #
#                                                                              #
#          Y = miu + movie_bias + user_bias + year_bias +                      #
#                  + sum(user&genre_bias) + sum(genre)                         #
#                                                                              #
#  - Cycles is number of individual trainings of model on randomply splitted   #
#    edx data. Final model is average of all individual training results.      #
#                                                                              #
################################################################################
#-------------------------------------------------------------------------------
#                           >>> Train model <<<
#-------------------------------------------------------------------------------
# how many individual trained model should be train
model_training_passes = 5
# variable for cycles tracking
cycles <- 0 
# Global variable for plotting <<<
alpha_plots <<- data_frame(matrix(ncol = 4, nrow = 0))
colnames(alpha_plots) <- c("alpha","RMSE","Type","Cycle")
# train the model
model <- replicate(model_training_passes, build_model())
#-------------------------------------------------------------------------------
#                  >>> Ensemble model - using average <<<
#-------------------------------------------------------------------------------
miu <- mean(unlist(model["miu",]))

movies <- data.frame(movieId = unlist(model["movies.movieId",]), movie_bias = unlist(model["movies.movie_bias",]))
movies <- movies %>% group_by(movieId) %>% summarise(movie_bias = mean(movie_bias))

users <- data.frame(userId = unlist(model["users.userId",]), user_bias = unlist(model["users.user_bias",]))
users <- users %>% group_by(userId) %>% summarise(user_bias = mean(user_bias))

years <- data.frame(year = unlist(model["years.year",]), year_bias = unlist(model["years.year_bias",]))
years <- years %>% group_by(year) %>% summarise(year_bias = mean(year_bias))

user_genre <- data.frame(userId = unlist(model["user_genre.userId",]), genres = unlist(model["user_genre.genres",]), user_genre_bias = unlist(model["user_genre.user_genre_bias",]))
user_genre <- user_genre %>% group_by(userId, genres) %>% summarise(user_genre_bias = mean(user_genre_bias)) %>% ungroup()

genres <- data.frame(genres = unlist(model["genres.genres",]), genre_bias = unlist(model["genres.genre_bias",]))
genres <- genres %>% group_by(genres) %>% summarise(genre_bias = mean(genre_bias))


################################################################################
#                                                                              #
#                                VALIDATE MODEL                                #
#                                                                              #
################################################################################
#-------------------------------------------------------------------------------
#                      >>> Start with Average miu <<<
#-------------------------------------------------------------------------------
y_hat <- rep(miu, length(validation$rating))
rmse_results <- data_frame(method="Average RMSE", 
                           RMSE = RMSE(y_hat,validation$rating))
#-------------------------------------------------------------------------------
#                          >>> Add Movide bias <<<
#-------------------------------------------------------------------------------
y_hat <- y_hat + validation %>% select(movieId) %>% 
  left_join(movies, "movieId") %>% 
  replace_na(list(movie_bias = 0)) %>% 
  .$movie_bias
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="+Movie bias RMSE: ",
                                     RMSE = RMSE(y_hat,validation$rating)))
#-------------------------------------------------------------------------------
#                           >>> Add User bias <<<
#-------------------------------------------------------------------------------
y_hat <- y_hat + validation %>% select(userId) %>% 
  left_join(users, "userId") %>% 
  replace_na(list(user_bias = 0)) %>% 
  .$user_bias
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="+User bias RMSE: ",
                                     RMSE = RMSE(y_hat,validation$rating)))
#-------------------------------------------------------------------------------
#                           >>> Add Year bias <<<
#-------------------------------------------------------------------------------
y_hat <- y_hat + validation %>% select(year) %>% 
  left_join(years, "year") %>% 
  replace_na(list(n = 0, year_bias = 0)) %>% 
  .$year_bias
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="+Yearr bias RMSE: ",
                                     RMSE = RMSE(y_hat,validation$rating)))
#-------------------------------------------------------------------------------
#                >>> Separate genres for validation data <<<
validation_genre <- validation %>% select(userId, movieId, genres) %>%
  separate_rows(genres, sep = "\\|")
#-------------------------------------------------------------------------------
#                        >>> Add User & genre bias <<<
#-------------------------------------------------------------------------------
y_hat <- y_hat + validation_genre %>%
  left_join(user_genre, c("userId", "genres")) %>%
  group_by(userId, movieId) %>%
  summarise(user_genre_bias = mean(user_genre_bias)) %>%
  replace_na(list(user_genre_bias = 0)) %>%
  .$user_genre_bias
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="+User & Genre bias RMSE: ",
                                     RMSE = RMSE(y_hat,validation$rating)))
#-------------------------------------------------------------------------------
#                           >>> Add Genre bias <<<
#-------------------------------------------------------------------------------
y_hat <- y_hat + validation_genre %>%
  left_join(genres, c("genres")) %>%
  group_by(userId, movieId) %>%
  summarise(genre_bias = mean(genre_bias)) %>%
  replace_na(list(genre_bias = 0)) %>%
  .$genre_bias
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="+Genre bias RMSE: ",
                                     RMSE = RMSE(y_hat,validation$rating)))

rmse_results$RMSE
#-------------------------------------------------------------------------------
#                             >>> Final RMSE <<<
#-------------------------------------------------------------------------------
RMSE <- RMSE(y_hat,validation$rating)

save.image(file='myEnvironment.RData')


