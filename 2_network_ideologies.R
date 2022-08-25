#!/usr/bin/env Rscript
# ------------------------------------------------------------------------------
# Filename:    2_network_ideologies.R
# Description: Runs network ideology model adapted from Barberà (2015)
# Source:      Barberà (2015)
# Duration:    ~ 24 hours
# ------------------------------------------------------------------------------

options_elec <- sort(c("2015-can-canada", "2014-can-quebec", "2014-nzl-new-zealand"))
options_mode <- sort(c("ca", "nuts"))

input_mode <- 2 # 1 : ca, 2 : nuts

for (input_elec in seq_along(options_elec)) {

  message("Creating general options")
  # Choose between 2015-can-canada, 2014-can-quebec and 2014-nzl-new-zealand

  election        <- options_elec[input_elec]
  party_thres     <- 25       # Threshold on number of VPL-followers required for a party (ex: 25)
  user_thres      <- 3        # Threshold on number of parties followed by a user (default : 2)
  matrix_ext      <- "csv"    # Default : csv (allows potential 'rdata' input)
  mode            <- options_mode[input_mode]
  suppress_other  <- T        # Suppress not-engaged political member

  # Parameters for mode == ca
  chosen_dim      <- ifelse(input_mode == 1, 1, '')  # if ca, chose dimension 1 or 2. if nuts, put at ''

  # Parameters for mode == nuts - Ignore if mode == 'ca
  engagement_init <- T        # Initialize at party values - suppress_other needs to be True
  mode_init       <- "party"  # party (engagement_init == T) or ca
  chosen_dim_init <- ""       # set empty if mode_init == party, set to 1 or 2 if ca

  # Stan Parameters
  n_iter          <- 1000
  n_warmup        <- 200
  thin            <- 2

  # Input files
  input_path    <- "../data/raw/follow/"
  clean_path    <- "../data/clean/"
  filename      <- paste0(clean_path, election, "/sparse_matrix.", matrix_ext)
  file_init     <- paste0(input_path, election, "/politicians_init.csv")

  ## Output files
  path          <- paste0("../data/ideologies/", election, "/")
  MLpath        <- paste0("../data/machine_learning/ideologies/", election, "/")
  detail_path   <- paste0(path,'details/')
  date          <- format(Sys.time(), "%Y-%m-%d_%I-%p")
  global_name   <- paste0("follow", "_", mode, chosen_dim)
  if (mode == "nuts") {
      global_name <- paste0(global_name, "_", mode_init, chosen_dim_init)
  }
  pol_name              <- paste0(global_name, "_politicians")
  users_name            <- paste0(global_name, "_users")

  politicians_file      <- paste0(path  , pol_name  , ".csv")   # main file
  users_file            <- paste0(path  , users_name, ".csv") # main file
  users_file_ML         <- paste0(MLpath, users_name, ".csv") # main file

  stan_file             <- paste0(detail_path, date, "_", global_name, ".rdata")
  parameters_file       <- paste0(detail_path, date, "_", global_name, ".csv")
  politicians_file_test <- paste0(detail_path, date, "_", pol_name, ".csv")
  users_file_test       <- paste0(detail_path, date, "_", users_name, ".csv")


  # 1) Load Matrix information ---------------------------------------------------
  message("1) Loading Matrix information")
  library(Matrix)

  if (matrix_ext == "csv") {
      indices <- read.csv(filename, header = FALSE, stringsAsFactors = FALSE)
      rows <- as.numeric(indices[1, ])
      rows <- rows[!is.na(rows)]
      columns <- as.numeric(indices[2, ])
      columns <- columns[!is.na(columns)]
      row_names <- indices[3, ]
      row_names <- as.character(row_names[!is.na(row_names)])
      col_names <- indices[4, ]
      col_names <- as.character(col_names[!is.na(col_names)])
      col_names <- col_names[col_names != ""]
      y <- sparseMatrix(i = rows, j = columns)
      row_names <- row_names[1:dim(y)[1]]
      col_names <- col_names[1:dim(y)[2]]
      rownames(y) <- row_names
      colnames(y) <- col_names
  } else if (matrix_ext == "rdata") {
      load(filename)
      col_names <- colnames(y)
      row_names <- rownames(y)
      col_names <- sapply(1:length(col_names), function(x) trimws(col_names[x]))  # Suppress white-space errors
      # Transform Twitter_account into Twitter_ID
      match_col <- read.csv(file_extended, stringsAsFactors = FALSE)
      match_col <- match_col[!duplicated(match_col$Twitter_ID), ]
      new_col_names <- as.character(sapply(1:length(col_names), function(x) match_col$Twitter_ID[match_col$Twitter_account ==
          col_names[x]]))
      # Suppress unretrieved politicians
      kept_pols <- new_col_names != "numeric(0)"
      y <- y[, kept_pols]
      col_names <- new_col_names[kept_pols]
      colnames(y) <- unlist(col_names)
      # col_names <- as.character(unlist(col_names))
  }

  message("2) Create initialization dataframe (optional)")
  # 2) Create initialization dataframe (optional) --------------------------------
  if (suppress_other || (engagement_init && mode_init == "party")) {
      df_init <- read.csv(file_init, stringsAsFactors = FALSE)
      df_init <- df_init[!duplicated(df_init$key), ]
  }

  message("3) Apply thresholds")
  # 3) Apply thresholds ----------------------------------------------------------

  # a) Keep engaged parties
  if (suppress_other) {
      engaged_parties <- col_names %in% df_init$key
      y               <- y[, engaged_parties]
      col_names       <- col_names[engaged_parties]
  }

  # b) Keep parties with more than party_thres followers
  kept_parties <- (colSums(y) >= party_thres)
  y            <- y[, kept_parties]
  col_names    <- col_names[kept_parties]
  # c) Keep users following more than user_thres parties
  kept_users   <- (rowSums(y) >= user_thres)
  y            <- y[kept_users, ]
  row_names    <- row_names[kept_users]
  # Compute final number of users and parties
  nparties     <- length(col_names)
  nusers       <- length(row_names)

  # Display sparsity
  sparsity <- 100 * sum(colSums(y))/(nusers * nparties) # Number of 1 entries
  force_value <- nusers/nparties
  dim(y)
  sum(colSums(y) == 0)
  sum(rowSums(y) == 0)


  # 4) STAN ----------------------------------------------------------------------
  if (mode == "nuts") {
    message("4) STAN Setup")
      # a) Initialization
      if (engagement_init) {
          # Option 1 : initialize using ca results ##
          if (mode_init == "ca") {
              y <- as.matrix(y)
              res <- ca::ca(y)
              phi_init_mat <- data.frame(res$colcoord[, chosen_dim_init])
              phi_init <- phi_init_mat[, 1]
          # Option 2 : initialize via party affiliation ##
          } else if (mode_init == "party") {
              df_merged <- merge(col_names, df_init, by.x = "x", by.y = "key", sort = F)
              if (dim(df_merged)[1] != length(col_names)) {
                  stop("Problem in the initialization process")
              }
              phi_init <- df_merged$value
          }
          # Option 3 : initialize randomly ##
      } else {
          phi_init <- runif(length(col_names), -1, 1)
      }
      # Finally : Assign initialization value
      start_phi <- as.numeric(phi_init)

      # b) Model
      J  <- dim(y)[1]
      K  <- dim(y)[2]
      N  <- J * K
      jj <- rep(1:J, times = K)
      kk <- rep(1:K, each = J)

      stan_data <- list(J = J, K = K, N = N, jj = jj, kk = kk, y = c(as.matrix(y)))

      # Remaining starting values
      colK <- colSums(y)
      rowJ <- rowSums(y)
      normalize <- function(x) {
          (x - mean(x))/sd(x)
      }
      inits <- rep(list(list(alpha       = normalize(log(colK + 1e-04)),
                             beta        = normalize(log(rowJ + 1e-04)),
                             theta       = rnorm(J),
                             phi         = start_phi,
                             mu_beta     = 0,
                             sigma_beta  = 1,
                             gamma       = abs(rnorm(1)),
                             mu_phi      = 0,
                             sigma_phi   = 1,
                             sigma_alpha = 1)), 2)

      library(rstan)

      stan.code <- "
    data {
      int<lower=1> J; // number of twitter users
      int<lower=1> K; // number of elite twitter accounts
      int<lower=1> N; // N = J x K
      int<lower=1,upper=J> jj[N]; // twitter user for observation n
      int<lower=1,upper=K> kk[N]; // elite account for observation n
      int<lower=0,upper=1> y[N]; // dummy if user i follows elite j
    }
    parameters {
      vector[K] alpha;
      vector[K] phi;
      vector[J] theta;
      vector[J] beta;
      real mu_beta;
      real<lower=0.1> sigma_beta;
      real mu_phi;
      real<lower=0.1> sigma_phi;
      real<lower=0.1> sigma_alpha;
      real gamma;
    }
    model {
      alpha ~ normal(0, sigma_alpha);
      beta ~ normal(mu_beta, sigma_beta);
      phi ~ normal(mu_phi, sigma_phi);
      theta ~ normal(0, 1);
      for (n in 1:N)
        y[n] ~ bernoulli_logit( alpha[kk[n]] + beta[jj[n]] -
                               gamma * square( theta[jj[n]] - phi[kk[n]] ) );
    }
    "
      rstan_options(auto_write = TRUE)
      options(mc.cores = parallel::detectCores())
  }

  message("5) Model Execution")
  # 5) Execution -----------------------------------------------------------------
  if (mode == "nuts") {
      stan.model <- stan(model_code = stan.code, data = stan_data, iter = 1, warmup = 0,
          chains = 2, init = inits, seed = 100)

      stan.fit <- stan(fit = stan.model, data = stan_data, iter = n_iter, warmup = n_warmup,
          chains = 2, thin = thin, init = inits, seed = 100)

  } else if (mode == "ca") {
      y <- as.matrix(y)
      res <- ca::ca(y)
  } else {
      stop("Problem : mode wrong defined")
  }

  message("Saving file")
  # 6) Saving files --------------------------------------------------------------
  if (mode == "nuts") {
      # a) Stan Model
      save(stan.fit, file = stan_file)
      # b) Phi and theta Extract samples
      samples       <- extract(stan.fit, pars = c("phi", "theta"))
      ## Capture phi and theta values
      phi_values0   <- samples$phi
      theta_values0 <- samples$theta
      phi_values    <- t(phi_values0)
      theta_values  <- t(theta_values0)
      phi_values    <- rowMeans(phi_values)
      theta_values  <- rowMeans(theta_values)

  } else if (mode == "ca") {
      phi_values0 <- data.frame(res$colcoord[, chosen_dim])
      phi_values <- phi_values0[, 1]
      theta_values0 <- data.frame(res$rowcoord[, chosen_dim])
      theta_values <- theta_values0[, 1]
  }

  # Write politicians file
  write.table(data.frame(col_names, phi_values),
              file = politicians_file, sep = ",",
              col.names = c("key", "value"), row.names = F)

  write.table(data.frame(col_names, phi_values),
              file = politicians_file_test, sep = ",",
              col.names = c("key", "value"), row.names = F)

  # Write users file
  write.table(data.frame(row_names, theta_values),
              file = users_file, sep = ",",
              col.names = c("key", "value"), row.names = F)

  write.table(data.frame(row_names, theta_values),
              file = users_file_ML, sep = ",",
              col.names = c("key", "value"), row.names = F)

  write.table(data.frame(row_names, theta_values),
              file = users_file_test, sep = ",",
              col.names = c("key", "value"), row.names = F)

  # Write parameters
  parameters <- data.frame(date, party_thres, user_thres, suppress_other, mode,
                           chosen_dim, engagement_init, "follow", n_iter,
                           n_warmup, thin, mode_init, chosen_dim_init, matrix_ext)

  write.table(parameters, file = parameters_file, sep = ",", row.names = F)
}
