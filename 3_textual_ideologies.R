#!/usr/bin/env Rscript
# ------------------------------------------------------------------------------
# Filename:    3_textual_ideologies.R
# Description: Runs wordfish classic, political, dla
# Source:      Slapin and Proksch (2008); based on Wordfish v1.3
# Duration:    ~ 1.5 hours
# ------------------------------------------------------------------------------

param_iter = read.csv("params_grid.csv", stringsAsFactors=FALSE)

for (i in 1:nrow(param_iter)){
    message("Starting with options")
    election          <- param_iter[i,][['elections']]
    wf_mode           <- param_iter[i,][['wf_modes']]
    user_mode         <- param_iter[i,][['user_modes']]
    ngram             <- param_iter[i,][['ngram']]
    chosen_dim        <- ""    # put at '' (allow debug with corresp. analysis)

    # User options
    min_words_users   <- param_iter[i,][['min_words_users']] # a user needs to tweet enough (example : 20 or 10)
    in_min_docs_users <- param_iter[i,][['in_min_docs_users']] # a word needs to belong to enough users (example: 4 or 3)
    # Candidates options
    min_words_party   <- param_iter[i,][['min_words_party']] # a user needs to tweet enough (example : 25 or 15)
    in_min_docs_party <- param_iter[i,][['in_min_docs_party']] # a word needs to belong to enough users (example : 5 or 3)
    suppress_other    <- T       # suppress unengaged politicians

    # Input files
    input_path = "../data/raw/text/"
    TM_file_users <- paste0(input_path, election, "/TM", ngram, ".csv")
    TM_file_party <- paste0(input_path, election, "/politicians_TM", ngram, ".csv")
    tw_users_file <- paste0(input_path, election, "/link.csv")
    tw_party_file <- paste0(input_path, election, "/politicians_link.csv")
    file_init     <- paste0(input_path, election, "/politicians_init.csv")

    ## Output files
    if (param_iter[i,][['params']] == "desc") {
        path               <- paste0("../data/ideologies/", election, "/")
        detail_path        <- paste0(path,'details/')
        date               <- format(Sys.time(), "%Y-%m-%d_%I-%p")
        global_name        <- paste0("text", "_", wf_mode, chosen_dim, "_",
        		     ngram, "_", user_mode)

        tweeters_file      <- paste0(path, global_name, ".csv") # main file
        parameters_file    <- paste0(detail_path, date, "_", global_name, "_parameters.csv")
        words_file         <- paste0(detail_path, date, "_", global_name, "_words.csv")
        tweeters_file_test <- paste0(detail_path, date, "_", global_name, ".csv")
    }
    ## Output files
    if (param_iter[i,][['params']] == "ml") {
        path               <- paste0("../data/machine_learning/ideologies/", election, "/")
        detail_path        <- paste0(path,'details/')
        date               <- format(Sys.time(), "%Y-%m-%d_%I-%p")
        global_name        <- paste0("text", "_", wf_mode, chosen_dim, "_",
        		     ngram, "_", user_mode)

        tweeters_file      <- paste0(path, global_name, ".csv") # main file
        parameters_file    <- paste0(detail_path, date, "_", global_name, "_parameters.csv")
        words_file         <- paste0(detail_path, date, "_", global_name, "_words.csv")
        tweeters_file_test <- paste0(detail_path, date, "_", global_name, ".csv")
    }


    # Functions --------------------------------------------------------------------

    # a) Wordfish
    wordfish <- function(input,
    	     wordsincol = FALSE,
    	     fixtwo     = FALSE,
    	     dir        = NULL,
    	     fixdoc     = c(1, 2, 0, 1),
    	     tol        = 1e-07,
    	     sigma      = 3,
    	     boots      = FALSE,
    	     nsim       = 500,
    	     writeout   = FALSE,
    	     output     = "wordfish_output") {
    dta <- input
    if (wordsincol == TRUE) {
    rownames(dta) <- dta[, 1]
    dta <- dta[, -c(1)]
    }

    dta    <- t(dta)
    words  <- colnames(dta)
    nparty <- nrow(dta)
    nword  <- ncol(dta)

    if (fixtwo == TRUE) {
    if (fixdoc[3] == fixdoc[4]) {
    cat("Warning: fixed omega values in 'fixdoc' cannot be identical. \n")
    stop()
    }

    identprint <- paste("Omegas identified with", rownames(dta)[fixdoc[1]], "=",
    		fixdoc[3], "and ", rownames(dta)[fixdoc[2]], "=", fixdoc[4])
    } else {

    if (sum(c(length(dir) == 2, is.numeric(dir))) != 2) {
    cat("Warning: option 'dir' in wordfish() is empty. You must specify two documents for global identification (e.g. dir=c(1,2) ).\n")
    stop()
    }

    identprint <- paste("Omegas identified with mean 0, st.dev. 1")
    }

    cat("======================================\n")
    cat("WORDFISH (Version 1.3)\n")
    cat("======================================\n")
    cat("Number of unique words: ", nword, "\n")
    cat("Number of documents: ", nparty, "\n")
    cat("Tolerance criterion: ", tol, "\n")
    cat("Identification: ", identprint, "\n")
    cat("======================================\n")

    # Generate starting values ========================

    if (fixtwo == FALSE) {

    rockingstarts <- function(dta) {
    cat("Performing mean 0 sd 1 starting value calc\n")
    P <- nrow(dta)
    W <- ncol(dta)
    numword <- rep(1:W, each = P)
    numparty <- rep(1:P, W)
    dat <- matrix(1, nrow = W * P, ncol = 3)
    dat[, 1] <- as.vector(as.matrix(dta))
    dat[, 2] <- as.vector(numword)
    dat[, 3] <- as.vector(numparty)
    dat <- data.frame(dat)
    colnames(dat) <- c("y", "word", "party")
    dat$word <- factor(dat$word)
    dat$party <- factor(dat$party)

    # Starting values for psi print(dta)
    psi <- log(colMeans(dta))
    # Starting values for alpha
    alpha <- log(rowMeans(dta)/rowMeans(dta)[1])

    # Starting values for beta and x
    ystar <- log(dat$y + 0.1) - alpha[dat$party] - psi[dat$word]
    # print(head(scale(matrix(ystar,nrow(dta),ncol(dta),byrow=FALSE))))
    res <- svd(matrix(ystar, nrow(dta), ncol(dta), byrow = FALSE), nu = 1)
    b <- as.vector(res$v[, 1] * res$d[1])

    omega1 <- as.vector(res$u) - res$u[1, 1]
    omega <- omega1/sd(omega1)
    b <- b * sd(omega1)

    # Create holding bins for some stuff for the convergence code
    min1 <- c(rep(1, nrow(dta) - 1))
    min2 <- c(rep(1, ncol(dta)))
    iter <- 0
    conv <- 0
    diffparam <- 0

    # Put everything together in a list
    list(alpha = as.vector(alpha), psi = as.vector(psi), b = b, omega = omega,
       min1 = min1, min2 = min2, iter = iter, conv = conv, diffparam = diffparam)
    }

    } else {

    rockingstarts <- function(dta, fixval) {
    cat("Performing fix two omega starting value calc\n")
    P <- nrow(dta)
    W <- ncol(dta)
    numword <- rep(1:W, each = P)
    numparty <- rep(1:P, W)
    dat <- matrix(1, nrow = W * P, ncol = 3)
    dat[, 1] <- as.vector(as.matrix(dta))
    dat[, 2] <- as.vector(numword)
    dat[, 3] <- as.vector(numparty)
    dat <- data.frame(dat)
    colnames(dat) <- c("y", "word", "party")
    dat$word <- factor(dat$word)
    dat$party <- factor(dat$party)

    # Starting values for psi
    psi <- log(colMeans(dta))
    # Starting values for alpha
    alpha <- log(rowMeans(dta)/rowMeans(dta)[1])

    # Starting values for beta and x
    ystar <- log(dat$y + 0.1) - alpha[dat$party] - psi[dat$word]
    res <- svd(matrix(ystar, nrow(dta), ncol(dta), byrow = FALSE), nu = 1)
    b <- as.vector(res$v[, 1] * res$d[1])

    omega <- as.vector(res$u)

    # Create holding bins for some stuff for the convergence code
    min1 <- c(rep(1, nrow(dta) - 1))
    min2 <- c(rep(1, ncol(dta)))
    iter <- 0
    conv <- 0
    diffparam <- 0

    # Put everything together in a list
    list(alpha = as.vector(alpha), psi = as.vector(psi), b = b, omega = omega,
       min1 = min1, min2 = min2, iter = iter, conv = conv, diffparam = diffparam)
    }
    }

    # Log-Likelihood Functions (Poisson model)
    # ========================================

    llik_psi_b <- function(p, y, omega, alpha, sigma) {
    # beta and psi will be estimated
    b <- p[1]
    psi <- p[2]
    lambda <- exp(psi + alpha + b * omega)  # Lambda parameter for Poisson distribution
    -(sum(-lambda + log(lambda) * y) - 0.5 * (b^2/sigma^2))  # Log-likelihood including normal prior on Beta
    }


    llik_alpha_1 <- function(p, y, b, psi) {
    # omega[1] is estimated
    omega <- p[1]
    lambda <- exp(psi + b * omega)  # Lambda parameter; alpha is excluded b/c it is set to be zero
    -sum(-lambda + log(lambda) * y)  # Log-likelihood
    }

    llik_alpha_omega <- function(p, y, b, psi) {
    # all other omegas and alphas are estimated
    omega <- p[1]
    alpha <- p[2]
    lambda <- exp(psi + alpha + b * omega)  # Lambda parameter
    -sum(-lambda + log(lambda) * y)  # Log-likelihood
    }


    llik_justalpha <- function(p, y, b, psi, omega) {
    # alpha is estimated
    alpha <- p[1]
    lambda <- exp(psi + alpha + b * omega)  # Lambda parameter
    -sum(-lambda + log(lambda) * y)  # Log-likelihood
    }





    if (fixtwo == FALSE) {


    cat("Performing mean 0 sd 1 EM algorithm\n")
    # Expectation-Maximization Algorithm FOR MEAN 0, SD 1 IDENTIFICATION
    # ==================================================================

    rockingpoisson <- function(dta, tol, sigma, params = NULL, dir = dir, printsum = TRUE) {

    P <- nrow(dta)
    W <- ncol(dta)

    if (is.null(params)) {
    params <- rockingstarts(dta)  # Call up starting value calculation
    }

    iter <- 2
    maxllik <- cbind(-1e+70, rep(0, 1400))
    ll.words <- matrix(-1e+70, W, 1400)
    diffllik <- 500

    # Set the convergence criterion
    conv <- tol
    params$conv <- conv

    while (diffllik > conv) {
    # Run algorithm if difference in LL > convergence criterion
    omegaprev <- params$omega
    bprev <- params$b
    alphaprev <- params$alpha
    psiprev <- params$psi

    # ESTIMATE OMEGA AND ALPHA

    if (printsum == TRUE) {
      cat("Iteration", iter - 1, "\n")
      cat("\tUpdating alpha and omega..\n")
    }



    # Estimate first omega (alpha is set to 0)
    resa <- optim(p = c(params$omega[1]), fn = llik_alpha_1, y = as.numeric(dta[1,
    									]), b = params$b, psi = params$psi, method = c("BFGS"))
    params$omega[1] <- resa$par[1]
    params$min1[1] <- -1 * resa$value
    params$alpha[1] <- 0
    ifelse(resa$convergence != 0, print("Warning: Optim Failed to Converge!"),
           NA)


    # Estimate all other omegas and alphas
    for (i in 2:P) {

      resa <- optim(par = c(params$omega[i], params$alpha[i]), fn = llik_alpha_omega,
    		y = as.numeric(dta[i, ]), b = params$b, psi = params$psi)
      params$omega[i] <- resa$par[1]
      params$alpha[i] <- resa$par[2]
      params$min1[i] <- -1 * resa$value
      ifelse(resa$convergence != 0, print("Warning: Optim Failed to Converge!"),
    	 NA)

    }

    flush.console()


    # Z-score transformation of estimates for omega (to identify model)
    omegabar <- mean(params$omega)
    b1 <- params$b
    params$b <- params$b * sd(params$omega)
    params$omega <- (params$omega - omegabar)/sd(params$omega)
    params$psi <- params$psi + b1 * omegabar

    # Global identification
    if (params$omega[dir[1]] > params$omega[dir[2]]) {
      params$omega <- params$omega * (-1)
    }



    # ESTIMATE PSI AND BETA
    if (printsum == TRUE) {
      cat("\tUpdating psi and beta..\n")
    }

    for (j in 1:W) {
      resb <- optim(par = c(params$b[j], params$psi[j]), fn = llik_psi_b,
    		y = dta[, j], omega = params$omega, alpha = params$alpha, sigma = sigma)
      params$b[j] <- resb$par[1]
      params$psi[j] <- resb$par[2]
      params$min2[j] <- -1 * resb$value
      ifelse(resa$convergence != 0, print("Warning: Optim Failed to Converge!"),
    	 NA)
    }

    flush.console()

    # Calculate Log-Likelihood
    maxllik[iter] <- sum(params$min2)
    diffparam <- mean(abs(params$omega - omegaprev))  # difference btw current & previous estimate for omega

    ll.words[, iter] <- params$min2
    diff.ll.words <- (ll.words[, iter] - ll.words[, iter - 1])
    diffllik <- sum(diff.ll.words)/abs(maxllik[iter])


    if (printsum == TRUE) {
      # print(sum(diff.ll.words)) print(abs(maxllik[iter]))
      cat("\tConvergence of LL: ", diffllik, "\n")
    }

    params$diffllik[iter - 1] <- diffllik
    params$diffparam[iter - 1] <- diffparam
    params$diffparam.last <- diffparam
    params$maxllik[iter - 1] <- maxllik[iter]
    params$iter <- iter - 1
    iter <- iter + 1
    }
    params$diffllik[1] <- NA
    return(params)
    }

    # Run the algorithm
    est <- rockingpoisson(dta, tol, sigma, dir = dir)
    } else {
    cat("Performing fix two omega EM algorithm\n")

    # Expectation-Maximization Algorithm FOR TWO FIXED OMEGAS
    # ==================================================================

    rockingpoisson <- function(dta, tol, sigma, params = NULL, fixdoc = fixdoc,
    		       printsum = TRUE) {

    P <- nrow(dta)
    W <- ncol(dta)

    if (is.null(params)) {
    params <- rockingstarts(dta, fixval = fixdoc)  # Call up starting value calculation
    }

    iter <- 2
    maxllik <- cbind(-1e+70, rep(0, 1000))
    ll.words <- matrix(-1e+70, W, 1000)

    diffllik <- 500

    # Set the convergence criterion
    conv <- tol
    params$conv <- conv

    while (diffllik > conv) {
    # Run algorithm if difference in LL > convergence criterion
    omegaprev <- params$omega
    bprev <- params$b
    alphaprev <- params$alpha
    psiprev <- params$psi

    # ESTIMATE OMEGA AND ALPHA

    if (printsum == TRUE) {
      cat("Iteration", iter - 1, "\n")
      cat("\tUpdating alpha and omega..\n")
    }


    # Set omegas and first alpha

    params$omega[fixdoc[1]] <- fixdoc[3]
    params$omega[fixdoc[2]] <- fixdoc[4]
    params$alpha[1] <- 0


    if (1 %in% fixdoc[1:2] == TRUE) {

      # if first doc is one of the fixed omegas, do nothing (alpha and omega are fixed)

    } else {
      # Estimate first omega (alpha is set to 0)
      resa <- optim(p = c(params$omega[1]), fn = llik_alpha_1, y = as.numeric(dta[1,
    									  ]), b = params$b, psi = params$psi, method = c("BFGS"))
      params$omega[1] <- resa$par[1]
      params$min1[1] <- -1 * resa$value
      params$alpha[1] <- 0
      ifelse(resa$convergence != 0, print("Warning: Optim Failed to Converge!"),
    	 NA)
    }




    # Estimate all other omegas and alphas
    for (i in 2:P) {


      if (sum(fixdoc[1:2] == i) == 1) {

        # Estimate just alpha
        resa <- optim(par = params$alpha[i], fn = llik_justalpha, y = as.numeric(dta[i,
    									     ]), b = params$b, psi = params$psi, omega = params$omega[i],
    		  method = c("BFGS"))
        params$alpha[P] <- resa$par[1]

        ifelse(resa$convergence != 0, print("Warning: Optim Failed to Converge!"),
    	   NA)

      } else {
        resa <- optim(par = c(params$omega[i], params$alpha[i]), fn = llik_alpha_omega,
    		  y = as.numeric(dta[i, ]), b = params$b, psi = params$psi)
        params$omega[i] <- resa$par[1]
        params$alpha[i] <- resa$par[2]
        params$min1[i] <- -1 * resa$value
        ifelse(resa$convergence != 0, print("Warning: Optim Failed to Converge!"),
    	   NA)
      }

    }


    flush.console()



    # ESTIMATE PSI AND BETA
    if (printsum == TRUE) {
      cat("\tUpdating psi and beta..\n")
    }

    for (j in 1:W) {
      resb <- optim(par = c(params$b[j], params$psi[j]), fn = llik_psi_b,
    		y = dta[, j], omega = params$omega, alpha = params$alpha, sigma = sigma)
      params$b[j] <- resb$par[1]
      params$psi[j] <- resb$par[2]
      params$min2[j] <- -1 * resb$value
      ifelse(resa$convergence != 0, print("Warning: Optim Failed to Converge!"),
    	 NA)
    }

    flush.console()

    # Calculate Log-Likelihood
    maxllik[iter] <- sum(params$min2)
    diffparam <- mean(abs(params$omega - omegaprev))  # difference between current and previous estimate for omega

    ll.words[, iter] <- params$min2
    diff.ll.words <- (ll.words[, iter] - ll.words[, iter - 1])
    diffllik <- sum(diff.ll.words)/abs(maxllik[iter])

    # print(sum(diff.ll.words)) print(abs(maxllik[iter]))
    if (printsum == TRUE) {
      cat("\tConvergence of LL: ", diffllik, "\n")
    }

    params$diffllik[iter - 1] <- diffllik
    params$diffparam[iter - 1] <- diffparam
    params$diffparam.last <- diffparam
    params$maxllik[iter - 1] <- maxllik[iter]
    params$iter <- iter - 1
    iter <- iter + 1
    }
    params$diffllik[1] <- NA
    return(params)
    }

    # Run the algorithm
    est <- rockingpoisson(dta, tol, sigma, fixdoc = fixdoc)
    }

    cat("======================================\n")
    cat("WORDFISH ML Estimation finished.\n")
    cat("======================================\n\n")

    # Write output
    output.documents <- cbind(est$omega, est$alpha)
    rownames(output.documents) <- rownames(dta)
    colnames(output.documents) <- c("omega", "alpha")
    output.words <- cbind(est$b, est$psi)
    rownames(output.words) <- words
    colnames(output.words) <- c("b", "psi")

    # Write estimation output file Include: Log-likelihood, iterations, number of
    # words, number of documents

    output.estimation <- cbind(nword, nparty, est$iter, sum(est$min2), est$conv,
    		     est$diffparam.last)
    colnames(output.estimation) <- c("Words", "Documents", "Iterations", "Log-Likelihood",
    			   "Convergence Criterion", "Difference in X")

    if (writeout == TRUE) {
    write.table(output.documents, file = paste(output, "documents.csv", sep = "_"))
    write.table(output.words, file = paste(output, "words.csv", sep = "_"))
    write.table(output.estimation, file = paste(output, "estimation.csv", sep = "_"))
    }

    ########################### Parametric Bootstrap Code

    bootstrap <- function(nsim, output.documents, output.words, nparty, nword) {

    cat("STARTING PARAMETRIC BOOTSTRAP\n")

    # input alpha and omega from estimation
    alpha.omega <- output.documents

    # input psis and betas from estimation
    psi.beta <- output.words

    # Create matrix of results.
    output.se.omega <- matrix(0, nparty, nsim)
    output.se.b <- matrix(0, nword, nsim)

    alpha <- alpha.omega[, 2]
    omega <- alpha.omega[, 1]
    psi <- psi.beta[, 2]
    b <- psi.beta[, 1]

    # create data matrix
    dtasim <- matrix(1, nrow = nparty, ncol = nword)
    cat("======================================\n")
    cat("Now running", nsim, "bootstrap trials.\n")
    cat("======================================\n")
    cat("Simulation ")

    for (k in 1:nsim) {

    cat(k, "...")

    # Generate new data using lambda
    for (i in 1:nparty) {
    dtasim[i, ] <- rpois(nword, exp(psi + alpha[i] + b * omega[i]))
    }

    alphastart <- alpha + rnorm(length(alpha.omega[, 1]), mean = 0, sd = (sd(alpha.omega[,
    								       2])/2))
    omegastart <- omega + rnorm(length(alpha.omega[, 1]), mean = 0, sd = (sd(alpha.omega[,
    								       1])/2))
    psistart <- psi + rnorm(length(psi.beta[, 1]), mean = 0, sd = (sd(psi.beta[,
    								2])/2))
    bstart <- b + rnorm(length(psi.beta[, 1]), mean = 0, sd = (sd(psi.beta[,
    							    1])/2))
    params <- list(alpha = alphastart, omega = omegastart, psi = psistart,
    	     b = bstart)


    if (fixtwo == FALSE) {
    est <- rockingpoisson(dtasim, tol, sigma, params = params, dir = dir,
    		      printsum = FALSE)
    } else {
    est <- rockingpoisson(dtasim, tol, sigma, params = params, fixdoc = fixdoc,
    		      printsum = FALSE)
    }


    # Store omegas
    output.se.omega[, k] <- est$omega
    # Store Bs
    output.se.b[, k] <- est$b
    }


    conf.documents <- matrix(0, nparty, 4)
    colnames(conf.documents) <- c("LB", "UB", "Omega: ML", "Omega: Sim Mean")
    rownames(conf.documents) <- rownames(dta)
    for (i in 1:nparty) {
    conf.documents[i, 1] <- quantile(output.se.omega[i, ], 0.025)
    conf.documents[i, 2] <- quantile(output.se.omega[i, ], 0.975)
    conf.documents[i, 3] <- omega[i]
    conf.documents[i, 4] <- mean(output.se.omega[i, ])
    }



    # CI for word weights
    conf.words <- matrix(0, nword, 4)
    colnames(conf.words) <- c("LB", "UB", "B: ML", "B: Sim Mean")
    rownames(conf.words) <- words


    for (i in 1:nword) {
    conf.words[i, 1] <- quantile(output.se.b[i, ], 0.025)
    conf.words[i, 2] <- quantile(output.se.b[i, ], 0.975)
    conf.words[i, 3] <- b[i]
    conf.words[i, 4] <- mean(output.se.b[i, ])
    }

    return(list(conf.documents = conf.documents, conf.words = conf.words))
    }

    if (boots == TRUE) {
    bootresult <- bootstrap(nsim, output.documents, output.words, nparty, nword)
    ci.documents <- bootresult$conf.documents
    ci.words <- bootresult$conf.words

    if (writeout == TRUE) {
    write.table(ci.words, file = paste(output, "words_95_ci.csv", sep = "_"))
    write.table(ci.documents, file = paste(output, "documents_95_ci.csv",
    				     sep = "_"))
    }

    }

    if (boots == F) {
    ci.documents <- NULL
    ci.words <- NULL
    }

    cat("Finished!\n")

    return(list(documents = output.documents, words = output.words, diffllik = est$diffllik,
          diffomega = est$diffparam, maxllik = est$maxllik, estimation = output.estimation,
          ci.documents = ci.documents, ci.words = ci.words))


    }

    # b) Wordfish 2 (fixed beta)
    wordfish2 <- function(words_weight,
    	      input,
    	      wordsincol = FALSE,
    	      fixtwo     = FALSE,
    	      dir        = NULL,
    	      fixdoc     = c(1, 2, 0, 1),
    	      tol        = 1e-07,
    	      sigma      = 3,
    	      boots      = FALSE,
    	      nsim       = 500,
    	      writeout   = FALSE,
    	      output     = "wordfish_output") {
    dta <- input
    if (wordsincol == TRUE) {
    rownames(dta) <- dta[, 1]
    dta <- dta[, -c(1)]
    }

    dta    <- t(dta)
    words  <- colnames(dta)
    nparty <- nrow(dta)
    nword  <- ncol(dta)

    if (fixtwo == TRUE) {
    if (fixdoc[3] == fixdoc[4]) {
    cat("Warning: fixed omega values in 'fixdoc' cannot be identical. \n")
    stop()
    }

    identprint <- paste("Omegas identified with", rownames(dta)[fixdoc[1]], "=",
    		fixdoc[3], "and ", rownames(dta)[fixdoc[2]], "=", fixdoc[4])
    } else {

    if (sum(c(length(dir) == 2, is.numeric(dir))) != 2) {
    cat("Warning: option 'dir' in wordfish2() is empty. You must specify two documents for global identification (e.g. dir=c(1,2) ).\n")
    stop()
    }

    identprint <- paste("Omegas identified with mean 0, st.dev. 1")
    }


    cat("======================================\n")
    cat("WORDFISH 2 DLA \n")
    cat("======================================\n")
    cat("Number of unique words: ", nword, "\n")
    cat("Number of documents: ", nparty, "\n")
    cat("Tolerance criterion: ", tol, "\n")
    cat("Identification: ", identprint, "\n")
    cat("======================================\n")

    # Generate starting values ========================

    if (fixtwo == FALSE) {



    rockingstarts <- function(dta) {
    cat("Performing mean 0 sd 1 starting value calc\n")
    P <- nrow(dta)
    W <- ncol(dta)
    numword <- rep(1:W, each = P)
    numparty <- rep(1:P, W)
    dat <- matrix(1, nrow = W * P, ncol = 3)
    dat[, 1] <- as.vector(as.matrix(dta))
    dat[, 2] <- as.vector(numword)
    dat[, 3] <- as.vector(numparty)
    dat <- data.frame(dat)
    colnames(dat) <- c("y", "word", "party")
    dat$word <- factor(dat$word)
    dat$party <- factor(dat$party)

    # Starting values for psi print(dta)
    psi <- log(colMeans(dta))
    # Starting values for alpha
    alpha <- log(rowMeans(dta)/rowMeans(dta)[1])

    # Starting values for beta and x
    ystar <- log(dat$y + 0.1) - alpha[dat$party] - psi[dat$word]
    # print(head(scale(matrix(ystar,nrow(dta),ncol(dta),byrow=FALSE))))
    res <- svd(matrix(ystar, nrow(dta), ncol(dta), byrow = FALSE), nu = 1)
    b <- as.vector(res$v[, 1] * res$d[1])

    omega1 <- as.vector(res$u) - res$u[1, 1]
    omega <- omega1/sd(omega1)
    b <- b * sd(omega1)



    # Create holding bins for some stuff for the convergence code
    min1 <- c(rep(1, nrow(dta) - 1))
    min2 <- c(rep(1, ncol(dta)))
    iter <- 0
    conv <- 0
    diffparam <- 0

    # Put everything together in a list
    list(alpha = as.vector(alpha), psi = as.vector(psi), b = b, omega = omega,
       min1 = min1, min2 = min2, iter = iter, conv = conv, diffparam = diffparam)
    }

    } else {


    rockingstarts <- function(dta, fixval) {
    cat("Performing fix two omega starting value calc\n")
    P <- nrow(dta)
    W <- ncol(dta)
    numword <- rep(1:W, each = P)
    numparty <- rep(1:P, W)
    dat <- matrix(1, nrow = W * P, ncol = 3)
    dat[, 1] <- as.vector(as.matrix(dta))
    dat[, 2] <- as.vector(numword)
    dat[, 3] <- as.vector(numparty)
    dat <- data.frame(dat)
    colnames(dat) <- c("y", "word", "party")
    dat$word <- factor(dat$word)
    dat$party <- factor(dat$party)

    # Starting values for psi
    psi <- log(colMeans(dta))
    # Starting values for alpha
    alpha <- log(rowMeans(dta)/rowMeans(dta)[1])

    # Starting values for beta and x
    ystar <- log(dat$y + 0.1) - alpha[dat$party] - psi[dat$word]
    res <- svd(matrix(ystar, nrow(dta), ncol(dta), byrow = FALSE), nu = 1)
    b <- as.vector(res$v[, 1] * res$d[1])

    omega <- as.vector(res$u)




    # Create holding bins for some stuff for the convergence code
    min1 <- c(rep(1, nrow(dta) - 1))
    min2 <- c(rep(1, ncol(dta)))
    iter <- 0
    conv <- 0
    diffparam <- 0

    # Put everything together in a list
    list(alpha = as.vector(alpha), psi = as.vector(psi), b = b, omega = omega,
       min1 = min1, min2 = min2, iter = iter, conv = conv, diffparam = diffparam)
    }


    }


    # Log-Likelihood Functions (Poisson model)
    # ========================================

    llik_psi_b <- function(p, y, omega, alpha, sigma) {
    # beta and psi will be estimated
    b <- p[1]
    psi <- p[2]
    lambda <- exp(psi + alpha + b * omega)  # Lambda parameter for Poisson distribution
    -(sum(-lambda + log(lambda) * y) - 0.5 * (b^2/sigma^2))  # Log-likelihood including normal prior on Beta
    }


    llik_alpha_1 <- function(p, y, b, psi) {
    # omega[1] is estimated
    omega <- p[1]
    lambda <- exp(psi + b * omega)  # Lambda parameter; alpha is excluded b/c it is set to be zero
    -sum(-lambda + log(lambda) * y)  # Log-likelihood
    }

    llik_alpha_omega <- function(p, y, b, psi) {
    # all other omegas and alphas are estimated
    omega <- p[1]
    alpha <- p[2]
    lambda <- exp(psi + alpha + b * omega)  # Lambda parameter
    -sum(-lambda + log(lambda) * y)  # Log-likelihood
    }


    llik_justalpha <- function(p, y, b, psi, omega) {
    # alpha is estimated
    alpha <- p[1]
    lambda <- exp(psi + alpha + b * omega)  # Lambda parameter
    -sum(-lambda + log(lambda) * y)  # Log-likelihood
    }





    if (fixtwo == FALSE) {


    cat("Performing mean 0 sd 1 EM algorithm\n")
    # Expectation-Maximization Algorithm FOR MEAN 0, SD 1 IDENTIFICATION
    # ==================================================================

    rockingpoisson <- function(dta, tol, sigma, params = NULL, dir = dir, printsum = TRUE) {

    P <- nrow(dta)
    W <- ncol(dta)

    if (is.null(params)) {
    params <- rockingstarts(dta)  # Call up starting value calculation
    }

    iter <- 2
    maxllik <- cbind(-1e+70, rep(0, 1400))
    ll.words <- matrix(-1e+70, W, 1400)
    diffllik <- 500

    # Set the convergence criterion
    conv <- tol
    params$conv <- conv

    while (diffllik > conv) {
    # Run algorithm if difference in LL > convergence criterion
    omegaprev <- params$omega
    bprev <- words_weight
    alphaprev <- params$alpha
    psiprev <- params$psi

    # ESTIMATE OMEGA AND ALPHA

    if (printsum == TRUE) {
      cat("Iteration", iter - 1, "\n")
      cat("\tUpdating alpha and omega..\n")
    }


    # Estimate first omega (alpha is set to 0)
    resa <- optim(p = c(params$omega[1]), fn = llik_alpha_1, y = as.numeric(dta[1,
    									]), b = words_weight, psi = params$psi, method = c("BFGS"))
    params$omega[1] <- resa$par[1]
    params$min1[1] <- -1 * resa$value
    params$alpha[1] <- 0
    ifelse(resa$convergence != 0, print("Warning: Optim Failed to Converge!"),
           NA)


    # Estimate all other omegas and alphas
    for (i in 2:P) {
      resa <- optim(par = c(params$omega[i], params$alpha[i]), fn = llik_alpha_omega,
    		y = as.numeric(dta[i, ]), b = words_weight, psi = params$psi)
      params$omega[i] <- resa$par[1]
      params$alpha[i] <- resa$par[2]
      params$min1[i] <- -1 * resa$value
      ifelse(resa$convergence != 0, print("Warning: Optim Failed to Converge!"),
    	 NA)

    }

    flush.console()


    # Z-score transformation of estimates for omega (to identify model)
    omegabar <- mean(params$omega)
    b1 <- words_weight
    params$omega <- (params$omega - omegabar)/sd(params$omega)
    params$psi <- params$psi + b1 * omegabar

    # Global identification
    if (params$omega[dir[1]] > params$omega[dir[2]]) {
      params$omega <- params$omega * (-1)
    }



    # ESTIMATE PSI AND BETA
    if (printsum == TRUE) {
      cat("\tUpdating psi and beta..\n")
    }

    for (j in 1:W) {
      resb <- optim(par = c(words_weight[j], params$psi[j]), fn = llik_psi_b,
    		y = dta[, j], omega = params$omega, alpha = params$alpha, sigma = sigma)
      params$psi[j] <- resb$par[2]
      params$min2[j] <- -1 * resb$value
      ifelse(resa$convergence != 0, print("Warning: Optim Failed to Converge!"),
    	 NA)
    }

    flush.console()

    # Calculate Log-Likelihood
    maxllik[iter] <- sum(params$min2)
    diffparam <- mean(abs(params$omega - omegaprev))  # difference btw current & previous estimate for omega

    ll.words[, iter] <- params$min2
    diff.ll.words <- (ll.words[, iter] - ll.words[, iter - 1])
    diffllik <- sum(diff.ll.words)/abs(maxllik[iter])


    if (printsum == TRUE) {
      # print(sum(diff.ll.words)) print(abs(maxllik[iter]))
      cat("\tConvergence of LL: ", diffllik, "\n")
    }

    params$diffllik[iter - 1] <- diffllik
    params$diffparam[iter - 1] <- diffparam
    params$diffparam.last <- diffparam
    params$maxllik[iter - 1] <- maxllik[iter]
    params$iter <- iter - 1
    iter <- iter + 1
    }
    params$diffllik[1] <- NA
    return(params)
    }

    # Run the algorithm
    est <- rockingpoisson(dta, tol, sigma, dir = dir)
    } else {
    cat("Performing fix two omega EM algorithm\n")

    # Expectation-Maximization Algorithm FOR TWO FIXED OMEGAS
    # ==================================================================

    rockingpoisson <- function(dta, tol, sigma, params = NULL, fixdoc = fixdoc,
    		       printsum = TRUE) {

    P <- nrow(dta)
    W <- ncol(dta)

    if (is.null(params)) {
    params <- rockingstarts(dta, fixval = fixdoc)  # Call up starting value calculation
    }

    iter <- 2
    maxllik <- cbind(-1e+70, rep(0, 1000))
    ll.words <- matrix(-1e+70, W, 1000)

    diffllik <- 500

    # Set the convergence criterion
    conv <- tol
    params$conv <- conv

    while (diffllik > conv) {
    # Run algorithm if difference in LL > convergence criterion
    omegaprev <- params$omega
    bprev <- params$b
    alphaprev <- params$alpha
    psiprev <- params$psi

    # ESTIMATE OMEGA AND ALPHA

    if (printsum == TRUE) {
      cat("Iteration", iter - 1, "\n")
      cat("\tUpdating alpha and omega..\n")
    }


    # Set omegas and first alpha

    params$omega[fixdoc[1]] <- fixdoc[3]
    params$omega[fixdoc[2]] <- fixdoc[4]
    params$alpha[1] <- 0


    if (1 %in% fixdoc[1:2] == TRUE) {

      # if first doc is one of the fixed omegas, do nothing (alpha and omega are fixed)

    } else {
      # Estimate first omega (alpha is set to 0)
      resa <- optim(p = c(params$omega[1]), fn = llik_alpha_1, y = as.numeric(dta[1,
    									  ]), b = params$b, psi = params$psi, method = c("BFGS"))
      params$omega[1] <- resa$par[1]
      params$min1[1] <- -1 * resa$value
      params$alpha[1] <- 0
      ifelse(resa$convergence != 0, print("Warning: Optim Failed to Converge!"),
    	 NA)
    }




    # Estimate all other omegas and alphas
    for (i in 2:P) {


      if (sum(fixdoc[1:2] == i) == 1) {

        # Estimate just alpha
        resa <- optim(par = params$alpha[i], fn = llik_justalpha, y = as.numeric(dta[i,
    									     ]), b = params$b, psi = params$psi, omega = params$omega[i],
    		  method = c("BFGS"))
        params$alpha[P] <- resa$par[1]

        ifelse(resa$convergence != 0, print("Warning: Optim Failed to Converge!"),
    	   NA)

      } else {
        resa <- optim(par = c(params$omega[i], params$alpha[i]), fn = llik_alpha_omega,
    		  y = as.numeric(dta[i, ]), b = params$b, psi = params$psi)
        params$omega[i] <- resa$par[1]
        params$alpha[i] <- resa$par[2]
        params$min1[i] <- -1 * resa$value
        ifelse(resa$convergence != 0, print("Warning: Optim Failed to Converge!"),
    	   NA)
      }

    }


    flush.console()



    # ESTIMATE PSI AND BETA
    if (printsum == TRUE) {
      cat("\tUpdating psi and beta..\n")
    }

    for (j in 1:W) {
      resb <- optim(par = c(params$b[j], params$psi[j]), fn = llik_psi_b,
    		y = dta[, j], omega = params$omega, alpha = params$alpha, sigma = sigma)
      params$b[j] <- resb$par[1]
      params$psi[j] <- resb$par[2]
      params$min2[j] <- -1 * resb$value
      ifelse(resa$convergence != 0, print("Warning: Optim Failed to Converge!"),
    	 NA)
    }

    flush.console()

    # Calculate Log-Likelihood
    maxllik[iter] <- sum(params$min2)
    diffparam <- mean(abs(params$omega - omegaprev))  # difference between current and previous estimate for omega

    ll.words[, iter] <- params$min2
    diff.ll.words <- (ll.words[, iter] - ll.words[, iter - 1])
    diffllik <- sum(diff.ll.words)/abs(maxllik[iter])

    # print(sum(diff.ll.words)) print(abs(maxllik[iter]))
    if (printsum == TRUE) {
      cat("\tConvergence of LL: ", diffllik, "\n")
    }

    params$diffllik[iter - 1] <- diffllik
    params$diffparam[iter - 1] <- diffparam
    params$diffparam.last <- diffparam
    params$maxllik[iter - 1] <- maxllik[iter]
    params$iter <- iter - 1
    iter <- iter + 1
    }
    params$diffllik[1] <- NA
    return(params)
    }

    # Run the algorithm
    est <- rockingpoisson(dta, tol, sigma, fixdoc = fixdoc)
    }

    cat("======================================\n")
    cat("WORDFISH 2 DLA ML Estimation finished.\n")
    cat("======================================\n\n")

    # Write output
    output.documents <- cbind(est$omega, est$alpha)
    rownames(output.documents) <- rownames(dta)
    colnames(output.documents) <- c("omega", "alpha")

    # Write estimation output file Include: Log-likelihood, iterations, number of
    # words, number of documents

    output.estimation <- cbind(nword, nparty, est$iter, sum(est$min2), est$conv,
    		     est$diffparam.last)
    colnames(output.estimation) <- c("Words", "Documents", "Iterations", "Log-Likelihood",
    			   "Convergence Criterion", "Difference in X")

    if (writeout == TRUE) {
    write.table(output.documents, file = paste(output, "documents.csv", sep = "_"))
    write.table(output.words, file = paste(output, "words.csv", sep = "_"))
    write.table(output.estimation, file = paste(output, "estimation.csv", sep = "_"))
    }

    ########################### Parametric Bootstrap Code

    bootstrap <- function(nsim, output.documents, output.words, nparty, nword) {

    cat("STARTING PARAMETRIC BOOTSTRAP\n")

    # input alpha and omega from estimation
    alpha.omega <- output.documents

    # input psis and betas from estimation
    psi.beta <- output.words

    # Create matrix of results.
    output.se.omega <- matrix(0, nparty, nsim)
    output.se.b <- matrix(0, nword, nsim)

    alpha <- alpha.omega[, 2]
    omega <- alpha.omega[, 1]
    psi <- psi.beta[, 2]
    b <- psi.beta[, 1]

    # create data matrix
    dtasim <- matrix(1, nrow = nparty, ncol = nword)
    cat("======================================\n")
    cat("Now running", nsim, "bootstrap trials.\n")
    cat("======================================\n")
    cat("Simulation ")

    for (k in 1:nsim) {

    cat(k, "...")

    # Generate new data using lambda
    for (i in 1:nparty) {
    dtasim[i, ] <- rpois(nword, exp(psi + alpha[i] + b * omega[i]))
    }

    alphastart <- alpha + rnorm(length(alpha.omega[, 1]), mean = 0, sd = (sd(alpha.omega[,
    								       2])/2))
    omegastart <- omega + rnorm(length(alpha.omega[, 1]), mean = 0, sd = (sd(alpha.omega[,
    								       1])/2))
    psistart <- psi + rnorm(length(psi.beta[, 1]), mean = 0, sd = (sd(psi.beta[,
    								2])/2))
    bstart <- b + rnorm(length(psi.beta[, 1]), mean = 0, sd = (sd(psi.beta[,
    							    1])/2))
    params <- list(alpha = alphastart, omega = omegastart, psi = psistart,
    	     b = bstart)


    if (fixtwo == FALSE) {
    est <- rockingpoisson(dtasim, tol, sigma, params = params, dir = dir,
    		      printsum = FALSE)
    } else {
    est <- rockingpoisson(dtasim, tol, sigma, params = params, fixdoc = fixdoc,
    		      printsum = FALSE)
    }


    # Store omegas
    output.se.omega[, k] <- est$omega
    # Store Bs
    output.se.b[, k] <- words_weight
    }


    conf.documents <- matrix(0, nparty, 4)
    colnames(conf.documents) <- c("LB", "UB", "Omega: ML", "Omega: Sim Mean")
    rownames(conf.documents) <- rownames(dta)
    for (i in 1:nparty) {
    conf.documents[i, 1] <- quantile(output.se.omega[i, ], 0.025)
    conf.documents[i, 2] <- quantile(output.se.omega[i, ], 0.975)
    conf.documents[i, 3] <- omega[i]
    conf.documents[i, 4] <- mean(output.se.omega[i, ])
    }



    # CI for word weights
    conf.words <- matrix(0, nword, 4)
    colnames(conf.words) <- c("LB", "UB", "B: ML", "B: Sim Mean")
    rownames(conf.words) <- words


    for (i in 1:nword) {
    conf.words[i, 1] <- quantile(output.se.b[i, ], 0.025)
    conf.words[i, 2] <- quantile(output.se.b[i, ], 0.975)
    conf.words[i, 3] <- b[i]
    conf.words[i, 4] <- mean(output.se.b[i, ])
    }

    return(list(conf.documents = conf.documents, conf.words = conf.words))
    }

    if (boots == TRUE) {
    bootresult <- bootstrap(nsim, output.documents, output.words, nparty, nword)
    ci.documents <- bootresult$conf.documents
    ci.words <- bootresult$conf.words

    if (writeout == TRUE) {
    write.table(ci.words, file = paste(output, "words_95_ci.csv", sep = "_"))
    write.table(ci.documents, file = paste(output, "documents_95_ci.csv",
    				     sep = "_"))
    }

    }

    if (boots == F) {
    ci.documents <- NULL
    ci.words <- NULL
    }


    cat("Finished!\n")

    return(list(documents = output.documents, diffllik = est$diffllik, diffomega = est$diffparam,
          maxllik = est$maxllik, estimation = output.estimation, ci.documents = ci.documents,
          ci.words = ci.words))

    }



    ### Decide which file to load Loading parameters
    if (user_mode == "politicians") {
    include_users <- F  # load user file
    include_party <- T  # load party file
    } else if (user_mode == "users") {
    include_users <- T  # load user file
    include_party <- (wf_mode == "wgt" || wf_mode == "wgt2")
    }
    mix_tm <- F  # mix the two tm matrices (not used any more, set to False)

    ##### 1- Filter USER DATA
    if (include_users && wf_mode != "wgt" && wf_mode != "wgt2") {
    # Load tables
    wordcountdata_users <- as.data.frame(data.table::fread(TM_file_users), , stringsAsFactors = FALSE)
    tweeters_users <- read.csv(tw_users_file, stringsAsFactors = FALSE)

    # Extract specific columns
    tweeters_users <- tweeters_users[, "Twitter_ID"]
    words_users <- wordcountdata_users[, 1]
    TM_users <- wordcountdata_users[, -1]
    npol_users <- ncol(TM_users)

    # Check that initialization works
    if (npol_users != length(tweeters_users)) {
    stop("Dimension problem of users input files")
    }


    # a) Keep words that appear in enough users documents
    kept_words_users <- (rowSums(TM_users > 0) > in_min_docs_users)
    TM_users <- TM_users[kept_words_users, ]
    words_users <- words_users[kept_words_users]

    # b) Keep users with enough specific words
    kept_tweeters_users <- (colSums(TM_users) > min_words_users)
    TM_users <- TM_users[, kept_tweeters_users]
    tweeters_users <- tweeters_users[kept_tweeters_users]


    }

    ### 2- Filter PARTY DATA
    if (include_party) {
    # Load tables
    wordcountdata_party <- as.data.frame(data.table::fread(TM_file_party), stringsAsFactors = FALSE)
    tweeters_party <- read.csv(tw_party_file, stringsAsFactors = FALSE)

    # Extract specific columns
    tweeters_party <- tweeters_party[, "Twitter_ID"]
    words_party <- wordcountdata_party[, 1]
    TM_party <- wordcountdata_party[, -1]
    npol_party <- ncol(TM_party)

    # Check that initialization works
    if (npol_party != length(tweeters_party)) {
    stop("Dimension problem of parties input files")
    }

    # Keep only parties with specific position in init
    if (suppress_other) {
    df_init <- read.csv(file_init, stringsAsFactors = FALSE)
    engaged_parties <- tweeters_party %in% df_init$key
    TM_party <- TM_party[, engaged_parties]
    tweeters_party <- tweeters_party[engaged_parties]
    }

    # a) Keep words that appear in enough party documents
    kept_words_party <- (rowSums(TM_party > 0) > in_min_docs_party)
    TM_party <- TM_party[kept_words_party, ]
    words_party <- words_party[kept_words_party]

    # b) Keep parties with enough specific words
    kept_tweeters_party <- (colSums(TM_party) > min_words_party)
    TM_party <- TM_party[, kept_tweeters_party]
    tweeters_party <- tweeters_party[kept_tweeters_party]

    }


    ### 3 - Choose between (Users, Parties or Users,Parties)
    if (mix_tm) {
    mix_list <- mix.TM(TM_users, TM_party, words_users, words_party)
    tweeters <- c(tweeters_users, tweeters_party)
    words <- mix_list$words
    TM <- mix_list$TM
    in_min_docs <- c(in_min_docs_users, in_min_docs_party)
    min_words_per_tweeter <- c(min_words_party, min_words_party)
    } else if (user_mode == "politicians" || (wf_mode == "wgt" || wf_mode == "wgt2")) {
    tweeters <- tweeters_party
    words <- words_party
    TM <- TM_party
    in_min_docs <- in_min_docs_party
    min_words_per_tweeter <- min_words_party
    } else if (user_mode == "users") {
    tweeters <- tweeters_users
    words <- words_users
    TM <- TM_users
    in_min_docs <- in_min_docs_users
    min_words_per_tweeter <- min_words_users
    } else {
    stop("Include at least users or parties")
    }

    # Raise in_min_doc_users if these values are non-zeros
    sum(rowSums(TM > 0) == 0)
    sum(colSums(TM > 0) == 0)
    dim(TM)


    # Run Wordifsh -----------------------------------------------------------------
    if (wf_mode == "wf" || (wf_mode == "wgt" || wf_mode == "wgt2")) {

    wf_out <- wordfish(TM, fixtwo = FALSE, dir = c(1, 2), wordsincol = FALSE, tol = 1e-04)
    omega <- wf_out$documents[, "omega"]
    beta <- wf_out$words[, "b"]
    psi <- wf_out$words[, "psi"]

    } else if (wf_mode == "ca") {
    TM_mat <- as.matrix(TM)
    res <- ca::ca(TM_mat)
    res_users <- data.frame(res$colcoord[, chosen_dim])
    omega <- res_users[, 1]

    } else {
    stop("Problem with wf_mode")
    }

    # Handle users weight using politicians information
    if(user_mode == 'users' && (wf_mode == 'wgt' || wf_mode == 'wgt2') ){

    wordcountdata_users<- as.data.frame(data.table::fread(TM_file_users),,stringsAsFactors=FALSE)
    tweeters_users     <-read.csv(tw_users_file, stringsAsFactors=FALSE)

    # 1) Create word-weight dataframe
    word_df <- data.frame(words,beta)

    # 2) Associate weights to words
    wordcountdata_users_weighted <- merge(wordcountdata_users,word_df,by.x = "V1",by.y = "words")
    L <- dim(wordcountdata_users_weighted)[2]
    words_weighted    <- wordcountdata_users_weighted[,1]
    TM_users_weighted <- wordcountdata_users_weighted[,2:(L-1)]
    beta_weighted     <- wordcountdata_users_weighted[,L]

    # 3) Suppress columns with zeros
    non_zero_idx <- (colSums(TM_users_weighted != 0) > min_words_users)
    TM_users_weighted <- TM_users_weighted[, non_zero_idx]
    tweeters_users_weighted <- tweeters_users[,'Twitter_ID']
    tweeters_users_weighted <- tweeters_users_weighted[non_zero_idx]

    # 3bis) Suppress lines with zeros
    kept_words_users_weighted <- (rowSums(TM_users_weighted > 0) > 0)
    TM_users_weighted <- TM_users_weighted[kept_words_users_weighted,]
    words_weighted     <- words_weighted[kept_words_users_weighted]
    beta_weighted      <- beta_weighted[kept_words_users_weighted]


    # 4) Compute opinion for each user
    if(wf_mode == 'wgt'){
    opinions <- (t(as.matrix(beta_weighted)) %*% as.matrix(TM_users_weighted))
    opinions <- opinions[1,] / as.matrix(colSums(TM_users_weighted))
    opinions <- as.numeric(opinions)
    }else if(wf_mode == 'wgt2'){
    sum(rowSums(TM_users_weighted > 0) == 0)
    sum(colSums(TM_users_weighted > 0) == 0)
    dim(TM)
    wf2_out <- wordfish2(beta_weighted,TM_users_weighted,fixtwo=FALSE,dir=c(1,2),wordsincol=FALSE,tol=1e-4)
    opinions  <- wf2_out$documents[,'omega']
    }

    # 5) Adapt names to save convention
    tweeters <- tweeters_users_weighted
    omega    <- opinions
    words    <- words_weighted
    beta     <- beta_weighted

    # 6) Define a TM matrix
    TM <- TM_users_weighted

    }


    # Save Output Files ------------------------------------------------------------

    # Write tweeters file
    write.table(data.frame(tweeters, omega), file = tweeters_file,
        sep = ",", col.names = c("key", "value"), row.names = F)
    write.table(data.frame(tweeters, omega), file = tweeters_file_test,
        sep = ",", col.names = c("key", "value"), row.names = F)

    # Write users file
    sort_beta = order(beta)
    sort_psi = order(psi)
    write.table(data.frame(words[sort_beta], beta[sort_beta], psi[sort_beta]),
        file = words_file, sep = ",",
        col.names = c("key", "weight", "popularity"), row.names = F)

    # Write parameters
    parameters <- data.frame(date, ngram, in_min_docs_users, min_words_users,
    		 in_min_docs_party, min_words_party, suppress_other, wf_mode, chosen_dim)
    write.table(parameters, file = parameters_file, sep = ",", row.names = F)
}
