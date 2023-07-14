# This file is part of the standard setup for testthat.
# It is recommended that you do not modify it.
#
# Where should you do additional test configuration?
# Learn more about the roles of various files in:
# * https://r-pkgs.org/testing-design.html#sec-tests-files-overview
# * https://testthat.r-lib.org/articles/special-files.html

library(testthat)
# library(pflexnetr)
devtools::load_all()

test_that("pflexnetr", {
        seed = 42069
        set.seed(seed)
        n = 100
        p = 1000
        q = 2
        maf = 1e-4
        h2 = 0.9
        X_sim = matrix(runif(n*p, min=maf, max=1-maf), nrow=n)
        b = rep(0, p)
        idx_b = sort(sample(c(1:p), q))
        b[idx_b] = 1.0
        xb = X_sim %*% b
        v_xb = var(xb)
        v_e = (v_xb/h2) - v_xb
        e = rnorm(n, mean=0, sd=sqrt(v_e))
        y = xb + e
        y_sim = scale(y, center=T, scale=T)[,1]
        x_train = X_sim[1:90, ]
        y_train = y_sim[1:90]
        x_test = X_sim[91:100, ]
        y_test = y_sim[91:100]
        out = pflexnet(x=cbind(rep(1,nrow(x_train)), x_train),
                        y=y_train,
                        row_idx=c(0:(nrow(x_train)-1)),
                        alpha=-1.0,
                        lambda_step_size=0.1,
                        r=1)
        beta = out[[1]]
        alpha = out[[2]]
        lambda = out[[3]]
        n_non_zero = sum(beta != 0) - 1 ### less the intercept
        y_hat = (cbind(rep(1,nrow(x_test)), x_test) %*% beta)[,1]

        correlation = cor(y_test, y_hat)
        rmse = sqrt(mean((y_test-y_hat)^2))
        print(paste0("alpha=", alpha, "; ", 
                     "lambda=", lambda, "; ",
                     "non-zero=", n_non_zero, "; ",
                     "correlation=", correlation, "; ",
                     "rmse=", rmse))

        expect_equal(cor(y_test, y_hat), 0.8776987, tolerance=1e-7)

        seed = 42069
        set.seed(seed)
        q = 20
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(paste0("q=", q))
        vec_q = c()
        vec_y_scaled = c()
        vec_mod = c()
        vec_cor = c()
        vec_rmse = c()
        vec_mbe = c()
        n = 100
        p = 1e4
        maf = 1e-4
        h2 = 0.75
        X_sim = matrix(runif(n*p, min=maf, max=1-maf), nrow=n)
        b = rep(0, p)
        idx_b = sort(sample(c(1:p), q))
        # b[idx_b] = rnorm(q)
        b[idx_b] = abs(rnorm(q))
        # b[idx_b] = -abs(rnorm(q))
        xb = X_sim %*% b
        v_xb = var(xb)
        v_e = (v_xb/h2) - v_xb
        e = rnorm(n, mean=0, sd=sqrt(v_e))
        y = xb + e
        y_sim = y
        # y_sim = scale(y_sim, scale=TRUE, center=TRUE)

        x_train = X_sim[1:90, ]
        y_train = y_sim[1:90]
        x_test = X_sim[91:100, ]
        y_test = y_sim[91:100]
        
        options(digits.secs=7)
        start_time = Sys.time()

        out = pflexnet(x=cbind(rep(1,nrow(x_train)), x_train),
                        y=y_train,
                        row_idx=c(0:(nrow(x_train)-1)),
                        alpha=-1.0,
                        lambda_step_size=0.1,
                        r=1)

        end_time = Sys.time()
        print(end_time - start_time)

        beta = out[[1]]
        alpha = out[[2]]
        lambda = out[[3]]
        n_non_zero = sum(beta != 0) - 1 ### less the intercept
        cor(beta[idx_b+1], b[idx_b])
        y_hat = (cbind(rep(1,nrow(x_test)), x_test) %*% beta)[,1]
        correlation = cor(y_test, y_hat)
        rmse = sqrt(mean((y_test-y_hat)^2))
        print(paste0("alpha=", alpha, "; ", 
                     "lambda=", lambda, "; ",
                     "non-zero=", n_non_zero, "; ",
                     "correlation=", correlation, "; ",
                     "rmse=", rmse))

    }
)
