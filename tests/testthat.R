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
        h2 = 0.75
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
        # y_sim = y
        x_train = X_sim[1:90, ]
        y_train = y_sim[1:90]
        x_test = X_sim[91:100, ]
        y_test = y_sim[91:100]
        out = pflexnet(x=cbind(rep(1,nrow(x_train)), x_train),
                        y=y_train,
                        row_idx=c(0:(nrow(x_train)-1)),
                        alpha=-1,
                        lambda_step_size=0.05,
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



        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        seed = 42069
        set.seed(seed)
        q = 20
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
                        alpha=1.0,
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


### More tests
y = rnorm(5)
X = cbind(rep(1, 5), matrix(rnorm(10), ncol=2))
b0 = solve(t(X) %*% X) %*% t(X) %*% y

y0 = X %*% b0
y - y0


x1 = cbind(rep(1, 5), X[,2])
x2 = cbind(rep(1, 5), X[,3])

b1 = solve(t(x1) %*% x1) %*% t(x1) %*% y
b2 = solve(t(x2) %*% x2) %*% t(x2) %*% y

(b1 + b2) / 2
b1 + b2

bh = c(mean(c(b1[1], b2[1])), b1[2], b2[2])
yh = X %*% bh

y - yh


print(sum((y - y0)^2))
print(sum((y - yh)^2))
