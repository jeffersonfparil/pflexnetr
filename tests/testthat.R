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
                        alpha=1.0,
                        lambda_step_size=0.05,
                        r=10)
        y_hat = (cbind(rep(1,nrow(x_test)), x_test) %*% out[[1]])[,1]
        expect_equal(cor(y_test, y_hat), 0.8776987, tolerance=1e-7)
    }
)
