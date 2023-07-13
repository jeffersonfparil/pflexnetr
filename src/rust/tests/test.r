library(glmnetUtils)
library(parallel)

PERF = function(y_test, y_hat) {
    n = length(y_test)
    if ((var(y_hat)==0) | (var(y_test)==0)) {
        cor = 0.0
    } else {
        cor = cor(y_test, y_hat)
    }
    e = y_test - y_hat
    mbe = ( sum(e)/n )
    mae = ( sum(abs(e))/n )
    mse = ( sum(e^2)/n )
    rmse = ( sqrt(sum(e^2)/n) )
    return(list(cor=cor, mbe=mbe, mae=mae, mse=mse, rmse=rmse, y_hat=y_hat, y_test=y_test))
}

OLS = function(x_train, x_test, y_train, y_test) {
    b_hat = t(x_train) %*% solve(x_train %*% t(x_train)) %*% y_train
    y_hat = (x_test %*% b_hat)[,1]
    return(PERF(y_test, y_hat))
}

LASSO = function(x_train, x_test, y_train, y_test) {
    nfolds = 10
    # for (r in 1:10) {
    mod_lasso = cva.glmnet(
        x_train,
        y_train,
        alpha = 1.0,
        nfolds = nfolds,
        foldid = sample(rep(seq_len(nfolds), length = nrow(x_train))),
        outerParallel = NULL,
        checkInnerParallel = TRUE
        )
    y_hat = predict(mod_lasso$modlist[[1]], newx=x_test, s="lambda.min")[,1]
    return(PERF(y_test, y_hat))
}

RIDGE = function(x_train, x_test, y_train, y_test) {
    nfolds = 10
    mod_lasso = cva.glmnet(
        x_train,
        y_train,
        alpha = 0.0,
        nfolds = nfolds,
        foldid = sample(rep(seq_len(nfolds), length = nrow(x_train))),
        outerParallel = NULL,
        checkInnerParallel = TRUE
        )
    y_hat = predict(mod_lasso$modlist[[1]], newx=x_test, s="lambda.min")[,1]
    return(PERF(y_test, y_hat))
}

ELASTIC = function(x_train, x_test, y_train, y_test) {
    cores = parallel::makeCluster(detectCores())
    nfolds = 10
    vec_alphas = sort(unique(c(seq(0, 1, len=11)^3, 1-seq(0, 1, len=11)^3)))
    mod_elastic = cva.glmnet(
        x_train,
        y_train,
        alpha = vec_alphas,
        nfolds = nfolds,
        foldid = sample(rep(seq_len(nfolds), length = nrow(x_train))),
        outerParallel = cores,
        checkInnerParallel = TRUE
        )
    vec_mse = c()
    for (i in (1:length(vec_alphas))) {
        # i = 1
        mod = mod_elastic$modlist[[i]]
        y_hat = predict(mod, newx=x_test, s="lambda.min")[,1]
        vec_mse = c(vec_mse, PERF(y_test, y_hat)$mse)
    }
    cbind(vec_alphas, vec_mse)
    idx = which(vec_mse==min(vec_mse, na.rm=TRUE))
    y_hat = predict(mod_elastic$modlist[[idx]], newx=x_test, s="lambda.min")[,1]
    return(PERF(y_test, y_hat))
}

rextendr::document(pkg="/data-weedomics-1/pflexnetr"); devtools::load_all("/data-weedomics-1/pflexnetr")
PFLEXNET = function(x_train, x_test, y_train, y_test) {
    ### Destandardise
    standardise = ((abs(1-sd(c(y_train, y_test)))< 0.1) & (abs(mean(c(y_train, y_test))) < 0.1))
    if (standardise == TRUE) {
        sigma = 2.00 * sd(c(y_train, y_test))
        mu = 2.00 * sigma
        y_train = (y_train * sigma) + mu
        # x_train = (x_train * sigma) + mu
        y_test = (y_test * sigma) + mu
        # x_test = (x_test * sigma) + mu
    }
    mod_pflexnet = pflexnet(cbind(rep(1,nrow(x_train)), x_train),
                            y_train,
                            c(0:(nrow(x_train)-1)),
                            -1.0,
                            0.05,
                            1)
    b_pflexnet = mod_pflexnet[[1]]
    # print("####################################################")
    # print(paste0("length(b_pflexnet)=", sum(b_pflexnet != 0)))
    # print(paste0("b_pflexnet = [", paste(b_pflexnet[b_pflexnet != 0][1:min(c(4, sum(b_pflexnet != 0)))], collapse=", ") ,"... ]"))
    alpha_pflexnet = mod_pflexnet[[2]]
    lambda_pflexnet = mod_pflexnet[[3]]
    y_hat = cbind(rep(1,nrow(x_test)), x_test) %*% b_pflexnet
    ### Restandardise
    if (standardise == TRUE) {
        y_train = (y_train-mu)/sigma
        y_test = (y_test-mu)/sigma
        y_hat = (y_hat-mu)/sigma
     }
    return(PERF(y_test, y_hat))
}

PFLEXNET_L1 = function(x_train, x_test, y_train, y_test) {
    mod_pflexnet = pflexnet(cbind(rep(1,nrow(x_train)), x_train),
                            y_train,
                            c(0:(nrow(x_train)-1)),
                            1.0,
                            0.05,
                            1)
    b_pflexnet = mod_pflexnet[[1]]
    # print("####################################################")
    # print(paste0("length(b_pflexnet)=", sum(b_pflexnet != 0)))
    # print(paste0("b_pflexnet = [", paste(b_pflexnet[b_pflexnet != 0][1:min(c(4, sum(b_pflexnet != 0)))], collapse=", ") ,"... ]"))
    alpha_pflexnet = mod_pflexnet[[2]]
    lambda_pflexnet = mod_pflexnet[[3]]
    y_hat = cbind(rep(1,nrow(x_test)), x_test) %*% b_pflexnet
    return(PERF(y_test, y_hat))
}

PFLEXNET_L2 = function(x_train, x_test, y_train, y_test) {
    mod_pflexnet = pflexnet(cbind(rep(1,nrow(x_train)), x_train),
                            y_train,
                            c(0:(nrow(x_train)-1)),
                            0.0,
                            0.05,
                            1)
    b_pflexnet = mod_pflexnet[[1]]
    # print("####################################################")
    # print(paste0("length(b_pflexnet)=", sum(b_pflexnet != 0)))
    # print(paste0("b_pflexnet = [", paste(b_pflexnet[b_pflexnet != 0][1:min(c(4, sum(b_pflexnet != 0)))], collapse=", ") ,"... ]"))
    alpha_pflexnet = mod_pflexnet[[2]]
    lambda_pflexnet = mod_pflexnet[[3]]
    y_hat = cbind(rep(1,nrow(x_test)), x_test) %*% b_pflexnet
    return(PERF(y_test, y_hat))
}

KFOLD_CV = function(x, y, r=10, k=10, print_time=FALSE) {
    # k = 10
    ols = c()
    lasso = c()
    ridge = c()
    elastic = c()
    pflex = c()
    pflex_L1 = c()
    pflex_L2 = c()

    n = length(y)
    s = floor(n/k)
    idx = sample(c(1:n), n, replace=FALSE)

    pb = txtProgressBar(min=0, max=r*k, initial=0, style=3)
    for (rep in 1:r) {
        for (fold in 1:k) {
            # rep=1; fold=1
            i = (fold-1)*s + 1
            j = fold*s
            if (fold == k) {
                j = n
            }
            bool_test = c(1:n) %in% c(i:j)
            bool_train = !bool_test
            idx_train = idx[bool_train]
            idx_test = idx[bool_test]

            x_train = x[idx_train, ]
            x_test = x[idx_test, ]
            y_train = y[idx_train]
            y_test = y[idx_test]

            options(digits.secs=7)
            if (print_time==TRUE) {
                print("OLS:")
                start_time = Sys.time()
            }
            ols = c(ols, OLS(x_train, x_test, y_train, y_test))
            if (print_time==TRUE) {
                end_time = Sys.time()
                print(end_time - start_time)
                print("LASSO:")
                start_time = Sys.time()
            }
            lasso = c(lasso, LASSO(x_train, x_test, y_train, y_test))
            if (print_time==TRUE) {
                end_time = Sys.time()
                print(end_time - start_time)
                print("RIDGE:")
                start_time = Sys.time()
            }
            ridge = c(ridge, RIDGE(x_train, x_test, y_train, y_test))
            if (print_time==TRUE) {
                end_time = Sys.time()
                print(end_time - start_time)
                print("ELASTIC:")
                start_time = Sys.time()
            }
            elastic = c(elastic, ELASTIC(x_train, x_test, y_train, y_test))
            if (print_time==TRUE) {
                end_time = Sys.time()
                print(end_time - start_time)
                print("PFLEXNET:")
                start_time = Sys.time()
            }
            pflex = c(pflex, PFLEXNET(x_train, x_test, y_train, y_test))
            if (print_time==TRUE) {
                end_time = Sys.time()
                print(end_time - start_time)
                print("PFLEXNET_L1:")
                start_time = Sys.time()
            }
            pflex_L1 = c(pflex_L1, PFLEXNET_L1(x_train, x_test, y_train, y_test))
            if (print_time==TRUE) {
                end_time = Sys.time()
                print(end_time - start_time)
                print("PFLEXNET_L2:")
                start_time = Sys.time()
            }
            pflex_L2 = c(pflex_L2, PFLEXNET_L2(x_train, x_test, y_train, y_test))
            if (print_time==TRUE) {
                end_time = Sys.time()
                print(end_time - start_time)
            }
            setTxtProgressBar(pb, ((rep-1)*k)+fold)
        }
    }
    close(pb)
    return(list(ols=ols,
                lasso=lasso,
                ridge=ridge,
                elastic=elastic,
                pflex=pflex,
                pflex_L1=pflex_L1,
                pflex_L2=pflex_L2))
}


##################
### UNIT TESTS ###
##################
# rextendr::document(pkg="/data-weedomics-1/pflexnetr"); devtools::load_all("/data-weedomics-1/pflexnetr")
fn_test = function() {
    out = data.frame()
    set.seed(123)
    # for (q in c(1, 2, 3, 4, 5, 10, 50, 100, 500)) {
    for (q in c(2, 20)) {
        # q = 2
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(paste0("q=", q))
        vec_q = c()
        vec_y_scaled = c()
        vec_mod = c()
        vec_cor = c()
        vec_rmse = c()
        vec_mbe = c()
        n = 100
        p = 1e5
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
        # y_sim = scale(y, center=T, scale=T)[,1]

        ### Using Arabidopsis data
        library(reticulate)
        np = import("numpy")
        # data reading
        # X_sim = np$load("./res/flowering_time_10deg_ld_filtered_maf0.05_windowkb10_r20.8.npy")
        # y_sim = as.vector(np$load("./res/flowering_time_10degphenotype_values.npy"))
        X_sim = np$load("./res/herbavore_resistance_G2P_ld_filtered_maf0.05_windowkb10_r20.8.npy")
        y_sim = as.vector(np$load("./res/herbavore_resistance_G2Pphenotype_values.npy"))
        idx_col = seq(1, ncol(X_sim), length=1e4)

        k = 10
        r = 1

        options(digits.secs=7)
        start_time = Sys.time()
        kfold_out = KFOLD_CV(x=X_sim[, idx], y=y_sim, k=k, r=r, print_time=TRUE)
        # kfold_out = KFOLD_CV(x=X_sim, y=y_sim, k=k, r=r, print_time=FALSE)
        end_time = Sys.time()
        print(end_time - start_time)

        plot_model = c()
        plot_y_hat = c()
        plot_y_test = c()
        idx_steps = seq(from=1, to=(r*k*7), by=7)
        for (mod in names(kfold_out)) {
            # mod = names(kfold_out)[1]
            p_y_hat = c()
            p_y_test = c()
            for (i_ in idx_steps) {
                # i_ = 1
                # eval(parse(text=paste0(mod, " = c(", mod, ", kfold_out$", mod, "[i_:(i_+4)])")))
                p_y_hat = c(p_y_hat, unlist(eval(parse(text=paste0("kfold_out$", mod, "[i_+5]")))))
                p_y_test = c(p_y_test, unlist(eval(parse(text=paste0("kfold_out$", mod, "[i_+6]")))))
            }
            plot_model = c(plot_model, rep(mod, length(p_y_hat)))
            plot_y_hat = c(plot_y_hat, p_y_hat)
            plot_y_test = c(plot_y_test, p_y_test)
        }
        plot_df = data.frame(model=plot_model, y_hat=plot_y_hat, y_test=plot_y_test)
        for (mod in unique(plot_df$model)) {
            # mod = unique(plot_df$model)[1]
            idx = plot_df$model == mod
            x = plot_df$y_test[idx]
            y = plot_df$y_hat[idx]
            vec_q = c(vec_q, q)
            vec_mod = c(vec_mod, mod)
            vec_cor = c(vec_cor, round(100*cor(x, y),2))
            vec_rmse = c(vec_rmse, sqrt(mean((x-y)^2)))
            vec_mbe = c(vec_mbe, mean(x-y))
            svg(paste0(mod, "-q", q, "-gp.svg"))
            plot(x=x, y=y, xlab="Observed", ylab="Predicted", pch=19, main=mod); grid()
            legend("topright", legend=c(paste0("cor = ", round(100*cor(x=x, y=y),2), "%"),
                                        paste0("rmse = ", round(sqrt(mean((x-y)^2)),7))))
            dev.off()
        }
        df_out = data.frame(q=vec_q, model=vec_mod, correlation=vec_cor, rmse=vec_rmse, mbe=vec_mbe,
                            b_mean=mean(b[idx_b]),
                            X_col_min=min(colMeans(X_sim)),
                            X_col_max=max(colMeans(X_sim)),
                            X_col_mean=mean(colMeans(X_sim)),
                            X_col_var=var(colMeans(X_sim)))
        if (nrow(out) == 0) {
            out = df_out
        } else {
            out = rbind(out, df_out)
        }
    }
    print(out)
    aggregate(correlation ~ model, data=out, FUN=mean)
    aggregate(rmse ~ model, data=out, FUN=mean)
    # aggregate(mbe ~ model, data=out, FUN=mean)
}

# fn_test()

