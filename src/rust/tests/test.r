library(glmnet)

PERF = function(y_test, y_hat, max_y) {
    n = length(y_test)
    if ((var(y_hat)==0) | (var(y_test)==0)) {
        cor = 0.0
    } else {
        cor = cor(y_test, y_hat)
    }
    e = y_test - y_hat
    mbe = ( sum(e)/n ) / max_y
    mae = ( sum(abs(e))/n ) / max_y
    mse = ( sum(e^2)/n ) / max_y
    rmse = ( sqrt(sum(e^2)/n) ) / max_y
    return(list(cor=cor, mbe=mbe, mae=mae, mse=mse, rmse=rmse, y_hat=y_hat, y_test=y_test))
}

OLS = function(x_train, x_test, y_train, y_test) {
    b_hat = t(x_train) %*% solve(x_train %*% t(x_train)) %*% y_train
    y_hat = (x_test %*% b_hat)[,1]
    y_max = max(c(y_test, y_train), na.rm=TRUE)
    # plot(abs(b_hat))
    # for (x in idx_b) {
    #     abline(v=x, col="red", lwd=2)
    # }
    return(PERF(y_test, y_hat, y_max))
}

LASSO = function(x_train, x_test, y_train, y_test) {
    mod_lasso = cv.glmnet(x=x_train, y=y_train, alpha=1.0)
    y_hat = predict(mod_lasso, newx=x_test, s="lambda.min")[,1]
    y_max = max(c(y_test, y_train), na.rm=TRUE)
    return(PERF(y_test, y_hat, y_max))
}

RIDGE = function(x_train, x_test, y_train, y_test) {
    mod_ridge = cv.glmnet(x=x_train, y=y_train, alpha=0.0)
    y_hat = predict(mod_ridge, newx=x_test, s="lambda.min")[,1]
    y_max = max(c(y_test, y_train), na.rm=TRUE)
    return(PERF(y_test, y_hat, y_max))
}

ELASTIC = function(x_train, x_test, y_train, y_test) {
    mod_elastic = cv.glmnet(x=x_train, y=y_train)
    y_hat = predict(mod_elastic, newx=x_test, s="lambda.min")[,1]
    y_max = max(c(y_test, y_train), na.rm=TRUE)
    return(PERF(y_test, y_hat, y_max))
}

rextendr::document(pkg="/data-weedomics-1/pflexnetr"); devtools::load_all("/data-weedomics-1/pflexnetr")
PFLEXNET = function(x_train, x_test, y_train, y_test) {
    mod_pflexnet = pflexnet(cbind(rep(1,nrow(x_train)), x_train),
                            y_train,
                            c(0:(nrow(x_train)-1)),
                            1.0,
                            FALSE,
                            0.1,
                            10)
    b_pflexnet = mod_pflexnet[[1]]
    print(paste0("length(b_pflexnet)=", sum(b_pflexnet != 0)))
    # print(paste0("b_pflexnet", b_pflexnet[b_pflexnet != 0]))
    alpha_pflexnet = mod_pflexnet[[2]]
    lambda_pflexnet = mod_pflexnet[[3]]
    y_hat = cbind(rep(1,nrow(x_test)), x_test) %*% b_pflexnet
    y_max = max(c(y_test, y_train), na.rm=TRUE)
    # PERF(y_test, y_hat, y_max)
    return(PERF(y_test, y_hat, y_max))
}


KFOLD_CV = function(x, y, r=5, k=10) {
    # k = 10
    ols = c()
    lasso = c()
    ridge = c()
    elastic = c()
    pflex = c()

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

            ols = c(ols, OLS(x_train, x_test, y_train, y_test))
            lasso = c(lasso, LASSO(x_train, x_test, y_train, y_test))
            ridge = c(ridge, RIDGE(x_train, x_test, y_train, y_test))
            elastic = c(elastic, ELASTIC(x_train, x_test, y_train, y_test))
            pflex = c(pflex, PFLEXNET(x_train, x_test, y_train, y_test))
            setTxtProgressBar(pb, ((rep-1)*k)+fold)
        }
    }
    close(pb)
    return(list(ols=ols,
                lasso=lasso,
                ridge=ridge,
                elastic=elastic,
                pflex=pflex))
}


##################
### UNIT TESTS ###
##################
vec_q = c()
vec_mod = c()
vec_cor = c()
vec_rmse = c()
vec_mbe = c()
for (q in c(1, 2, 3, 4, 5, 10, 50, 100, 500)) {
    # q = 10
    set.seed(123)
    n = 100
    p = 1000
    maf = 1e-4
    h2 = 0.75
    X_sim = matrix(runif(n*p, min=maf, max=1-maf), nrow=n)
    # X_sim = matrix(sample(c(0,1), size=n*p, replace=TRUE), nrow=n)
    b = rep(0, p)
    idx_b = sort(sample(c(1:p), q))
    # b[idx_b] = rnorm(q)
    # b[idx_b] = abs(rnorm(q))
    b[idx_b] = -abs(rnorm(q))
    xb = X_sim %*% b
    v_xb = var(xb)
    v_e = (v_xb/h2) - v_xb
    e = rnorm(n, mean=0, sd=sqrt(v_e))
    y = xb + e
    # y_sim = y
    # y_sim = scale(y, center=T, scale=T)[,1]
    # y_sim = 100 * (y - min(y)) / (max(y) - min(y))
    y_sim = 1 * (y - min(y)) / (max(y) - min(y))
    # y_sim = y - mean(y)
    # y_sim = y *100

    k = 10
    r = 3
    start_time = Sys.time()
    kfold_out = KFOLD_CV(x=X_sim, y=y_sim, k=k, r=r)
    end_time = Sys.time()
    print(paste0("Time elapsed: ", end_time - start_time))

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
        idx = plot_df$model == mod
        x = plot_df$y_test[idx]
        y = plot_df$y_hat[idx]
        vec_q = c(vec_q, q)
        vec_mod = c(vec_mod, mod)
        vec_cor = c(vec_cor, round(100*cor(x, y),2))
        vec_rmse = c(vec_rmse, sqrt(mean((x-y)^2)))
        vec_mbe = c(vec_mbe, mean(x-y))
        svg(paste0(mod, "-gp.svg"))
        plot(x=x, y=y, xlab="Observed", ylab="Predicted", pch=19, main=mod); grid()
        legend("topright", legend=paste0("cor = ", round(100*cor(x=x, y=y),2), "%"))
        dev.off()
    }
}
out = data.frame(q=vec_q, model=vec_mod, correlation=vec_cor, rmse=vec_rmse, mbe=vec_mbe)
print(out)

aggregate(correlation ~ model, data=out, FUN=mean)
aggregate(rmse ~ model, data=out, FUN=mean)


rextendr::document(pkg="/data-weedomics-1/pflexnetr"); devtools::load_all("/data-weedomics-1/pflexnetr")
vec_q = c()
vec_mod = c()
vec_cor = c()
vec_rmse = c()
vec_mbe = c()
q = 2
set.seed(123)
n = 100
p = 1000
maf = 1e-4
h2 = 0.75
X_sim = matrix(runif(n*p, min=maf, max=1-maf), nrow=n)
# X_sim = matrix(sample(c(0,1), size=n*p, replace=TRUE), nrow=n)
b = rep(0, p)
idx_b = sort(sample(c(1:p), q))
# b[idx_b] = rnorm(q)
b[idx_b] = +abs(rnorm(q))
xb = X_sim %*% b
v_xb = var(xb)
v_e = (v_xb/h2) - v_xb
e = rnorm(n, mean=0, sd=sqrt(v_e))
y = xb + e
y_sim = y
# y_sim = scale(y, center=T, scale=T)[,1]
# y_sim = 100 * (y - min(y)) / (max(y) - min(y))
# y_sim = -1 * (y - min(y)) / (max(y) - min(y))
# y_sim = y - max(y)
# y_sim = y *100

k = 10
r = 3
start_time = Sys.time()
kfold_out = KFOLD_CV(x=X_sim, y=y_sim, k=k, r=r)
end_time = Sys.time()
print(paste0("Time elapsed: ", end_time - start_time, " seconds."))

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
    idx = plot_df$model == mod
    x = plot_df$y_test[idx]
    y = plot_df$y_hat[idx]
    vec_q = c(vec_q, q)
    vec_mod = c(vec_mod, mod)
    vec_cor = c(vec_cor, round(100*cor(x, y),2))
    vec_rmse = c(vec_rmse, sqrt(mean((x-y)^2)))
    vec_mbe = c(vec_mbe, mean(x-y))
    svg(paste0(mod, "-gp.svg"))
    plot(x=x, y=y, xlab="Observed", ylab="Predicted", pch=19, main=mod); grid()
    legend("topright", legend=paste0("cor = ", round(100*cor(x=x, y=y),2), "%"))
    dev.off()
}
out = data.frame(q=vec_q, model=vec_mod, correlation=vec_cor, rmse=vec_rmse, mbe=vec_mbe)
print(out)
