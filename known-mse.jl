type WeinerProcess
    a :: Float64
    σ :: Float64
end

function MSE(p :: WeinerProcess, T)
    x = 2*p.a*T  
    p.σ^2/(2*p.a) * ( ( exp(x) - 1 ) / x - 1)
end

function total_error(users::Vector{WeinerProcess}, rates)
    N = length(users)
    T = 1.0 ./rates
    sum([MSE(users[i], T[i]) for i in 1:N])
end


# We implement two functions to find the best response at the user. The first
# directly optimizes the user's objective, the sencond finds a root of the
# derivative being equal to zero. 

using Optim

user_objective(p :: WeinerProcess, T, λ) = MSE(p, T) + λ/T

function user_optim_A(p :: WeinerProcess, λ :: Float64)
   # I did not find an implementation of gradient search in Julia
   # The code below uses Brent's method (whatever that is)
   result = Optim.optimize(r -> user_objective(p, 1/r, λ), 0, 1e6)
   return Optim.minimizer(result)
end

using Roots
function user_optim_B(p::WeinerProcess, λ)
    f(T) = exp(2*p.a*T) * (2*p.a*T - 1) + 1 - 4*p.a^2*λ/p.σ^2
    root = fzero(r -> f(1/r), 0, 1e6) 
end

# There were some experiments where option `B` gave incorrect results. Option
# `A` is faster, which is an added benefit!
user_optim = user_optim_A

using DataFrames

function syncTrace(users::Vector{WeinerProcess}, C::Float64;
    iterations :: Int     = 1_000,
       initial :: Float64 = 1e-3,
        decay1 :: Float64 = 0.9,
	    decay2 :: Float64 = 0.999,
	   epsilon :: Float64 = 1e-8,
        alpha  :: Float64 = 0.01,
        )

    N = length(users)

    r = zeros(Float64, (iterations,N))
    Λ = zeros(Float64, iterations)
    err = zeros(Float64, iterations)

    Λ[1] = initial

    moment1 = 0.0
    moment2 = 0.0

    weight1 = decay1
    weight2 = decay2

    @inbounds for t in 1:iterations
        #r[t,:] = pmap(i -> user_optim(users[i], Λ[t]), 1:N)
        @inbounds for i in 1:N
            r[t,i] = user_optim(users[i], Λ[t])
        end
        err[t] = total_error(users, r[t,:])

        if t < iterations
            gradient = C - sum(r[t,:])

            moment1 = decay1 * moment1 + (1 - decay1) * gradient
            moment2 = decay2 * moment2 + (1 - decay2) * gradient^2

            corrected1 = moment1/(1 - weight1)
            corrected2 = moment2/(1 - weight2)

            weight1 *= decay1
            weight2 *= decay2

            delta  = corrected1 / ( sqrt(corrected2) + epsilon)

            Λ[t+1] = Λ[t] - alpha * delta
            Λ[t+1] = max(Λ[t+1], 1e-8)
        end
    end

    df = DataFrame()
    df[:iter] = 1:iterations
    df[:lambda] = Λ

    @inbounds for i in 1:N
        df[Symbol("r", i)] = r[:,i]
    end
    df[:C] = vec(sum(r,2))
    df[:MSE] = err
    return df
end

function asyncTrace(users::Vector{WeinerProcess}, C::Float64;
    iterations :: Int     = 1_000,
       initial :: Float64 = 1e-3,
        decay1 :: Float64 = 0.9,
	    decay2 :: Float64 = 0.999,
	   epsilon :: Float64 = 1e-8,
        alpha  :: Float64 = 0.01,
        )

    N = length(users)

    current_time = zeros(Float64, iterations)

    r  = zeros(Float64, (iterations,N))
    Λ  = zeros(Float64, iterations)
    tx = zeros(Int32, iterations)
    err = zeros(Float64, iterations)

    next_sampling_times = zeros(Float64, N)

    Λ[1] = initial

    moment1 = 0.0
    moment2 = 0.0

    weight1 = decay1
    weight2 = decay2

    current_time[1] = 0.0
    @inbounds for i in 1:N
        r[1,i] = user_optim(users[i], Λ[1]) + 1e-2*rand()
    end

    @inbounds for i in 1:N
        next_sampling_times[i] += 1.0/r[1,i]
    end

    @inbounds for k in 1:iterations
        err[k]  = total_error(users, r[k,:])

        (current_time[k], tx[k]) = findmin(next_sampling_times)

        if k < iterations
            gradient = C - sum(r[k,:])

            moment1 = decay1 * moment1 + (1 - decay1) * gradient
            moment2 = decay2 * moment2 + (1 - decay2) * gradient^2

            corrected1 = moment1/(1 - weight1)
            corrected2 = moment2/(1 - weight2)

            weight1 *= decay1
            weight2 *= decay2

            delta  = corrected1 / ( sqrt(corrected2) + epsilon)

            Λ[k+1] = Λ[k] - alpha * delta
            Λ[k+1] = max(Λ[k+1], 1e-8)

            r[k+1,:]     = r[k,:]
            r[k+1,tx[k]] = user_optim(users[tx[k]], Λ[k+1])
            next_sampling_times[tx[k]] += 1.0/r[k+1, tx[k]]
        end
    end

    df = DataFrame()
    df[:iter] = 1:iterations
    df[:lambda] = Λ

    @inbounds for i in 1:N
        df[Symbol("r", i)] = r[:,i]
    end
    df[:tx] = tx
    df[:C] = vec(sum(r,2))
    df[:time] = current_time
    df[:MSE] = err
    return df
end

using NLopt

function optimal_rates(users :: Vector{WeinerProcess}, C :: Float64)
    N = length(users)

    objective(rates, gradient)  = total_error(users, rates)
    constraint(rates, gradient) = C - sum(rates) 

    opt = Opt(:LN_COBYLA, N)

    lower_bounds!(opt, zeros(N))
    upper_bounds!(opt, C*ones(N))
    min_objective!(opt, objective)
    equality_constraint!(opt, constraint)

    guess = ones(N)./N

    (err, rates, ret) = NLopt.optimize(opt, guess)
end

users = [ WeinerProcess(1,1), WeinerProcess(1,1), WeinerProcess(1,1), 
          WeinerProcess(1,1), WeinerProcess(1,2)]

change = [ WeinerProcess(1,1), WeinerProcess(1,1), WeinerProcess(1,1), 
           WeinerProcess(1,2)]
