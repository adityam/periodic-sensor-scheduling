type WeinerProcess
    a :: Float64
    σ :: Float64
end

using Roots
using DataFrames

function user_optim(p::WeinerProcess, λ)
    f(T) = exp(2*p.a*T) * (2*p.a*T - 1) + 1 - 4*p.a^2*λ/p.σ^2
    root = fzero(r -> f(1/r), 0, 1e6) 
end

function generateTrace(users::Vector{WeinerProcess}, C::Float64;
            iterations=1000,
            λ=1e-2, α=0.1)

    N = length(users)

    r = zeros(Float64, (iterations,N))
    Λ = zeros(Float64, iterations)

    Λ[1] = λ

    for t in 1:iterations
        for i in 1:N
            r[t,i] = user_optim(users[i], Λ[t])
        end

        if t < iterations
            Λ[t+1] = Λ[t] - α*( C - sum(r[t,:]))
            Λ[t+1] = max(Λ[t+1], 1e-3)
        end
    end
    df = DataFrame()
    df[:lambda] = Λ
    for i in 1:N
        df[symbol("r", i)] = r[:,i]
    end
    return df
end

users = [ WeinerProcess(1,1), WeinerProcess(1,1), WeinerProcess(1,1) ]
