type ADAM
    decay1  :: Float64
    decay2  :: Float64
    epsilon :: Float64
    alpha   :: Float64
    weight1 :: Float64
    weight2 :: Float64
    moment1 :: Float64
    moment2 :: Float64
end

ADAM(;decay1=0.9, decay2=0.999, epsilon=1e-8, alpha=0.01) =
        ADAM(decay1, decay2, epsilon, alpha, decay1, decay2, 0.0, 0.0)

function gradient_step!(adam, gradient)
    adam.moment1 = adam.decay1 * adam.moment1 + (1 - adam.decay1) * gradient
    adam.moment2 = adam.decay2 * adam.moment2 + (1 - adam.decay2) * gradient^2

    corrected1 = adam.moment1/(1 - adam.weight1)
    corrected2 = adam.moment2/(1 - adam.weight2)

    adam.weight1 *= adam.decay1
    adam.weight2 *= adam.decay2

    delta  = corrected1 / ( sqrt(corrected2) + adam.epsilon)

    return adam.alpha * delta
end

