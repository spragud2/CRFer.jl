

"""
    LSE(x)

Log-sum-exp optimized for large numbers. Necessary to properly sum log-probabilities
inside a likelihood function.

Also useful as a continuous and differentiable approximation to the maximum function!

"""
function LSE(x::Vector{Float64})
    c = maximum(x)
    return c + log(sum(exp.(x .- c)))
end

function standardize!(X::Matrix;dims=2)

    μ = mean(X,dims=dims)
    σ =  std(X,dims=dims) .+ .01

    @. X = (X - μ) / σ

end 


# Some convenient shorthand
∑ = sum
dot(a,b) = ∑(a .* b) 
