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

function standardize!(X::Matrix{Real})
    for col ∈ eachcol(X)
        μ = mean(col)
        σ = std(col,μ)
        col = @. (col - μ) / σ
    end 
end 


# Some convenient shorthand
∑ = sum
dot(a,b) = ∑(a .* b) 
mean(x) = ∑(x) / length(x)
std(x,μ) = √(∑((xi - μ)^2 for xi in x)/ length(x))