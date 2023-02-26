using Flux
using Distributions: Uniform
using BioSequences
using IterTools


"""
    CRF

A Conditional Random Field (CRF). The struct contains a set of parameters
that can be any number of things. By default, the model contains parameters for k-mers in each
state/domain and transition parameters. 

A CRF with just emission and transition features is essentially a
Hidden Markov Model (HMM), except that the model is discriminative (classifier, p(y|x)) rather than
generative p(x,y).

"""
struct CRF
    θ_kmers::Vector{Float64}
    θ_transition::Vector{Float64}
end 

"""
    CRF(kmer_feature_size,transition_feature_size)

Function to initialize a new, untrained CRF model
"""
CRF(kmer_feature_size::Integer, transition_feature_size::Integer) =
  CRF(Flux.glorot_uniform(kmer_feature_size), Flux.glorot_uniform(transition_feature_size))


"""

(model::CRF)(x,y,states,kmer_features,transition_features,🐢)

Forward pass of the model and/or negative log-likelihood of the CRF.

Definition the function in this way allows a CRF object to be called as a function.

```julia-repl
> model = CRF(4,4)
> negativeloglikelihood = model(x,y,states,kmer_features,transition_features,🐢)
```


This function calculates the likelihood of a sequence given the current
parameter values. 

This includes the calculation of the partition function for the CRF model, 
i.e. the normalization constant to transform the score into a proper probability. 

The partition function is calculated using the forward algorithm (as in HMMs) in log-space.

This function therefore is the CRF likelihood function, and returns the negative
log-likelihood, such that we can perform gradient descent to optimize the feature function
parameters

🐢 -> L2 regularization scale constant. Sets the degree to which we wish to regularlize the parameters. Probably should not be passed in bayesian model?

"""
function (m::CRF)(x,y,
                    states,
                    kmer_features,
                    transition_features;
                    🐢 = .01,
                )

    ## sequence score given current parameters

    score = ∑(m.θ_kmers .* f(x[1],y[1],kmer_features))
    score += ∑(m.θ_transition .* f(y[1],y[1],transition_features))
    for t ∈ 2:lastindex(x)
        score += ∑(m.θ_kmers .* f(x[t],y[t],kmer_features))
        score += ∑(m.θ_transition .* f(y[t],y[t-1],transition_features))
    end


    ## partition function

    ## Gets the summed scores across all possible label
    ## sequences, such that the unnormalized score above
    ## becomes proper probability.

    U = vec([dot(m.θ_kmers,f(x[1],i,kmer_features)) for i in states])
    T = vec([dot(m.θ_transition,f(i,i,transition_features)) for i in states])
    α_prior = U+T 

    for t ∈ 2:lastindex(x)
        α_x = vec([dot(m.θ_kmers,f(x[t],j,kmer_features)) for j in states])
        α_y = vec([LSE(vec([dot(m.θ_transition,f(i,j,transition_features)) for j in states])) for i in states])
        α_prior = LSE(α_prior) .+ α_x .+ α_y
    end
    Z = LSE(α_prior)

    # calculation in log space, so (score - Z) is 
    # p(y|x,θ), or the probability of sequence labels
    # given the input sequence and model parameters θ
    return -(score - Z)  + (🐢 * (∑(m.θ_kmers.^2) + ∑(m.θ_transition.^2)))  # per-tok mean negative log likelihood + L2 reg
end


# function viterbi(model::CRF,x)
#     x
# end 


"""
    train!(model,input;args)

Takes an input CRF model and performs gradient descent. Updates model parameters in place over
N training epochs. The CRF likelihood function is convex, however a learning rate should be used
to avoid bouncing around the global minimum.

Input needs to be passed as a tuple that includes 

input = (x,y,labels,features...)


"""
function train!(model::CRF,
                input;
                N = 100,
                🐢 = .01,
                return_loss_curve = true)
    
    loss_history = Vector{Float32}(undef,N)

    for i ∈ 1:N
        local loss
        gs = gradient(Flux.params(model)) do
            loss = model(input...)
            return loss
        end
        loss_history[i] = loss
        for p ∈ Flux.params(model)
            p .-= 🐢 * gs[p]
        end
    end

end



