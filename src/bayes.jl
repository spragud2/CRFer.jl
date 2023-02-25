using Turing


## Notes

## I think if we tell flux to ignore the gradient for trellis computation
## We can forward and backward passes and get a categorical distribution
## at each position and then feed that to the turing  model likelihood?


@model function bayesCRF(xs, ys, nparameters, reconstruct,
                        kmer_features,transition_features; alpha=0.09)
    # Create the weight and bias vector.
    parameters ~ MvNormal(Zeros(nparameters), I / alpha)

    # Construct NN from parameters
    crf = reconstruct(parameters)
    # Forward NN to make predictions
    preds = crf(xs,ys,labels,kmer_features,transition_features)

    # Observe each prediction.
    for i in 1:lastindex(ys)
        ys[i] ~ Bernoulli(preds[i])
    end
end;