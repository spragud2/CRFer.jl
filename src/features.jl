"""
    f(a,b,features)

Identity function for CRF features in a sequence. Returns 1 where true and 0 everywhere else.

"""
function f(a,b,features)
    [(a,b)] .== features
end



kmer_features(emissions,labels) = vec([i for i in product(emissions,labels)])
transition_features(labels) = vec([i for i in product(labels,labels)])

