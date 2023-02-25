module CRFer


include("crf.jl")
include("utils.jl")
include("features.jl")
include("sequences.jl")



x = dna"ATCG"^4 * dna"TTTT"^4 
x = sequence_to_kmers(x,1)

f1 = []
f2 = []
labels = ['A' 'B']
emissions = kmers(1)
i = 1

y = "A"^16 * "B"^16
y = [i for i in y]

NLL = model(x,y,labels,kmer_features,transition_features)

# Allows the internal parameters of the CRF model to be updated
# during gradient descent
Flux.@functor CRF

model = CRF(8,4)
🐢 = .1
N = 100
loss_history = Vector{Float32}(undef,N)
parameters_initial, reconstruct = Flux.destructure(model)



end # module
