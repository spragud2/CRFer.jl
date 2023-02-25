module CRFer

using Comonicon

include("crf.jl")
include("utils.jl")
include("features.jl")
include("sequences.jl")


module model

using Comonicon
using FASTX

@cast train(x) = x
@cast predict(x) = x

end

module kmertools

using Comonicon


"""
Get k-mer frequencies for a set of RNA or DNA sequences.

# Args

- `seqs`: Path to sequence file
- `k`: Value of k 

# Flags
- `-r, --rna`: Return strings and kmers in RNA instead of DNA
- `-l, --lnorm`: Length normalize the k-mer counts to frequencies, default=True
- `-s, --std`: Standardize the k-mer frequencies to z-scores, default=True
"""
@cast function kmerfreqs(
                        seqs::String,
                        k::Int;
                        rna::Bool=false,
                        lnorm::Bool=true,
                        std::Bool=true,
                        )

    seqs = read_seqs(seqs;rna=rna)
    

    

    

end 


@cast function kmercorr(seqs;
                        k=4) 
    x
end


@cast umap(x) = x
@cast ppca(x) = x


end 

@cast model
@cast kmertools 



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


@main

end # module
