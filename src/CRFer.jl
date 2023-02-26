module CRFer

using Comonicon

"""
Tools for kmers!

"""
module kmertools

using BioSequences
using Comonicon
using DataFrames
using CSV

include("sequences.jl")
include("utils.jl")

"""
Get k-mer frequencies for a set of RNA or DNA sequences.

# Args

- `seqs`: Path to sequence file
- `k`: Value of k 
- `o`: Output file
# Flags
- `-r, --rna`: Return strings and kmers in RNA instead of DNA
- `-l, --lnorm`: Disable length normalize the k-mer counts to frequencies, default=True
- `-s, --std`: Disable standardize the k-mer frequencies to z-scores, default=True
"""
@cast function freq(
                        seqs::String,
                        k::Int,
                        o::String;
                        rna::Bool=false,
                        lnorm::Bool=false,
                        std::Bool=false,
                    )

    if lnorm || std
        error("unimplemented")
    end

    nuctype = rna ? RNA : DNA
    ids,seqs = read_seqs(seqs;rna=rna)
    
    X = count_kmers(seqs,k;nuctype=nuctype)

    standardize!(X)

    out_df = DataFrame(X,ids,makeunique=true)

    CSV.write(o,out_df)

end 


@cast function kmercorr(seqs;
                        k=4) 
    x
end


@cast umap(x) = x
@cast ppca(x) = x
@cast LDA(x) = x

end 

@cast kmertools 


module crf

using Comonicon
using CSV
using DataFrames
using ProgressBars
using Plots

using BSON: @save, @load

include("sequences.jl")
include("utils.jl")
include("crf.jl")
include("features.jl")

"""
Get a CSV of sequence domain labels for a reference sequence `ref` using a second fasta file `queries` containing
regions of interest inside the reference.

The fasta ids need to be the identifier for the domain, i.e. "B" for both Xist repeat B1 & B2.

This provides the training set of domain labels for fitting the CRF model.

# Args

- `ref`: Path to fasta file containing single sequence to be labeled
- `queries`: Path to fasta file containing subsequences of interest (these sequences must be in the `ref` file!)

# Options
- `-o, --outfile`: Output file

"""
@cast function label(ref::String,queries::String;o::String="./labels.csv")
    ref_id,ref_seq = read_seqs(ref)
    @assert length(ref_seq) == 1

    ref_id,ref_seq = ref_id[1],ref_seq[1]


    ref_labels = ["Z" for _ ∈ 1:length(ref_seq)]

    query_ids,query_seqs = read_seqs(queries)

    for i ∈ 1:lastindex(query_ids)
        q = query_seqs[i]
        query = ExactSearchQuery(q)
        r = findall(query,ref_seq)
        ref_labels[r[1]] .= query_ids[i]

    end

    df = DataFrame([ref_labels],[ref_id])
    CSV.write(o,df)


end 

@cast function train(
                    sequences::String,
                    labels::String,
                    states::String;
                    o::String="model.bson",
                    k::Int=4,
                    lr::Float64 = 0.1,
                    N::Int = 10,
                    )
    states = [state for state in readlines(states)]
    kmer_tuples = kmers(k)

    ids,seqs = read_seqs(sequences)
    Y = DataFrame(CSV.File(labels))
    Y = [i for i ∈ eachcol(Y)]

    training_data = zip(seqs,Y)
    transitions = transition_features(states)
    emissions = kmer_features(kmer_tuples,states)
    model = CRF(size(emissions,1),size(transitions,1))

    loss_history = Vector{Float32}(undef,N)

    for epoch ∈ ProgressBar(1:N)
            for (x,y) ∈ training_data
                x = sequence_to_kmers(x,k)
                input = [x[1:1000],y[1:1000],states,emissions,transitions]
                loss = train!(model,input...; 🐢 = lr)
                loss_history[epoch] = loss
            end 
    end 

    println("Saving loss history graph...")
    g = plot(loss_history,
            linewidth=4,
            grid=:none,
            label="Per Token NLL")
    ylabel!("Loss")
    xlabel!("Iteration")

    savefig(g,"./loss.pdf")


    println("Saving model")

    @save o model



end 
@cast predict(x) = x
end

@cast crf


@main

end # module
